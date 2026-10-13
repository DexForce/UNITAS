from __future__ import annotations

import copy
import random
from contextlib import contextmanager

import numpy as np
import torch

from unitas.inference import sampling_mode_for_domain
from unitas.models.config import temporal_spec_for_domain
from unitas.inference.runtime import move_batch


class Policy:
    def __init__(
        self,
        model,
        args,
        device,
        *,
        domain: str,
        steps=10,
        execute_steps=0,
        seed=160484,
    ):
        self.model, self.args, self.device = (model, args, device)
        self.domain, self.steps, self.seed = (domain, steps, seed)
        self.sampling_mode = sampling_mode_for_domain(domain)
        self.sampling_generator = (
            torch.Generator(device="cpu") if self.sampling_mode == "episode_cpu" else None
        )
        self.spec = temporal_spec_for_domain(args, domain)
        horizon = int(self.spec["prediction_horizon"])
        self.execute_steps = execute_steps or horizon
        if not 1 <= self.execute_steps <= horizon:
            raise ValueError(f"execute_steps must be between 1 and {horizon}")
        from unitas.inference.adapters import create_observation_builder

        self.builder = create_observation_builder(
            domain, args, num_gripper_tokens=int(model.config.num_gripper_tokens), seed=seed
        )
        if domain == "libero":
            from unitas.inference.adapters.libero import LiberoEEFActionConverter

            self.converter = LiberoEEFActionConverter(
                max_translation_step_m=None, max_rotation_step_deg=None
            )
        elif domain == "robotwin":
            from unitas.inference.adapters.robotwin import DirectEEFActionConverter

            self.converter = DirectEEFActionConverter(
                max_translation_step_m=0.08,
                max_rotation_step_deg=45.0,
                gripper_binary_threshold=None,
            )
        else:
            self.converter = None
        if domain != "vlabench" and steps != int(model.config.mot_num_denoise_steps):
            raise ValueError("This checkpoint requires matching gripper and MoT denoising steps.")
        self.reset()

    def reset(self):
        self.builder.reset()
        self.replans = 0
        if self.sampling_generator is not None:
            self.sampling_generator.manual_seed(self.seed)
        elif self.sampling_mode == "episode_global":
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.seed)

    def observe(self, observation):
        self.builder.update_history(observation)

    @contextmanager
    def transaction(self):
        """Commit a request only if history updates, inference and output validation succeed."""
        state = {
            key: copy.deepcopy(value)
            for key, value in vars(self.builder).items()
            if key == "history" or key.startswith("last_")
        }
        converter_state = (
            {
                key: copy.deepcopy(value)
                for key, value in vars(self.converter).items()
                if key.startswith("last_")
            }
            if self.converter is not None
            else {}
        )
        replans = self.replans
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        generator_rng = (
            self.sampling_generator.get_state() if self.sampling_generator is not None else None
        )
        try:
            yield
        except BaseException:
            vars(self.builder).update(state)
            if self.converter is not None:
                vars(self.converter).update(converter_state)
            self.replans = replans
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            if generator_rng is not None:
                self.sampling_generator.set_state(generator_rng)
            raise

    @torch.inference_mode()
    def act(self, observation, instruction):
        cpu_batch = self.builder.build_batch(observation, instruction)
        if self.domain == "robotwin":
            parity = self.builder.eef_frame_parity_report(
                cpu_batch, observation, translation_tolerance_m=1e-05, rotation_tolerance_deg=0.05
            )
            if not parity["passed"]:
                raise ValueError("RoboTwin TCP and model frames do not agree.")
        batch = move_batch(cpu_batch, self.device)
        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            encoded = self.model.encode_batch(batch)
            if self.sampling_mode == "replan":
                torch.manual_seed(self.seed + self.replans)
            sampling = {}
            if self.sampling_generator is not None:
                sampling["sampling_generator"] = self.sampling_generator
            if self.domain == "vlabench":
                sampling["mot_action_steps"] = self.steps
            prediction = self.model.decoder.sample(
                **encoded.decoder_kwargs(),
                **self.model.sampling_inputs(batch),
                steps=self.steps,
                **sampling,
            )
        poses = prediction["pred_eef_poses"][0].float().cpu().numpy()
        opening = prediction["pred_eef_open"][0].float().cpu().numpy()
        if self.domain == "libero":
            current_open = np.clip(np.abs(observation["robot0_gripper_qpos"]).mean() / 0.04, 0, 1)
            actions = self.converter.trajectory_to_actions(
                poses,
                opening,
                world_from_robot_base=observation["world_from_robot_base"],
                current_tcp_world=observation["controller_tcp_world"],
                current_open_probability=float(current_open),
                chunk_size=self.execute_steps,
            ).actions
        elif self.domain == "robotwin":
            actions = self.converter.trajectory_to_actions(
                poses,
                opening,
                current_eef_poses_model=batch["history_eef_poses"][0, -1].cpu().numpy(),
                world_from_model=batch["__world_from_model__"][0].cpu().numpy(),
                observation=observation,
                chunk_size=self.execute_steps,
            )
            if not self.converter.last_calibration_report["passed"]:
                raise ValueError("RoboTwin TCP-to-planner calibration failed.")
        else:
            from unitas.inference.adapters.vlabench import vlabench_predictions_to_actions

            actions = vlabench_predictions_to_actions(
                poses,
                opening,
                world_from_robot_base=observation["world_from_robot_base"],
                chunk_size=self.execute_steps,
            )
        self.replans += 1
        return np.asarray(actions, dtype=np.float32)
