from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def predict_scene(
    model, batch, xyz0, *, steps=10, chunk_size=3000, policy_seed=42, scene_seed=210921
):
    import torch

    if steps < 1 or chunk_size < 1 or chunk_size > 3000:
        raise ValueError("steps must be positive and chunk_size must be between 1 and 3000")
    device = batch["history_robot_flows"].device
    horizon = int(batch["scene_flow_horizon"].item())
    trajectory = np.empty((horizon + 1, len(xyz0), 3), dtype=np.float32)
    trajectory[0] = xyz0
    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"),
    ):
        encoded = model.encode_batch(batch)
        kwargs = model.sampling_inputs(batch)
        torch.manual_seed(policy_seed)
        generated = model.decoder.sample(
            **encoded.decoder_kwargs(), **kwargs, steps=steps, mot_action_steps=steps
        )
        mask = generated["gripper_token_mask"][0].bool()
        initial_gripper = generated["gripper_xyz0"]
        scene_kwargs = {
            key: kwargs[key]
            for key in (
                "history_gripper_flows",
                "gripper_mask",
                "gripper_flow_fps",
                "gripper_flow_horizon",
            )
        }
        condition = model.decoder.prepare_scene_condition(
            **encoded.decoder_kwargs(),
            **scene_kwargs,
            clean_gripper_latent=generated["pred_gripper_latent_fm"],
        )
        order = np.random.default_rng(scene_seed).permutation(len(xyz0))
        for index, start in enumerate(range(0, len(xyz0), chunk_size)):
            ids = order[start : start + chunk_size]
            queries = torch.as_tensor(xyz0[ids], device=device)[None]
            torch.manual_seed(scene_seed + index)
            scene = model.decoder.sample_scene_flow_from_gripper(
                **encoded.decoder_kwargs(),
                **scene_kwargs,
                scene_xyz0=queries,
                scene_mask=torch.ones(queries.shape[:2], device=device, dtype=torch.bool),
                scene_flow_fps=batch["scene_flow_fps"],
                scene_flow_horizon=batch["scene_flow_horizon"],
                clean_gripper_latent=generated["pred_gripper_latent_fm"],
                scene_steps=steps,
                condition=condition,
            )
            active = scene["scene_token_mask"][0].bool()
            selected = scene["scene_selected_indices"][0, active].cpu().numpy()
            if not np.array_equal(selected, np.arange(len(ids))):
                raise RuntimeError("Dense inference must retain every query point")
            trajectory[1:, ids] = scene["pred_scene_future_xyz"][0, :, active].float().cpu().numpy()
        gripper = (
            torch.cat([initial_gripper[:, None], generated["pred_future_xyz"]], dim=1)[0, :, mask]
            .float()
            .cpu()
            .numpy()
        )
    if not np.isfinite(trajectory).all() or not np.isfinite(gripper).all():
        raise RuntimeError("Non-finite predicted trajectories")
    return {
        "scene": trajectory,
        "gripper": gripper,
        "eef_poses": generated["pred_eef_poses"][0].float().cpu().numpy(),
        "eef_open": generated["pred_eef_open"][0].float().cpu().numpy(),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict action-conditioned dense scene flow.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--florence-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=3000)
    parser.add_argument("--policy-seed", type=int, default=42)
    parser.add_argument("--scene-seed", type=int, default=210921)
    args = parser.parse_args()
    from unitas.inference.runtime import load_model
    from unitas.scene_flow.io import load_input

    model, _, device = load_model(args.checkpoint, args.florence_dir, args.device)
    batch, xyz0, colors, domain, instruction = load_input(args.input, device)
    prediction = predict_scene(
        model,
        batch,
        xyz0,
        steps=args.steps,
        chunk_size=args.chunk_size,
        policy_seed=args.policy_seed,
        scene_seed=args.scene_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **prediction,
        colors=colors,
        domain=domain,
        instruction=instruction,
        fps=float(batch["scene_flow_fps"].item()),
        gripper_fps=float(batch["gripper_flow_fps"].item()),
        policy_seed=args.policy_seed,
        scene_seed=args.scene_seed,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()
