"""Construct the same observation adapter for policy and scene-flow inference."""

from unitas.models.config import as_bool, temporal_spec_for_domain


def create_observation_builder(domain, args, *, num_gripper_tokens=None, seed=160484):
    tokens = int(num_gripper_tokens or args.action_flow_num_gripper_tokens)
    if domain == "libero":
        from .libero import LiberoOnlineBatchBuilder

        return LiberoOnlineBatchBuilder(args, num_gripper_tokens=tokens, seed=seed)
    if domain == "vlabench":
        from .vlabench import VLABenchOnlineBatchBuilder

        return VLABenchOnlineBatchBuilder(args, num_gripper_tokens=tokens, seed=seed)
    if domain == "robotwin":
        from .robotwin import RobotWinOnlineBatchBuilder

        spec = temporal_spec_for_domain(args, domain)
        return RobotWinOnlineBatchBuilder(
            num_gripper_tokens_per_arm=tokens,
            prediction_horizon=spec["prediction_horizon"],
            history_horizon=spec["history_horizon"],
            flow_fps=spec["gripper_flow_fps"],
            image_size=int(getattr(args, "action_flow_vlm_image_size", 224)),
            coordinate_frame=str(getattr(args, "robotwin_coordinate_frame", "source")),
            disable_center_shift=as_bool(getattr(args, "disable_center_shift", False)),
            seed=int(getattr(args, "seed", 160484)),
        )
    raise ValueError(f"Unsupported policy domain: {domain}")
