from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from unitas.inference.observations import decode_arrays
from unitas.scene_flow.io import save_input


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a current observation for scene flow inference."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--query-frame", choices=["world", "model"], default="world")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path)
    args = parser.parse_args()
    if args.assets_dir:
        os.environ["UNITAS_ASSETS_DIR"] = str(args.assets_dir.resolve())
    import torch

    from unitas.models.config import temporal_spec_for_domain
    from unitas.inference.adapters import create_observation_builder
    from unitas.inference.runtime import validate_checkpoint_args

    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    raw = checkpoint["args"]
    saved = SimpleNamespace(**raw if isinstance(raw, dict) else vars(raw))
    del checkpoint
    payload = json.loads(args.observation.read_text())
    domain = payload["domain"]
    if domain not in {"libero", "robotwin", "vlabench"}:
        parser.error("The observation adapter supports libero, robotwin and vlabench")
    observation = decode_arrays(payload["observation"])
    history = decode_arrays(payload.get("history", "[]"))
    spec = temporal_spec_for_domain(saved, domain)
    validate_checkpoint_args(saved, domain)
    builder = create_observation_builder(domain, saved)
    for frame in history:
        builder.update_history(frame)
    batch = builder.build_batch(observation, payload["language_instruction"])
    batch["scene_flow_fps"] = torch.tensor([spec["scene_flow_fps"]], dtype=torch.float32)
    batch["scene_flow_horizon"] = torch.tensor([spec["scene_flow_horizon"]], dtype=torch.long)
    if args.queries is None:
        from unitas.scene_flow.io import backproject_queries

        xyz0, colors = backproject_queries(batch)
    else:
        with np.load(args.queries, allow_pickle=False) as queries:
            xyz0, colors = queries["xyz0"].copy(), queries["colors"].copy()
    if args.queries is not None and args.query_frame == "world":
        transform = batch.get("__world_from_model__")
        if transform is None:
            transform = observation["world_from_robot_base"]
        if hasattr(transform, "numpy"):
            transform = transform.numpy()
        transform = np.asarray(transform).reshape(4, 4)
        inverse = np.linalg.inv(transform)
        xyz0 = xyz0 @ inverse[:3, :3].T + inverse[:3, 3]
    save_input(
        args.output, batch, xyz0, colors, domain=domain, instruction=payload["language_instruction"]
    )


if __name__ == "__main__":
    main()
