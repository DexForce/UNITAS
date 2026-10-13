from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from unitas.inference import ACTION_DIMENSIONS
from unitas.inference.observations import decode_arrays


class ActRequest(BaseModel):
    domain: str
    language_instruction: str
    observation: str
    history: str = "[]"


def create_app(policy, domain: str) -> FastAPI:
    app = FastAPI(title="UNITAS Policy")
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok", "domain": domain, "protocol": getattr(policy, "protocol", {})}

    @app.post("/reset")
    def reset():
        with lock:
            policy.reset()
        return {"ok": True}

    @app.post("/act")
    def act(request: ActRequest):
        if request.domain != domain:
            raise HTTPException(400, "Request domain does not match the loaded policy.")
        try:
            observation = decode_arrays(request.observation)
            history = decode_arrays(request.history)
            if not isinstance(observation, dict) or not isinstance(history, list):
                raise ValueError("observation must be an object and history must be a list")
            if not all((isinstance(frame, dict) for frame in history)):
                raise ValueError("history must contain state objects")
            with lock, policy.transaction():
                for frame in history:
                    policy.observe(frame)
                actions = np.asarray(
                    policy.act(observation, request.language_instruction), np.float32
                )
                width = ACTION_DIMENSIONS[domain]
                if (
                    actions.ndim != 2
                    or actions.shape[0] == 0
                    or actions.shape[1] != width
                    or (not np.isfinite(actions).all())
                ):
                    raise RuntimeError("Invalid predicted actions")
                return {"action": actions.tolist()}
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            logging.exception("Policy inference failed")
            raise HTTPException(500, "Policy inference failed.") from exc

    return app


def main():
    parser = argparse.ArgumentParser(description="Serve a UNITAS policy over HTTP.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--florence-dir", type=Path, required=True)
    parser.add_argument("--domain", choices=list(ACTION_DIMENSIONS), required=True)
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--execute-steps", type=int)
    parser.add_argument("--seed", type=int, default=160484)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.execute_steps is None:
        args.execute_steps = {"libero": 10, "vlabench": 5, "robotwin": 0}[args.domain]
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if args.assets_dir:
        import os

        os.environ["UNITAS_ASSETS_DIR"] = str(args.assets_dir.resolve())
    import uvicorn

    from unitas.inference.policy import Policy
    from unitas.inference.runtime import file_sha256, load_model, source_fingerprint

    model, saved_args, device = load_model(
        args.checkpoint,
        args.florence_dir,
        args.device,
        inference_kind=args.domain,
        seed=args.seed,
    )
    policy = Policy(
        model,
        saved_args,
        device,
        domain=args.domain,
        steps=args.steps,
        execute_steps=args.execute_steps,
        seed=args.seed,
    )
    policy.protocol = {
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "source_sha256": source_fingerprint(),
        "denoise_steps": args.steps,
        "execute_steps": policy.execute_steps,
        "sampling_seed": args.seed,
        "sampling_mode": policy.sampling_mode,
        "florence_dtype": saved_args.florence_dtype,
        "florence_attention": saved_args.florence_attn_implementation,
        "state_conditioning": saved_args.action_flow_mot_state_conditioning,
        "context_conditioning": saved_args.action_flow_context_conditioning,
        "temporal_spec": policy.spec,
    }
    if args.domain == "robotwin":
        policy.protocol["action_limits"] = {"translation_m": 0.08, "rotation_deg": 45.0}
    uvicorn.run(create_app(policy, args.domain), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
