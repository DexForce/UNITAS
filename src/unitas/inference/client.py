from __future__ import annotations

from typing import Any

import json_numpy
import numpy as np
import requests

from unitas.inference import ACTION_DIMENSIONS


class PolicyClient:
    def __init__(self, host: str, port: int, *, domain: str, timeout: float = 300):
        if domain not in ACTION_DIMENSIONS:
            raise ValueError(f"Unsupported policy domain: {domain}")
        self.base_url = f"http://{host}:{int(port)}"
        self.domain = domain
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def reset(self) -> None:
        response = self.session.post(f"{self.base_url}/reset", timeout=self.timeout)
        response.raise_for_status()

    def health(self) -> dict:
        response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def act(
        self, observation: dict[str, Any], instruction: str, history: list[dict[str, Any]]
    ) -> np.ndarray:
        response = self.session.post(
            f"{self.base_url}/act",
            json={
                "domain": self.domain,
                "language_instruction": instruction,
                "observation": json_numpy.dumps(observation),
                "history": json_numpy.dumps(history),
            },
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            try:
                detail = response.json().get(
                    "detail",
                    response.reason_phrase
                    if hasattr(response, "reason_phrase")
                    else response.reason,
                )
            except ValueError:
                detail = response.text
            raise requests.HTTPError(
                f"Policy server returned {response.status_code}: {detail}", response=response
            )
        response.raise_for_status()
        actions = np.asarray(response.json()["action"], dtype=np.float32)
        width = ACTION_DIMENSIONS[self.domain]
        if (
            actions.ndim != 2
            or actions.shape[0] == 0
            or actions.shape[1] != width
            or not np.isfinite(actions).all()
        ):
            raise ValueError(f"Invalid {self.domain} action chunk: {actions.shape}")
        return actions

    def close(self) -> None:
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
