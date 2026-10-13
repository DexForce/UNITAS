from __future__ import annotations

import json
from pathlib import Path

import json_numpy
import numpy as np


def decode_arrays(text: str):
    def validate(value):
        if isinstance(value, dict):
            if "__numpy__" in value and "dtype" in value:
                if np.dtype(value["dtype"]).hasobject:
                    raise ValueError("Object arrays are not accepted")
            for item in value.values():
                validate(item)
        elif isinstance(value, list):
            for item in value:
                validate(item)

    validate(json.loads(text))
    return json_numpy.loads(text)


def save_observation(path: Path | None, *, domain: str, observation, instruction: str, history):
    if path is None or path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "domain": domain,
                "language_instruction": instruction,
                "observation": json_numpy.dumps(observation),
                "history": json_numpy.dumps(history),
            }
        )
        + "\n"
    )
