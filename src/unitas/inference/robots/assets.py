from __future__ import annotations

import os
from pathlib import Path
from importlib.resources import files


def asset_root() -> Path:
    configured = os.environ.get("UNITAS_ASSETS_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(str(files("unitas.assets")))


def resolve_robot_urdf(domain: str) -> str:
    paths = {
        "libero": "libero_panda/urdf/panda_arm_hand.urdf",
        "vlabench": "libero_panda/urdf/panda_arm_hand.urdf",
        "robotwin": "robotwin_embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf",
    }
    if domain not in paths:
        raise ValueError(f"Unsupported robot domain: {domain}")
    return str(asset_root() / paths[domain])
