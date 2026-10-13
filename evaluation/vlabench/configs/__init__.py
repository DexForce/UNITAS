"""The five fixed tracks used for UNITAS VLABench evaluation."""

import hashlib
import json
from pathlib import Path


CONFIG_DIR = Path(__file__).parent
TRACKS = (
    "track_1_in_distribution_new",
    "track_2_cross_category",
    "track_3_common_sense_new",
    "track_4_semantic_instruction_new",
    "track_6_unseen_texture_new",
)


def load_configs(config_dir: Path = CONFIG_DIR) -> dict:
    manifest = json.loads((CONFIG_DIR / "manifest.json").read_text())
    tracks = {}
    for name in TRACKS:
        path = config_dir / "tracks" / f"{name}.json"
        raw = path.read_bytes()
        expected = manifest["tracks"][name]
        if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError(f"{name} does not match the released evaluation configuration")
        tracks[name] = json.loads(raw)
        if {task: len(episodes) for task, episodes in tracks[name].items()} != expected["tasks"]:
            raise ValueError(f"Unexpected episode identities in {name}")
    return tracks
