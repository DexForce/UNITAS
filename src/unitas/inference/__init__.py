"""Model serving and observation preparation, independent of simulator SDKs."""

ACTION_DIMENSIONS = {"libero": 7, "robotwin": 16, "vlabench": 8}


def sampling_mode_for_domain(domain: str) -> str:
    return {
        "robotwin": "replan",
        "libero": "episode_cpu",
        "vlabench": "episode_global",
        "scene_flow": "replan",
    }[domain]
