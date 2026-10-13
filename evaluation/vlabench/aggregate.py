"""Summarize fixed VLABench episodes without hiding environment errors."""

import argparse
import json
from pathlib import Path

from evaluation.results import write_json
from evaluation.vlabench.configs import CONFIG_DIR, load_configs


def aggregate(output_dir: Path, configs: dict | None = None) -> dict:
    configs = load_configs() if configs is None else configs
    tracks = {}
    for track, tasks in configs.items():
        expected = completed = successes = errors = 0
        by_task = {}
        for task, episodes in tasks.items():
            count = valid = succeeded = failed_environment = 0
            expected += len(episodes)
            for index in range(len(episodes)):
                path = output_dir / track / task / f"episode_{index:03d}.json"
                if not path.exists():
                    continue
                result = json.loads(path.read_text())
                if (result.get("track"), result.get("task"), result.get("episode")) != (
                    track,
                    task,
                    index,
                ):
                    raise ValueError(f"Mismatched episode identity: {path}")
                count += 1
                if result.get("error"):
                    failed_environment += 1
                else:
                    valid += 1
                    succeeded += int(bool(result["success"]))
            completed += count
            successes += succeeded
            errors += failed_environment
            by_task[task] = {
                "expected": len(episodes),
                "completed": count,
                "valid": valid,
                "successes": succeeded,
                "errors": failed_environment,
                "success_rate_planned": succeeded / len(episodes),
                "success_rate_valid": succeeded / valid if valid else None,
            }
        valid = completed - errors
        tracks[track] = {
            "expected": expected,
            "completed": completed,
            "valid": valid,
            "successes": successes,
            "errors": errors,
            "complete": completed == expected,
            "success_rate_planned": successes / expected,
            "success_rate_valid": successes / valid if valid else None,
            "tasks": by_task,
        }
    expected = sum(t["expected"] for t in tracks.values())
    completed = sum(t["completed"] for t in tracks.values())
    successes = sum(t["successes"] for t in tracks.values())
    errors = sum(t["errors"] for t in tracks.values())
    valid = completed - errors
    summary = {
        "expected": expected,
        "completed": completed,
        "valid": valid,
        "successes": successes,
        "errors": errors,
        "complete": completed == expected,
        "success_rate_planned": successes / expected,
        "success_rate_valid": successes / valid if valid else None,
        "five_track_mean_planned": sum(t["success_rate_planned"] for t in tracks.values())
        / len(tracks),
        "five_track_mean_valid": (
            sum(t["success_rate_valid"] for t in tracks.values()) / len(tracks)
            if all(t["success_rate_valid"] is not None for t in tracks.values())
            else None
        ),
        "tracks": tracks,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.output_dir, load_configs(args.config_dir)), indent=2))


if __name__ == "__main__":
    main()
