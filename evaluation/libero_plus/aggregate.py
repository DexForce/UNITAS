#!/usr/bin/env python3
"""Aggregate resumable LIBERO-Plus episode JSON files."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from evaluation.results import write_json, read_result, protocol_id


DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _bucket() -> dict[str, int]:
    return {"successes": 0, "failures": 0, "errors": 0, "completed": 0}


def aggregate(
    output_dir: Path,
    libero_plus_root: Path,
    suites: tuple[str, ...] = DEFAULT_SUITES,
    categories: tuple[str, ...] | None = None,
    expected_override: int | None = None,
) -> dict[str, Any]:
    classification = json.loads(
        (libero_plus_root / "libero/libero/benchmark/task_classification.json").read_text(
            encoding="utf-8"
        )
    )
    unknown = set(suites) - set(classification)
    if unknown:
        raise ValueError(f"unknown suites: {sorted(unknown)}")
    protocol = json.loads((output_dir / "run_config.json").read_text())
    if protocol.get("benchmark") != "LIBERO-Plus" or "jobs" not in protocol:
        raise ValueError("A LIBERO-Plus run manifest is required; use a new output directory")
    if protocol.get("classification_sha256") != protocol_id(classification):
        raise ValueError("Benchmark classification differs from the recorded run")
    run_id = protocol_id(protocol)
    plan = {(job["suite"], job["task_id"]) for job in protocol["jobs"]}
    if not plan or len(plan) != len(protocol["jobs"]):
        raise ValueError("Run manifest has empty or duplicate task identities")
    for suite, task_id in plan:
        if suite not in classification or not 0 <= task_id < len(classification[suite]):
            raise ValueError("Run manifest contains a task outside the benchmark")
    selected = {
        (suite, task_id)
        for suite, task_id in plan
        if suite in suites
        and (categories is None or str(classification[suite][task_id]["category"]) in categories)
    }
    expected = len(selected)
    if not expected:
        raise ValueError("No planned tasks match the requested summary")
    if expected_override is not None and expected_override != expected:
        raise ValueError("--expected must equal the task count in the run manifest")
    by_suite: dict[str, dict[str, int]] = defaultdict(_bucket)
    by_category: dict[str, dict[str, int]] = defaultdict(_bucket)
    by_difficulty: dict[str, dict[str, int]] = defaultdict(_bucket)

    seen = set()
    for path in sorted((output_dir / "episodes").glob("*/*.json")):
        result = read_result(path, run_id)
        identity = (str(result["suite"]), int(result["task_id"]))
        if identity not in plan:
            raise ValueError(f"Result is outside the run manifest: {path}")
        if identity in seen:
            raise ValueError(f"Duplicate episode identity: {identity}")
        seen.add(identity)
        metadata = classification[identity[0]][identity[1]]
        if result["task_name"] != metadata["name"] or result["category"] != metadata["category"]:
            raise ValueError(f"Episode metadata differs from the benchmark: {path}")
        if result.get("status") not in {"ok", "error"}:
            raise ValueError(f"Invalid episode status: {path}")
        if identity not in selected:
            continue
        targets = (
            by_suite[str(result["suite"])],
            by_category[str(result["category"])],
            by_difficulty[str(result.get("difficulty_level"))],
        )
        is_error = result.get("status") != "ok"
        is_success = bool(result.get("success", False)) and not is_error
        for bucket in targets:
            bucket["completed"] += 1
            if is_error:
                bucket["errors"] += 1
            elif is_success:
                bucket["successes"] += 1
            else:
                bucket["failures"] += 1

    completed = sum(bucket["completed"] for bucket in by_suite.values())
    successes = sum(bucket["successes"] for bucket in by_suite.values())
    errors = sum(bucket["errors"] for bucket in by_suite.values())

    def rates(groups: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
        answer = {}
        for key, value in sorted(groups.items()):
            valid = value["successes"] + value["failures"]
            answer[key] = {
                **value,
                "success_rate": value["successes"] / valid if valid else None,
            }
        return answer

    summary = {
        "protocol_id": run_id,
        "expected": expected,
        "completed": completed,
        "remaining": expected - completed,
        "complete": completed == expected,
        "planned_success_rate": successes / expected,
        "successes": successes,
        "errors": errors,
        "valid_success_rate": successes / (completed - errors) if completed > errors else None,
        "by_suite": rates(by_suite),
        "by_category": rates(by_category),
        "by_difficulty": rates(by_difficulty),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--libero-plus-root", type=Path, required=True)
    parser.add_argument(
        "--suites",
        default=",".join(DEFAULT_SUITES),
        help="Comma-separated suites included in the expected result count",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument(
        "--categories",
        default=None,
        help="Optional comma-separated exact task-classification categories",
    )
    parser.add_argument(
        "--expected",
        type=int,
        default=None,
        help="Check the expected count against the recorded task manifest",
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    suites = tuple(name.strip() for name in args.suites.split(",") if name.strip())
    categories = (
        tuple(value.strip() for value in args.categories.split(",") if value.strip())
        if args.categories is not None
        else None
    )
    while True:
        summary = aggregate(
            output,
            Path(args.libero_plus_root),
            suites,
            categories,
            args.expected,
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if not args.watch or summary["remaining"] <= 0:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
