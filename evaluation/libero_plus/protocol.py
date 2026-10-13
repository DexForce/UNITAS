"""Deterministic task selection shared by all LIBERO-Plus shards."""

import random


def select_jobs(
    classification,
    suites,
    *,
    categories=None,
    task_ids=None,
    sample_tasks=0,
    sample_seed=160484,
    max_tasks=0,
    num_shards=1,
):
    if not suites or len(set(suites)) != len(suites):
        raise ValueError("Suites must be nonempty and unique")
    if set(suites) - classification.keys():
        raise ValueError("Unknown LIBERO-Plus suite")
    if min(sample_tasks, max_tasks) < 0 or num_shards < 1:
        raise ValueError("Task limits must be nonnegative and num_shards positive")
    jobs = [
        (suite, index, item) for suite in suites for index, item in enumerate(classification[suite])
    ]
    if categories is not None:
        available = {str(item["category"]) for _, _, item in jobs}
        if not categories or set(categories) - available:
            raise ValueError("Unknown or empty category selection")
        jobs = [job for job in jobs if str(job[2]["category"]) in categories]
    if task_ids is not None:
        if len(suites) != 1 or not task_ids:
            raise ValueError("Task IDs require one suite and a nonempty selection")
        if set(task_ids) - {job[1] for job in jobs}:
            raise ValueError("Task IDs outside the selected suite/categories")
        jobs = [job for job in jobs if job[1] in task_ids]
    if sample_tasks:
        if sample_tasks > len(jobs):
            raise ValueError("Sample size exceeds the selected task count")
        jobs = sorted(
            random.Random(sample_seed).sample(jobs, sample_tasks), key=lambda job: (job[0], job[1])
        )
    # Capping every interleaved shard is equivalent to capping this global list.
    if max_tasks:
        jobs = jobs[: max_tasks * num_shards]
    if not jobs:
        raise ValueError("No tasks selected")
    return jobs
