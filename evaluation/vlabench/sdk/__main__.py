"""Apply the evaluation compatibility patch to the pinned VLABench checkout."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--check", action="store_true", help="Verify compatibility without modifying the SDK"
    )
    args = parser.parse_args()
    directory = Path(__file__).parent
    manifest = json.loads((directory / "manifest.json").read_text())
    needs_patch = False
    for name, hashes in manifest["files"].items():
        path = args.repo / "VLABench" / name
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        if current not in hashes.values():
            raise ValueError(
                f"{name} differs from the supported VLABench revision {manifest['revision']}"
            )
        if name == "tasks/dm_task.py" and current == hashes["base_sha256"]:
            needs_patch = True
    if needs_patch:
        patch = directory / "compat.patch"
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=args.repo, check=True)
        if args.check:
            raise SystemExit(
                "SDK revision verified; run again without --check to add check_success()"
            )
        subprocess.run(["git", "apply", str(patch)], cwd=args.repo, check=True)
    print("VLABench evaluation SDK is compatible.")


if __name__ == "__main__":
    main()
