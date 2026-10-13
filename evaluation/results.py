"""Atomic writes for resumable evaluation output."""

import json
import hashlib
import os
import tempfile
from pathlib import Path


def protocol_id(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def record_protocol(path: Path, value, *, result_glob: str | None = None) -> str:
    """Keep concurrent shards from mixing different evaluation settings."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(f".{path.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists() and result_glob and next(path.parent.glob(result_glob), None):
            raise ValueError("Results without a run protocol exist; use a new output directory")
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(
                "Output directory contains a different evaluation protocol; use a new directory"
            )
        write_json(path, value)
    return protocol_id(value)


def read_result(path: Path, run_id: str, **identity) -> dict:
    """Reject records from another run or episode before resuming or aggregating."""
    value = json.loads(path.read_text())
    if value.get("protocol_id") != run_id:
        raise ValueError(f"Result belongs to a different or unidentified protocol: {path}")
    if any(value.get(key) != expected for key, expected in identity.items()):
        raise ValueError(f"Mismatched result identity: {path}")
    if type(value.get("success")) is not bool:
        raise ValueError(f"Invalid success value: {path}")
    return value
