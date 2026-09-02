"""Platform-local cache location and fingerprint for release snapshots."""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# One fixed cache directory; drift inputs change the recorded fingerprint,
# not the path. The previous baseline is replaced in place so stale
# per-fingerprint caches never accumulate.
SNAPSHOT_DIRECTORY = ROOT / ".pixi" / "release-contract"
COMPLETE_MARKER = ".complete"
FINGERPRINT_INPUTS = (
    "tests/snapshots/release/pixi.toml",
    "tests/snapshots/release/pixi.lock",
    "tests/_estimation_snapshot_cache.py",
    # The contract module owns the release version, cases, data, and extraction.
    "tests/_estimation_snapshot_contract.py",
    "tests/_feols_test_cases.py",
    "scripts/generate_estimation_snapshots.py",
)


def snapshot_fingerprint() -> str:
    """Hash the locked baseline inputs and the current platform."""
    digest = hashlib.sha256(
        f"{platform.system().lower()}:{platform.machine().lower()}".encode()
    )
    for relative_path in FINGERPRINT_INPUTS:
        digest.update(relative_path.encode())
        digest.update((ROOT / relative_path).read_bytes())
    return digest.hexdigest()[:16]
