"""Record this platform's pyfixest release baseline.

Run through the locked workspace in `tests/snapshots/release/`, which pins the
release wheel:

    pixi run --locked --clean-env \
        --manifest-path tests/snapshots/release/pixi.toml record

The recording is `tests/test_release_contract.py` itself, executed against the
pinned release, so the baseline and the comparison can never describe different
case matrices.
"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Import the release wheel before the checkout joins sys.path. Pytest prepends
# the rootdir once it starts collecting, so this import — under `python -P` —
# is what keeps `pyfixest` bound to site-packages for the whole run.
import pyfixest  # noqa: E402

if Path(pyfixest.__file__).resolve().is_relative_to(ROOT / "pyfixest"):
    raise RuntimeError(
        "Imported pyfixest from this checkout instead of the pinned release wheel."
    )

sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from tests._release_baseline import (  # noqa: E402
    CACHE_PATH,
    RELEASE_VERSION,
    fingerprint,
)


def main() -> int:
    installed = importlib.metadata.version("pyfixest")
    if installed != RELEASE_VERSION:
        raise RuntimeError(
            f"Expected pyfixest=={RELEASE_VERSION} but found {installed}."
        )
    if CACHE_PATH.exists():
        import gzip
        import json

        with gzip.open(CACHE_PATH, "rt") as handle:
            if json.load(handle).get("fingerprint") == fingerprint():
                print(f"Baseline is current: {CACHE_PATH.relative_to(ROOT)}")
                return 0

    print(f"Recording pyfixest {installed} from {pyfixest.__file__}")
    # No coverage plugin and no xdist in the release environment, and a single
    # process so the session finalizer writes one complete file.
    return pytest.main(
        [
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "--record-release-baseline",
            str(ROOT / "tests" / "test_release_contract.py"),
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
