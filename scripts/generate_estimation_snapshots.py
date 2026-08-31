"""Generate or verify the release numerical-contract artifacts explicitly.

Run this script with an isolated interpreter containing exactly the recorded
released wheel. It refuses to write unless ``--write`` is passed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PLATFORM = ("Darwin", "arm64")

# Import the release wheel before adding any checkout-local helper to sys.path.
# The Pixi task uses ``python -P`` so the checkout cannot
# shadow this package just because the command is launched at repository root.
import pyfixest  # noqa: E402

_module_path = Path(pyfixest.__file__).resolve()
if _module_path.is_relative_to(ROOT / "pyfixest"):
    raise RuntimeError(
        "The generator imported pyfixest from this checkout, not the isolated release wheel."
    )

sys.path.insert(0, str(ROOT / "tests"))

from _estimation_snapshot_contract import (  # noqa: E402
    AUGMENTATION_SEED,
    DATA_SEED,
    NOBS,
    RELEASE_VERSION,
    SCHEMA_VERSION,
    SNAPSHOT_DIR,
    build_cases,
    extract_snapshot,
    fast_case_ids,
    fit_case,
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _artifacts() -> dict[Path, dict[str, object]]:
    if (platform.system(), platform.machine()) != CANONICAL_PLATFORM:
        raise RuntimeError(
            "Byte-for-byte snapshot regeneration is supported only on the "
            f"canonical {CANONICAL_PLATFORM[0]}-{CANONICAL_PLATFORM[1]} platform. "
            "Other locked platforms run the tolerance-aware contract tests."
        )
    installed_version = importlib.metadata.version("pyfixest")
    if installed_version != RELEASE_VERSION:
        raise RuntimeError(
            f"Expected pyfixest=={RELEASE_VERSION}, found {installed_version}. "
            "Use an isolated interpreter containing the recorded release wheel."
        )
    cases = build_cases()
    snapshots: dict[str, dict[str, object]] = {}
    for case in cases:
        estimator = str(case["estimator"])
        snapshots.setdefault(estimator, {})[str(case["id"])] = extract_snapshot(
            fit_case(case)
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "baseline": {
            "pyfixest_version": installed_version,
            "pyfixest_distribution": "pyfixest",
            "pyfixest_module_location": "site-packages/pyfixest",
            "python_environment": "tests/snapshots/release/.pixi/envs/default",
            "python_version": platform.python_version(),
            "canonical_platform": "Darwin-arm64",
            "numpy_version": importlib.metadata.version("numpy"),
            "pandas_version": importlib.metadata.version("pandas"),
            "scipy_version": importlib.metadata.version("scipy"),
        },
        "generation_command": "pixi run --manifest-path tests/snapshots/release/pixi.toml generate",
        "data": {
            "base_seed": DATA_SEED,
            "augmentation_seed": AUGMENTATION_SEED,
            "nobs_before_complete_case": NOBS,
            "description": (
                "Base data use a deterministic NumPy Generator; a separate deterministic "
                "augmentation stream supplies IV Z1/X_endog/Y_iv and positive integer "
                "fweights. f3 has seven missing rows for complete-case SSC paths."
            ),
        },
        "cases": cases,
        "snapshot_files": {
            estimator: f"{estimator}.json" for estimator in sorted(snapshots)
        },
        "fast_case_ids": sorted(fast_case_ids(cases)),
    }
    artifacts: dict[Path, dict[str, object]] = {
        SNAPSHOT_DIR / "manifest.json": manifest
    }
    artifacts.update(
        {
            SNAPSHOT_DIR / f"{estimator}.json": {
                "schema_version": SCHEMA_VERSION,
                "estimator": estimator,
                "cases": values,
            }
            for estimator, values in snapshots.items()
        }
    )
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--write", action="store_true", help="explicitly replace committed artifacts"
    )
    action.add_argument(
        "--check", action="store_true", help="verify regeneration would produce no diff"
    )
    args = parser.parse_args()
    artifacts = _artifacts()

    if args.write:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        for path, value in artifacts.items():
            path.write_bytes(_json_bytes(value))
        return

    changed = [
        path
        for path, value in artifacts.items()
        if not path.exists() or path.read_bytes() != _json_bytes(value)
    ]
    if changed:
        raise SystemExit(
            "Release snapshots differ: "
            + ", ".join(str(path.relative_to(ROOT)) for path in changed)
        )


if __name__ == "__main__":
    main()
