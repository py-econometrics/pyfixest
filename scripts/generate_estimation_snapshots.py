"""Prepare the platform-local release numerical-contract cache."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Import the release wheel before adding the checkout to sys.path. The Pixi
# task uses ``python -P`` so the checkout cannot shadow this package just
# because the command is launched at repository root.
import pyfixest  # noqa: E402

_module_path = Path(pyfixest.__file__).resolve()
if _module_path.is_relative_to(ROOT / "pyfixest"):
    raise RuntimeError(
        "The generator imported pyfixest from this checkout, not the isolated release wheel."
    )

sys.path.insert(0, str(ROOT))

from tests._estimation_snapshot_cache import (  # noqa: E402
    COMPLETE_MARKER,
    SNAPSHOT_DIRECTORY,
    snapshot_fingerprint,
)
from tests._estimation_snapshot_contract import (  # noqa: E402
    DATA_SEED,
    NOBS,
    RELEASE_VERSION,
    SCHEMA_VERSION,
    build_cases,
    extract_snapshot,
    fit_case,
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _artifacts(output_dir: Path) -> dict[Path, dict[str, object]]:
    installed_version = importlib.metadata.version("pyfixest")
    if installed_version != RELEASE_VERSION:
        raise RuntimeError(
            f"Expected pyfixest=={RELEASE_VERSION}, found {installed_version}. "
            "Use an isolated interpreter containing the recorded release wheel."
        )
    cases = build_cases()
    snapshots: dict[str, dict[str, object]] = {}
    print(f"Generating {len(cases)} release snapshot cases...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for case in cases:
            estimator = str(case["estimator"])
            snapshots.setdefault(estimator, {})[str(case["id"])] = extract_snapshot(
                fit_case(case, release=True), estimator=estimator
            )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "baseline": {
            "pyfixest_version": installed_version,
            "pyfixest_distribution": "pyfixest",
            "pyfixest_module_location": "site-packages/pyfixest",
            "python_environment": "tests/snapshots/release/.pixi/envs/default",
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "architecture": platform.machine(),
            "numpy_version": importlib.metadata.version("numpy"),
            "pandas_version": importlib.metadata.version("pandas"),
            "scipy_version": importlib.metadata.version("scipy"),
        },
        "fingerprint": snapshot_fingerprint(),
        "generation_command": (
            "pixi run --locked --clean-env --manifest-path "
            "tests/snapshots/release/pixi.toml prepare"
        ),
        "data": {
            "base_seed": DATA_SEED,
            "nobs_before_missing_values": NOBS,
            "description": (
                "Deterministic NumPy generators provide linear, count, binary, IV, "
                "quantile, weight, and offset inputs for every case."
            ),
        },
        "case_ids": sorted(str(case["id"]) for case in cases),
        "snapshot_files": {
            estimator: f"{estimator}.json" for estimator in sorted(snapshots)
        },
    }
    artifacts: dict[Path, dict[str, object]] = {output_dir / "manifest.json": manifest}
    artifacts.update(
        {
            output_dir / f"{estimator}.json": {
                "schema_version": SCHEMA_VERSION,
                "estimator": estimator,
                "cases": values,
            }
            for estimator, values in snapshots.items()
        }
    )
    return artifacts


def _prepare(*, force: bool) -> Path:
    target = SNAPSHOT_DIRECTORY
    complete = target / COMPLETE_MARKER
    fingerprint = snapshot_fingerprint()
    if not force and target.exists() and not complete.exists():
        raise RuntimeError(f"Incomplete release snapshot cache: {target}")
    if not force and complete.exists() and complete.read_text().strip() == fingerprint:
        print(f"Release snapshot cache hit: {target.relative_to(ROOT)}")
        return target

    cache_root = target.parent
    cache_root.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=cache_root))
    try:
        for path, value in _artifacts(temporary).items():
            path.write_bytes(_json_bytes(value))
        (temporary / COMPLETE_MARKER).write_text(snapshot_fingerprint() + "\n")
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Prepared release snapshot cache: {target.relative_to(ROOT)}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="replace this platform's current cache"
    )
    args = parser.parse_args()
    _prepare(force=args.force)


if __name__ == "__main__":
    main()
