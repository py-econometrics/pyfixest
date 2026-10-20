"""Test fresh dependency resolutions without modifying a locked Pixi environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from packaging.requirements import Requirement

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # pytest provides tomli on Python 3.10

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ("formulaic", "narwhals", "numpy", "pandas", "scipy")
# pandas/narwhals floors are qualified by the dependency-floor PR (#1946).
# SciPy 1.6 cannot run with NumPy 1.25 or Python 3.10; 1.9.2 is the oldest
# compatible wheel. These combinations do not change the public requirements.
MINIMUMS = {
    "formulaic": "1.2.2",
    "narwhals": "1.17.0",
    "numpy": "1.25.2",
    "pandas": "1.5.0",
    "scipy": "1.9.2",
}
TESTS = (
    "tests/test_formulaic_compat.py",
    "tests/test_formula_parse.py",
    "tests/test_i.py",
    "tests/test_iv.py",
    "tests/test_predict_resid_fixef.py",
    "tests/test_api.py",
    "tests/test_errors.py",
    "tests/test_vs_r_fast.py::test_fast_against_fixest",
    "tests/test_vs_fixest.py::test_parsed_stage_expressions_against_fixest",
    "tests/test_others.py",
)


def requirements(lane: str) -> list[str]:
    """Combine the checkout's declared bounds with the lane's exact floors."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    declared = {
        requirement.name: requirement
        for text in project["project"]["dependencies"]
        if (requirement := Requirement(text)).name in DEPENDENCIES
    }
    selected = []
    for name in DEPENDENCIES:
        requirement = declared[name]
        if name == "formulaic" and lane == "main":
            selected.append(
                "formulaic @ git+https://github.com/matthewwardrop/formulaic.git@main"
            )
            continue
        if lane == "minimum" or (
            lane == "formulaic-minimum" and name in ("formulaic", "narwhals")
        ):
            selected.append(f"{requirement},=={MINIMUMS[name]}")
        else:
            selected.append(str(requirement))
    # Refresh the optional compiler as well so the newest NumPy is constrained
    # by a matching numba rather than the stale compiler in the locked prefix.
    selected.append("numba>=0.58.0")
    return selected


def validate(lane: str, overlay: Path) -> None:
    """Confirm imported versions, overlay ownership, source checkout and extension."""
    for text in requirements(lane):
        requirement = Requirement(text)
        module = importlib.import_module(requirement.name)
        version = importlib.metadata.version(requirement.name)
        location = Path(module.__file__).resolve()
        if not location.is_relative_to(overlay):
            raise RuntimeError(
                f"{requirement.name} was imported outside {overlay}: {location}"
            )
        if not requirement.specifier.contains(version, prereleases=True):
            raise RuntimeError(
                f"Imported {requirement.name} {version} does not satisfy {text}"
            )
        print(f"{requirement.name} {version}: {location}", flush=True)
        if requirement.url:
            distribution = importlib.metadata.distribution(requirement.name)
            direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
            print(f"Upstream source: {direct_url}", flush=True)
    for name in ("pyfixest", "pyfixest.core._core_impl"):
        module = importlib.import_module(name)
        location = Path(module.__file__).resolve()
        if not location.is_relative_to(ROOT / "pyfixest"):
            raise RuntimeError(f"{name} did not load from this checkout: {location}")
        print(f"{name}: {location}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "lane", choices=("minimum", "formulaic-minimum", "latest", "main")
    )
    parser.add_argument("--overlay", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.overlay is not None:
        validate(args.lane, args.overlay.resolve())
        env = os.environ.copy()
        env.setdefault("PYTEST_XDIST_AUTO_NUM_WORKERS", "4")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-rs",
                "-n",
                "auto",
                "--no-cov",
                "--disable-warnings",
                "-m",
                "not (extended or against_r_extended or hac)",
                *TESTS,
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        return
    with TemporaryDirectory(prefix="pyfixest-dependencies-") as target:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--target",
                target,
                *requirements(args.lane),
            ],
            check=True,
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((target, str(ROOT)))
        subprocess.run(
            [sys.executable, __file__, args.lane, "--overlay", target],
            cwd=ROOT,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
