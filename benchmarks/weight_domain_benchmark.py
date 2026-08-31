#!/usr/bin/env python3
"""Benchmark retained estimator state and temporary weight-domain work.

The default ``smoke`` profile is intentionally modest. Use ``--profile large``
for the review gate described in the estimator-state refactor. The harness
times public workflows, measures incremental allocations with ``tracemalloc``,
and counts unique NumPy buffers retained directly by the fitted result and its
dataclass state.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Literal

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1

Estimator = Literal["ols", "iv", "poisson"]
Operation = Literal["fit", "vcov"]
Tier = Literal["common", "inference"]


@dataclass(frozen=True, slots=True)
class BenchmarkScenario:
    """One deterministic public estimation or inference workflow."""

    name: str
    estimator: Estimator
    operation: Operation
    tier: Tier
    nobs: int
    n_covariates: int
    weighted: bool
    fixed_effects: bool
    vcov: str | dict[str, str] = "iid"
    repeated_vcov_calls: int = 1


@dataclass(frozen=True, slots=True)
class Measurement:
    """One measured workflow execution."""

    wall_time_seconds: float
    traced_peak_bytes: int
    retained_ndarray_bytes: int


def _profile_sizes(profile: str) -> dict[str, tuple[int, int]]:
    """Return deterministic ``(nobs, n_covariates)`` sizes for a profile."""
    if profile == "smoke":
        return {
            "narrow": (10_000, 5),
            "wide": (5_000, 20),
            "iv": (10_000, 3),
            "poisson": (5_000, 3),
            "inference": (10_000, 5),
        }
    if profile == "large":
        return {
            "narrow": (1_000_000, 10),
            "wide": (100_000, 100),
            "iv": (250_000, 8),
            "poisson": (100_000, 5),
            "inference": (250_000, 10),
        }
    raise ValueError(f"Unknown benchmark profile: {profile!r}")


def build_scenarios(profile: str) -> tuple[BenchmarkScenario, ...]:
    """Build the fixed scenario matrix for ``profile``."""
    sizes = _profile_sizes(profile)
    narrow_n, narrow_k = sizes["narrow"]
    wide_n, wide_k = sizes["wide"]
    iv_n, iv_k = sizes["iv"]
    poisson_n, poisson_k = sizes["poisson"]
    inference_n, inference_k = sizes["inference"]

    scenarios = [
        BenchmarkScenario(
            name="ols-unweighted-narrow",
            estimator="ols",
            operation="fit",
            tier="common",
            nobs=narrow_n,
            n_covariates=narrow_k,
            weighted=False,
            fixed_effects=False,
        ),
        BenchmarkScenario(
            name="ols-aweights-narrow",
            estimator="ols",
            operation="fit",
            tier="common",
            nobs=narrow_n,
            n_covariates=narrow_k,
            weighted=True,
            fixed_effects=False,
        ),
        BenchmarkScenario(
            name="ols-aweights-fe-narrow",
            estimator="ols",
            operation="fit",
            tier="common",
            nobs=narrow_n,
            n_covariates=narrow_k,
            weighted=True,
            fixed_effects=True,
        ),
        BenchmarkScenario(
            name="ols-aweights-wide",
            estimator="ols",
            operation="fit",
            tier="common",
            nobs=wide_n,
            n_covariates=wide_k,
            weighted=True,
            fixed_effects=False,
        ),
        BenchmarkScenario(
            name="iv-aweights-narrow",
            estimator="iv",
            operation="fit",
            tier="common",
            nobs=iv_n,
            n_covariates=iv_k,
            weighted=True,
            fixed_effects=False,
        ),
        BenchmarkScenario(
            name="poisson-aweights-fe",
            estimator="poisson",
            operation="fit",
            tier="common",
            nobs=poisson_n,
            n_covariates=poisson_k,
            weighted=True,
            fixed_effects=True,
        ),
    ]

    inference_specs: tuple[tuple[str, str | dict[str, str], Tier], ...] = (
        ("hc1", "HC1", "common"),
        ("hc2", "HC2", "inference"),
        ("hc3", "HC3", "inference"),
        ("crv1", {"CRV1": "cluster"}, "common"),
        ("crv3", {"CRV3": "cluster"}, "inference"),
    )
    scenarios.extend(
        BenchmarkScenario(
            name=f"vcov-{name}",
            estimator="ols",
            operation="vcov",
            tier=tier,
            nobs=inference_n,
            n_covariates=inference_k,
            weighted=True,
            fixed_effects=False,
            vcov=vcov,
            repeated_vcov_calls=5,
        )
        for name, vcov, tier in inference_specs
    )
    return tuple(scenarios)


def _build_data(scenario: BenchmarkScenario, seed: int) -> pd.DataFrame:
    """Generate one deterministic data frame outside the timed region."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(scenario.nobs, scenario.n_covariates))
    xnames = [f"x{index}" for index in range(scenario.n_covariates)]
    data = pd.DataFrame(X, columns=xnames)

    beta = np.linspace(0.2, 0.8, scenario.n_covariates)
    f1 = rng.integers(0, max(20, scenario.nobs // 100), size=scenario.nobs)
    f2 = rng.integers(0, 25, size=scenario.nobs)
    fixed_effect_signal = 0.02 * f1 + 0.05 * f2
    noise = rng.normal(size=scenario.nobs)

    data["w"] = 0.5 + rng.random(scenario.nobs)
    data["f1"] = f1
    data["f2"] = f2
    data["cluster"] = rng.integers(0, 100, size=scenario.nobs)

    linear_predictor = X @ beta + fixed_effect_signal
    if scenario.estimator == "iv":
        instrument = rng.normal(size=scenario.nobs)
        endogeneity = rng.normal(size=scenario.nobs)
        endogenous = (
            instrument + 0.5 * endogeneity + rng.normal(scale=0.25, size=scenario.nobs)
        )
        data["z"] = instrument
        data["endog"] = endogenous
        data["y"] = linear_predictor + 0.75 * endogenous + endogeneity + noise
    elif scenario.estimator == "poisson":
        mean = np.exp(np.clip(-0.5 + 0.05 * linear_predictor, -3.0, 3.0))
        data["y"] = rng.poisson(mean)
    else:
        data["y"] = linear_predictor + noise
    return data


def _formula(scenario: BenchmarkScenario) -> str:
    """Construct the public formula for ``scenario``."""
    covariates = " + ".join(f"x{index}" for index in range(scenario.n_covariates))
    main = f"y ~ {covariates}"
    fixed_effects = "f1 + f2" if scenario.fixed_effects else None
    if scenario.estimator == "iv":
        return (
            f"{main} | {fixed_effects} | endog ~ z"
            if fixed_effects is not None
            else f"{main} | endog ~ z"
        )
    return f"{main} | {fixed_effects}" if fixed_effects is not None else main


def _fit(scenario: BenchmarkScenario, data: pd.DataFrame) -> Any:
    """Fit one scenario through the public pyfixest API."""
    import pyfixest as pf

    kwargs: dict[str, Any] = {
        "data": data,
        "vcov": "iid",
    }
    if scenario.weighted:
        kwargs.update(weights="w", weights_type="aweights")
    formula = _formula(scenario)
    if scenario.estimator == "poisson":
        return pf.fepois(
            formula,
            iwls_tol=1e-8,
            iwls_maxiter=100,
            **kwargs,
        )
    return pf.feols(formula, **kwargs)


def retained_ndarray_bytes(value: object) -> int:
    """Count bytes held by unique NumPy buffers in supported containers.

    The fitted model is passed as its ``vars()`` mapping, so arbitrary nested
    objects such as pandas frames and cache services are intentionally not
    traversed. Frozen estimator-state dataclasses are traversed recursively.
    Aliases and views backed by the same allocation are counted once.
    """
    seen_objects: set[int] = set()
    seen_buffers: set[int] = set()

    def buffer_owner(array: np.ndarray) -> tuple[object, int]:
        owner: object = array
        base: object | None = array.base
        while isinstance(base, np.ndarray):
            owner = base
            base = base.base
        if base is not None:
            owner = base
        while isinstance(owner, memoryview):
            owner = owner.obj
        try:
            buffer_bytes = memoryview(owner).nbytes
        except TypeError:
            buffer_bytes = array.nbytes
        return owner, int(buffer_bytes)

    def walk(item: object) -> int:
        object_id = id(item)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)

        if isinstance(item, np.ndarray):
            owner, buffer_bytes = buffer_owner(item)
            buffer_id = id(owner)
            if buffer_id in seen_buffers:
                return 0
            seen_buffers.add(buffer_id)
            return buffer_bytes
        if is_dataclass(item) and not isinstance(item, type):
            return sum(walk(getattr(item, field.name)) for field in fields(item))
        if isinstance(item, Mapping):
            return sum(walk(mapped_value) for mapped_value in item.values())
        if isinstance(item, (tuple, list, set, frozenset)):
            return sum(walk(element) for element in item)
        return 0

    return walk(value)


def _measure_once(
    scenario: BenchmarkScenario,
    data: pd.DataFrame,
) -> Measurement:
    """Measure one fit or repeated-vcov workflow."""
    fit = None
    if scenario.operation == "vcov":
        fit = _fit(scenario, data)

    gc.collect()
    tracemalloc.start(1)
    try:
        start = time.perf_counter()
        if scenario.operation == "fit":
            fit = _fit(scenario, data)
        else:
            assert fit is not None
            for _ in range(scenario.repeated_vcov_calls):
                fit.vcov(scenario.vcov)
        wall_time = time.perf_counter() - start
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert fit is not None
    retained_bytes = retained_ndarray_bytes(vars(fit))
    return Measurement(
        wall_time_seconds=wall_time,
        traced_peak_bytes=int(peak_bytes),
        retained_ndarray_bytes=retained_bytes,
    )


def _warm_up(scenario: BenchmarkScenario, data: pd.DataFrame) -> None:
    """Warm imports, compiled kernels, and estimator dispatch for a scenario."""
    fit = _fit(scenario, data)
    if scenario.operation == "vcov":
        fit.vcov(scenario.vcov)
    del fit
    gc.collect()


def summarize_measurements(measurements: Sequence[Measurement]) -> dict[str, float]:
    """Return median wall-time and memory metrics."""
    if not measurements:
        raise ValueError("At least one measurement is required.")
    return {
        "median_wall_time_seconds": median(
            measurement.wall_time_seconds for measurement in measurements
        ),
        "median_traced_peak_bytes": median(
            measurement.traced_peak_bytes for measurement in measurements
        ),
        "median_retained_ndarray_bytes": median(
            measurement.retained_ndarray_bytes for measurement in measurements
        ),
    }


def _git_revision() -> str | None:
    """Read the current Git revision without requiring Git metadata."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run_benchmarks(
    scenarios: Iterable[BenchmarkScenario],
    *,
    profile: str,
    repetitions: int,
    warmups: int,
    seed: int,
) -> dict[str, Any]:
    """Run scenarios and return a JSON-serializable benchmark document."""
    import pyfixest as pf

    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    if warmups < 0:
        raise ValueError("warmups must be non-negative")

    scenario_results: dict[str, Any] = {}
    for scenario_index, scenario in enumerate(scenarios):
        data = _build_data(scenario, seed + scenario_index)
        for _ in range(warmups):
            _warm_up(scenario, data)
        measurements = [_measure_once(scenario, data) for _ in range(repetitions)]
        scenario_results[scenario.name] = {
            "tier": scenario.tier,
            "config": asdict(scenario),
            "samples": [asdict(measurement) for measurement in measurements],
            "summary": summarize_measurements(measurements),
        }
        del data
        gc.collect()

    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "profile": profile,
            "repetitions": repetitions,
            "warmups": warmups,
            "seed": seed,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pyfixest": pf.__version__,
            "git_revision": _git_revision(),
            "peak_memory_metric": "incremental tracemalloc peak during operation",
            "retained_memory_metric": (
                "unique NumPy buffers directly retained by fitted result/container state"
            ),
        },
        "scenarios": scenario_results,
    }


def _relative_change(base: float, head: float) -> float | None:
    """Return fractional change, or ``None`` for a zero baseline."""
    if base == 0:
        return 0.0 if head == 0 else None
    return (head - base) / base


def compare_benchmarks(
    base: Mapping[str, Any],
    head: Mapping[str, Any],
    *,
    common_limit: float,
    inference_limit: float,
    retained_growth_bytes: int,
) -> dict[str, Any]:
    """Compare two benchmark documents and apply the review thresholds."""
    if common_limit < 0 or inference_limit < 0:
        raise ValueError("Wall-time limits must be non-negative.")
    if retained_growth_bytes < 0:
        raise ValueError("retained_growth_bytes must be non-negative.")
    if base.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("The base benchmark has an unsupported schema version.")
    if head.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("The head benchmark has an unsupported schema version.")

    comparison_settings = (
        "profile",
        "repetitions",
        "warmups",
        "seed",
        "python",
        "platform",
        "pyfixest",
    )
    base_metadata = base.get("metadata", {})
    head_metadata = head.get("metadata", {})
    for setting in comparison_settings:
        if base_metadata.get(setting) != head_metadata.get(setting):
            raise ValueError(f"Benchmark metadata differs for {setting!r}.")

    base_scenarios = base.get("scenarios", {})
    head_scenarios = head.get("scenarios", {})
    if set(base_scenarios) != set(head_scenarios):
        missing_from_head = sorted(set(base_scenarios) - set(head_scenarios))
        missing_from_base = sorted(set(head_scenarios) - set(base_scenarios))
        raise ValueError(
            "Benchmark scenarios differ: "
            f"missing from head={missing_from_head}, "
            f"missing from base={missing_from_base}."
        )

    comparisons: list[dict[str, Any]] = []
    for name in sorted(base_scenarios):
        base_scenario = base_scenarios[name]
        head_scenario = head_scenarios[name]
        if base_scenario["config"] != head_scenario["config"]:
            raise ValueError(f"Scenario configuration differs for {name!r}.")

        tier = base_scenario["tier"]
        if tier not in ("common", "inference"):
            raise ValueError(f"Unknown benchmark tier {tier!r} for {name!r}.")
        wall_limit = common_limit if tier == "common" else inference_limit
        base_summary = base_scenario["summary"]
        head_summary = head_scenario["summary"]
        base_wall = float(base_summary["median_wall_time_seconds"])
        head_wall = float(head_summary["median_wall_time_seconds"])
        base_peak = float(base_summary["median_traced_peak_bytes"])
        head_peak = float(head_summary["median_traced_peak_bytes"])
        base_retained = int(base_summary["median_retained_ndarray_bytes"])
        head_retained = int(head_summary["median_retained_ndarray_bytes"])

        wall_passed = head_wall <= base_wall * (1 + wall_limit)
        retained_passed = head_retained <= base_retained + retained_growth_bytes
        comparisons.append(
            {
                "name": name,
                "tier": tier,
                "wall_limit_fraction": wall_limit,
                "base_wall_time_seconds": base_wall,
                "head_wall_time_seconds": head_wall,
                "wall_change_fraction": _relative_change(base_wall, head_wall),
                "base_traced_peak_bytes": base_peak,
                "head_traced_peak_bytes": head_peak,
                "traced_peak_change_fraction": _relative_change(base_peak, head_peak),
                "base_retained_ndarray_bytes": base_retained,
                "head_retained_ndarray_bytes": head_retained,
                "retained_ndarray_change_bytes": head_retained - base_retained,
                "wall_passed": wall_passed,
                "retained_passed": retained_passed,
                "passed": wall_passed and retained_passed,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "limits": {
            "common_wall_fraction": common_limit,
            "inference_wall_fraction": inference_limit,
            "retained_growth_bytes": retained_growth_bytes,
            "peak_memory": "reported but not gated",
        },
        "passed": all(comparison["passed"] for comparison in comparisons),
        "comparisons": comparisons,
    }


def _write_json(document: Mapping[str, Any], output: Path | None) -> None:
    """Write a JSON document to a path or standard output."""
    rendered = json.dumps(document, indent=2, sort_keys=True)
    if output is None:
        print(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{rendered}\n", encoding="utf-8")


def _format_mebibytes(value: float) -> str:
    return f"{value / (1024**2):.1f}"


def _print_run_summary(document: Mapping[str, Any]) -> None:
    """Print a compact human-readable run summary to standard error."""
    print(
        "scenario                              wall(s)  peak(MiB)  retained(MiB)",
        file=sys.stderr,
    )
    for name, result in document["scenarios"].items():
        summary = result["summary"]
        print(
            f"{name:<37} "
            f"{summary['median_wall_time_seconds']:>7.3f}  "
            f"{_format_mebibytes(summary['median_traced_peak_bytes']):>9}  "
            f"{_format_mebibytes(summary['median_retained_ndarray_bytes']):>13}",
            file=sys.stderr,
        )


def _format_change(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1%}"


def _print_comparison_summary(document: Mapping[str, Any]) -> None:
    """Print a compact comparison summary to standard error."""
    print(
        "scenario                              wall change  peak change  arrays     status",
        file=sys.stderr,
    )
    for result in document["comparisons"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"{result['name']:<37} "
            f"{_format_change(result['wall_change_fraction']):>11}  "
            f"{_format_change(result['traced_peak_change_fraction']):>11}  "
            f"{result['retained_ndarray_change_bytes'] / (1024**2):>+7.1f} MiB  "
            f"{status}",
            file=sys.stderr,
        )


def _parser() -> argparse.ArgumentParser:
    """Build the benchmark command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a benchmark profile")
    run_parser.add_argument(
        "--profile",
        choices=("smoke", "large"),
        default="smoke",
        help="safe smoke sizes (default) or the explicit review-gate sizes",
    )
    run_parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="run only this named scenario; repeat the option to select several",
    )
    run_parser.add_argument("--repetitions", type=int)
    run_parser.add_argument("--warmups", type=int, default=1)
    run_parser.add_argument("--seed", type=int, default=20260831)
    run_parser.add_argument("--output", type=Path)

    compare_parser = subparsers.add_parser(
        "compare", help="compare base and head benchmark JSON"
    )
    compare_parser.add_argument("base", type=Path)
    compare_parser.add_argument("head", type=Path)
    compare_parser.add_argument("--common-limit", type=float, default=0.05)
    compare_parser.add_argument("--inference-limit", type=float, default=0.10)
    compare_parser.add_argument(
        "--retained-growth-bytes",
        type=int,
        default=0,
        help="per-scenario retained NumPy growth allowance (default: 0)",
    )
    compare_parser.add_argument("--output", type=Path)
    return parser


def _selected_scenarios(
    profile: str, selected_names: Sequence[str] | None
) -> tuple[BenchmarkScenario, ...]:
    """Select requested scenarios and reject unknown names."""
    scenarios = build_scenarios(profile)
    if not selected_names:
        return scenarios
    by_name = {scenario.name: scenario for scenario in scenarios}
    unknown = sorted(set(selected_names) - set(by_name))
    if unknown:
        raise ValueError(
            f"Unknown scenarios {unknown}. Available scenarios: {sorted(by_name)}"
        )
    return tuple(by_name[name] for name in selected_names)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line benchmark or comparison workflow."""
    args = _parser().parse_args(argv)
    if args.command == "run":
        repetitions = args.repetitions
        if repetitions is None:
            repetitions = 3 if args.profile == "smoke" else 5
        document = run_benchmarks(
            _selected_scenarios(args.profile, args.scenarios),
            profile=args.profile,
            repetitions=repetitions,
            warmups=args.warmups,
            seed=args.seed,
        )
        _print_run_summary(document)
        _write_json(document, args.output)
        return 0

    base = json.loads(args.base.read_text(encoding="utf-8"))
    head = json.loads(args.head.read_text(encoding="utf-8"))
    comparison = compare_benchmarks(
        base,
        head,
        common_limit=args.common_limit,
        inference_limit=args.inference_limit,
        retained_growth_bytes=args.retained_growth_bytes,
    )
    _print_comparison_summary(comparison)
    _write_json(comparison, args.output)
    return 0 if comparison["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
