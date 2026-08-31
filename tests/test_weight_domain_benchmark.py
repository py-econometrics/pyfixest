from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from benchmarks.weight_domain_benchmark import (
    Measurement,
    build_scenarios,
    compare_benchmarks,
    retained_ndarray_bytes,
    summarize_measurements,
)


@dataclass(frozen=True, slots=True)
class _ArrayState:
    primary: np.ndarray
    alias: np.ndarray
    other: tuple[np.ndarray, ...]


def _benchmark_document(
    *,
    common_wall: float,
    inference_wall: float,
    retained: int,
) -> dict[str, Any]:
    def scenario(name: str, tier: str, wall: float) -> dict[str, Any]:
        return {
            "tier": tier,
            "config": {"name": name},
            "summary": {
                "median_wall_time_seconds": wall,
                "median_traced_peak_bytes": 100,
                "median_retained_ndarray_bytes": retained,
            },
        }

    return {
        "schema_version": 1,
        "metadata": {
            "profile": "smoke",
            "repetitions": 3,
            "warmups": 1,
            "seed": 123,
            "python": "3.12.0",
            "platform": "test-platform",
            "pyfixest": "0.test",
        },
        "scenarios": {
            "common": scenario("common", "common", common_wall),
            "inference": scenario("inference", "inference", inference_wall),
        },
    }


def test_retained_ndarray_bytes_deduplicates_aliases() -> None:
    allocation = np.ones(12)
    primary = allocation.reshape((4, 3))
    alias = primary[:, :]
    other = np.ones(5)
    state = _ArrayState(primary=primary, alias=alias, other=(other,))

    assert retained_ndarray_bytes({"state": state, "again": primary}) == (
        allocation.nbytes + other.nbytes
    )


def test_summarize_measurements_uses_medians() -> None:
    measurements = [
        Measurement(3.0, 300, 30),
        Measurement(1.0, 100, 10),
        Measurement(2.0, 200, 20),
    ]

    assert summarize_measurements(measurements) == {
        "median_wall_time_seconds": 2.0,
        "median_traced_peak_bytes": 200,
        "median_retained_ndarray_bytes": 20,
    }


def test_compare_benchmarks_applies_tier_limits_and_retained_gate() -> None:
    base = _benchmark_document(common_wall=1.0, inference_wall=1.0, retained=100)
    head = _benchmark_document(common_wall=1.04, inference_wall=1.09, retained=100)

    comparison = compare_benchmarks(
        base,
        head,
        common_limit=0.05,
        inference_limit=0.10,
        retained_growth_bytes=0,
    )

    assert comparison["passed"]
    assert all(result["passed"] for result in comparison["comparisons"])

    head["scenarios"]["common"]["summary"]["median_retained_ndarray_bytes"] = 101
    comparison = compare_benchmarks(
        base,
        head,
        common_limit=0.05,
        inference_limit=0.10,
        retained_growth_bytes=0,
    )
    assert not comparison["passed"]


def test_compare_benchmarks_rejects_different_scenarios() -> None:
    base = _benchmark_document(common_wall=1.0, inference_wall=1.0, retained=100)
    head = _benchmark_document(common_wall=1.0, inference_wall=1.0, retained=100)
    del head["scenarios"]["inference"]

    with pytest.raises(ValueError, match="Benchmark scenarios differ"):
        compare_benchmarks(
            base,
            head,
            common_limit=0.05,
            inference_limit=0.10,
            retained_growth_bytes=0,
        )


def test_compare_benchmarks_rejects_different_run_settings() -> None:
    base = _benchmark_document(common_wall=1.0, inference_wall=1.0, retained=100)
    head = _benchmark_document(common_wall=1.0, inference_wall=1.0, retained=100)
    metadata = head["metadata"]
    assert isinstance(metadata, dict)
    metadata["repetitions"] = 5

    with pytest.raises(ValueError, match="metadata differs for 'repetitions'"):
        compare_benchmarks(
            base,
            head,
            common_limit=0.05,
            inference_limit=0.10,
            retained_growth_bytes=0,
        )


@pytest.mark.parametrize("setting", ["python", "platform", "pyfixest"])
def test_compare_benchmarks_rejects_different_environments(setting: str) -> None:
    base = _benchmark_document(common_wall=1.0, inference_wall=1.0, retained=100)
    head = _benchmark_document(common_wall=1.0, inference_wall=1.0, retained=100)
    metadata = head["metadata"]
    assert isinstance(metadata, dict)
    metadata[setting] = "different"

    with pytest.raises(ValueError, match=rf"metadata differs for '{setting}'"):
        compare_benchmarks(
            base,
            head,
            common_limit=0.05,
            inference_limit=0.10,
            retained_growth_bytes=0,
        )


def test_profiles_keep_default_safe_and_expose_review_sizes() -> None:
    smoke = {scenario.name: scenario for scenario in build_scenarios("smoke")}
    large = {scenario.name: scenario for scenario in build_scenarios("large")}

    assert max(scenario.nobs for scenario in smoke.values()) <= 10_000
    assert large["ols-unweighted-narrow"].nobs == 1_000_000
    assert large["ols-unweighted-narrow"].n_covariates == 10
    assert large["ols-aweights-wide"].nobs == 100_000
    assert large["ols-aweights-wide"].n_covariates == 100
