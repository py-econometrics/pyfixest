"""Numerical regression checks against the released pyfixest contract."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _estimation_snapshot_cache import (
    COMPLETE_MARKER,
    snapshot_dir,
    snapshot_fingerprint,
)
from _estimation_snapshot_contract import (
    build_case_groups,
    extract_snapshot,
    fit_case,
    load_json,
)

CASE_GROUPS = build_case_groups()

# Regression tolerances compare two pyfixest versions on the same deterministic
# inputs; live-R tests retain their own estimator-specific correctness bounds.
TOLERANCES: dict[str, dict[str, tuple[float, float]]] = {
    "feols": {
        "coef": (1e-8, 1e-9),
        "vcov": (1e-7, 1e-9),
        "inference": (2e-7, 1e-9),
        "samples": (1e-6, 1e-9),
        "metrics": (1e-7, 1e-9),
    },
    "fepois": {
        "coef": (1e-6, 1e-8),
        # The post-0.60 step-halving and warm-start rewrite moves the expanded
        # formula matrix slightly; these bounds remain tighter than live-R.
        "vcov": (2e-6, 4e-7),
        "inference": (2e-5, 2e-6),
        "samples": (1e-6, 1e-8),
        "metrics": (1e-7, 1e-8),
    },
    "feglm": {
        "coef": (1e-6, 3e-7),
        # Logit/probit use new IRLS starting values after 0.60.0. The measured
        # drift is bounded here; live-R validates the new results directly.
        "vcov": (5e-5, 5e-7),
        "inference": (2e-5, 4e-6),
        "samples": (5e-6, 2e-6),
        "metrics": (1e-7, 1e-8),
    },
    "quantreg": {
        "coef": (1e-7, 1e-8),
        "vcov": (1e-7, 1e-8),
        "inference": (1e-7, 1e-8),
        "samples": (1e-7, 1e-8),
        "metrics": (1e-7, 1e-8),
    },
}


def _params(group: str) -> list[Any]:
    return [pytest.param(case, id=case["id"]) for case in CASE_GROUPS[group]]


@pytest.fixture(scope="session")
def release_contract() -> tuple[dict[str, Any], dict[str, dict[str, object]]]:
    cache_dir = snapshot_dir()
    if not (cache_dir / COMPLETE_MARKER).exists():
        pytest.fail(
            "Release snapshot cache is missing. Run "
            "`pixi run -e py312 test-estimation-snapshots` to prepare it."
        )
    manifest = load_json(cache_dir / "manifest.json")
    expected: dict[str, dict[str, object]] = {}
    for filename in sorted(cache_dir.glob("*.json")):
        if filename.name != "manifest.json":
            expected.update(load_json(filename)["cases"])
    return manifest, expected


def _flatten_numeric(value: Any, *, path: str) -> dict[str, float | None]:
    if isinstance(value, dict):
        flattened: dict[str, float | None] = {}
        for key, child in value.items():
            flattened.update(_flatten_numeric(child, path=f"{path}.{key}"))
        return flattened
    if isinstance(value, list):
        flattened = {}
        for index, child in enumerate(value):
            flattened.update(_flatten_numeric(child, path=f"{path}[{index}]"))
        return flattened
    return {path: value}


def _assert_numeric_tree(
    actual: Any,
    expected: Any,
    *,
    tolerance: tuple[float, float],
    quantity: str,
) -> None:
    actual_values = _flatten_numeric(actual, path=quantity)
    expected_values = _flatten_numeric(expected, path=quantity)
    assert set(actual_values) == set(expected_values), f"{quantity} shape differs"
    paths = sorted(expected_values)
    assert [actual_values[path] is None for path in paths] == [
        expected_values[path] is None for path in paths
    ], f"{quantity} non-finite positions differ"
    finite = [path for path in paths if expected_values[path] is not None]
    np.testing.assert_allclose(
        [actual_values[path] for path in finite],
        [expected_values[path] for path in finite],
        rtol=tolerance[0],
        atol=tolerance[1],
        err_msg=f"{quantity} differs from the pyfixest release contract",
    )


def _assert_case(case, release_contract) -> None:
    _, expected_cases = release_contract
    estimator = str(case["estimator"])
    actual = extract_snapshot(fit_case(case, release=False), estimator=estimator)
    expected = expected_cases[case["id"]]
    assert set(actual) == set(expected), "snapshot fields differ"
    assert actual["metadata"] == expected["metadata"], "metadata differs"
    tolerances = TOLERANCES[estimator]
    for quantity in sorted(set(expected) - {"metadata"}):
        tolerance_name = {
            "coef": "coef",
            "vcov": "vcov",
            "resid_sample": "samples",
            "predict_sample": "samples",
            "deviance": "metrics",
        }.get(quantity, "inference")
        _assert_numeric_tree(
            actual[quantity],
            expected[quantity],
            tolerance=tolerances[tolerance_name],
            quantity=quantity,
        )


def test_estimation_snapshot_inventory(release_contract) -> None:
    manifest, expected_cases = release_contract
    fingerprint = snapshot_fingerprint()
    case_ids = {case["id"] for cases in CASE_GROUPS.values() for case in cases}
    assert case_ids == {case["id"] for case in manifest["cases"]}
    assert case_ids == set(expected_cases)
    assert manifest["fingerprint"] == fingerprint
    assert (snapshot_dir() / COMPLETE_MARKER).read_text().strip() == fingerprint


@pytest.mark.parametrize("case", _params("feols"))
def test_feols_release_contract(case, release_contract):
    _assert_case(case, release_contract)


@pytest.mark.parametrize("case", _params("fepois"))
def test_fepois_release_contract(case, release_contract):
    _assert_case(case, release_contract)


@pytest.mark.parametrize("case", _params("iv"))
def test_iv_release_contract(case, release_contract):
    _assert_case(case, release_contract)


@pytest.mark.parametrize("case", _params("feglm"))
def test_feglm_release_contract(case, release_contract):
    _assert_case(case, release_contract)


@pytest.mark.parametrize("case", _params("quantreg"))
def test_quantreg_release_contract(case, release_contract):
    _assert_case(case, release_contract)


@pytest.mark.parametrize("case", _params("ssc"))
def test_ssc_release_contract(case, release_contract):
    _assert_case(case, release_contract)
