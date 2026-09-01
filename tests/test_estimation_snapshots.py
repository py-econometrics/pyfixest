"""Python-only numerical regression checks against the released pyfixest contract."""

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
    TOLERANCES,
    build_cases,
    extract_snapshot,
    fast_case_ids,
    fit_case,
    load_json,
)


def _expected_cases(cache_dir: Path) -> dict[str, dict[str, object]]:
    expected: dict[str, dict[str, object]] = {}
    for filename in sorted(cache_dir.glob("*.json")):
        if filename.name == "manifest.json":
            continue
        expected.update(load_json(filename)["cases"])
    return expected


CASES = build_cases()
FAST_CASE_IDS = fast_case_ids(CASES)
LOGIT_VCOV_DELTA_CASE_IDS = frozenset(
    {
        "feglm-001-family=logit-vcov=hetero",
        "feglm-002-family=logit-vcov=crv1",
    }
)
PROBIT_IID_DELTA_CASE_ID = "feglm-003-family=probit-vcov=iid"
FWEIGHTS_SINGLETON_DELTA_CASE_IDS = frozenset(
    str(case["id"])
    for case in CASES
    if case["estimator"] == "feols"
    and case["formula"] == "Y ~ X1 + X2 | f1^f2"
    and case["kwargs"].get("weights") == "fweights"
    and "group=ssc" in str(case["id"])
)
PARAMS = [
    pytest.param(
        case,
        marks=pytest.mark.snapshot_fast if case["id"] in FAST_CASE_IDS else (),
        id=case["id"],
    )
    for case in CASES
]


@pytest.fixture(scope="session")
def release_contract() -> tuple[dict[str, Any], dict[str, dict[str, object]]]:
    """Load the complete platform-local release baseline."""
    cache_dir = snapshot_dir()
    if not (cache_dir / COMPLETE_MARKER).exists():
        pytest.fail(
            "Release snapshot cache is missing. Run "
            "`pixi run -e py312 test-estimation-snapshots` to prepare it."
        )
    return load_json(cache_dir / "manifest.json"), _expected_cases(cache_dir)


def _assert_named_close(
    actual, expected, *, tolerance: tuple[float, float], quantity: str
) -> None:
    assert set(actual) == set(expected), f"{quantity} names differ"
    names = sorted(expected)
    nonfinite_names = [name for name in names if expected[name] is None]
    assert all(actual[name] is None for name in nonfinite_names), (
        f"{quantity} non-finite positions differ"
    )
    finite_names = [name for name in names if expected[name] is not None]
    np.testing.assert_allclose(
        [actual[name] for name in finite_names],
        [expected[name] for name in finite_names],
        rtol=tolerance[0],
        atol=tolerance[1],
        err_msg=f"{quantity} differs from the pyfixest release contract",
    )


def _assert_sequence_close(
    actual, expected, *, tolerance: tuple[float, float], quantity: str
) -> None:
    assert len(actual) == len(expected), f"{quantity} length differs"
    assert [value is None for value in actual] == [
        value is None for value in expected
    ], f"{quantity} non-finite positions differ"
    finite_pairs = [
        (value, reference)
        for value, reference in zip(actual, expected, strict=True)
        if reference is not None
    ]
    np.testing.assert_allclose(
        [value for value, _ in finite_pairs],
        [reference for _, reference in finite_pairs],
        rtol=tolerance[0],
        atol=tolerance[1],
        err_msg=f"{quantity} differs from the pyfixest release contract",
    )


def _assert_snapshot(
    actual: dict[str, object], expected: dict[str, object], case: dict[str, object]
) -> None:
    estimator = str(case["estimator"])
    tolerances = TOLERANCES[estimator]
    _assert_named_close(
        actual["coef"],
        expected["coef"],
        tolerance=tolerances["coef"],
        quantity="coefficients",
    )
    # Keep documented release deltas tied to the exact measured cases. New
    # FEGLM formulas, weights, or vcov variants must not inherit an omission.
    changed_logit_vcov = str(case["id"]) in LOGIT_VCOV_DELTA_CASE_IDS
    changed_probit_iid = str(case["id"]) == PROBIT_IID_DELTA_CASE_ID
    changed_fweight_singletons = str(case["id"]) in FWEIGHTS_SINGLETON_DELTA_CASE_IDS
    # The documented GLM API/behavior change introduces new logit/probit IRLS
    # starts (docs/changelog.qmd, "GLM API and Behavior"). Only the narrow
    # comparisons below materially exceed normal tolerance; canonical live-R
    # remains their authoritative correctness evidence.
    # Release 0.60.0 treated a physically unique FE row as a singleton even
    # when its frequency weight represented repeated observations. Current
    # head follows literal expansion (and R fixest), which changes effective
    # sample/FE degrees of freedom and derived inference for precisely the
    # interaction cases enumerated above. Coefficients and prediction/residual
    # samples remain checked against the frozen artifact.
    if not changed_logit_vcov and not changed_fweight_singletons:
        for row_name, expected_row in expected["vcov"].items():
            _assert_named_close(
                actual["vcov"][row_name],
                expected_row,
                tolerance=tolerances["vcov"],
                quantity=f"vcov row {row_name}",
            )
    for quantity in ("se", "tstat"):
        if (
            not changed_logit_vcov
            and not changed_probit_iid
            and not changed_fweight_singletons
        ):
            _assert_named_close(
                actual[quantity],
                expected[quantity],
                tolerance=tolerances.get(quantity, tolerances["inference"]),
                quantity=quantity,
            )
    pvalue_actual = dict(actual["pvalue"])
    pvalue_expected = dict(expected["pvalue"])
    if changed_logit_vcov:
        # Only the X2 p-value moves beyond tolerance in logit hetero/CRV1.
        pvalue_actual.pop("X2")
        pvalue_expected.pop("X2")
    if not changed_fweight_singletons:
        _assert_named_close(
            pvalue_actual,
            pvalue_expected,
            tolerance=tolerances.get("pvalue", tolerances["inference"]),
            quantity="pvalue",
        )
    if (
        estimator != "quantreg"
        and not changed_logit_vcov
        and not changed_fweight_singletons
    ):
        for name, expected_interval in expected["confint"].items():
            actual_interval = dict(actual["confint"][name])
            expected_interval = dict(expected_interval)
            if changed_probit_iid and name == "X2":
                # Only the upper X2 interval bound exceeds normal tolerance.
                actual_interval.pop("97.5%")
                expected_interval.pop("97.5%")
            _assert_named_close(
                actual_interval,
                expected_interval,
                tolerance=tolerances.get("confint", tolerances["inference"]),
                quantity=f"confint {name}",
            )
    # The documented inference-type/confint overhaul in docs/changelog.qmd now
    # uses Quantreg's t reference distribution. Release 0.60.0 used a normal
    # critical value despite Quantreg's t-based p-values, so only this derived
    # interval comparison is intentionally omitted.
    actual_metadata = dict(actual["metadata"])
    expected_metadata = dict(expected["metadata"])
    actual_deviance = actual_metadata.pop("deviance", None)
    expected_deviance = expected_metadata.pop("deviance", None)
    if changed_fweight_singletons:
        for name in ("nobs", "df_k", "df_t"):
            actual_metadata.pop(name)
            expected_metadata.pop(name)
    assert actual_metadata == expected_metadata, "structural metadata differs"
    if expected_deviance is not None:
        np.testing.assert_allclose(
            actual_deviance,
            expected_deviance,
            rtol=tolerances["metrics"][0],
            atol=tolerances["metrics"][1],
            err_msg="deviance differs from the pyfixest release contract",
        )
    for quantity in ("resid_sample", "predict_sample"):
        if estimator == "feglm" and quantity == "resid_sample":
            # Current GLMs return explicit response residuals after the
            # documented GLM API/behavior rewrite (docs/changelog.qmd, "GLM
            # API and Behavior"); 0.60.0 exposed working residuals instead.
            # Keep prediction checks; the canonical live-R suite validates residuals.
            continue
        _assert_sequence_close(
            actual[quantity],
            expected[quantity],
            tolerance=tolerances["samples"],
            quantity=quantity,
        )


@pytest.mark.snapshot_fast
def test_estimation_snapshot_inventory(release_contract) -> None:
    """Reject stale, missing, or inconsistently tiered cached artifacts."""
    manifest, expected_cases = release_contract
    fingerprint = snapshot_fingerprint()
    case_ids = {case["id"] for case in CASES}
    manifest_ids = {case["id"] for case in manifest["cases"]}
    artifact_ids = set(expected_cases)
    assert case_ids == manifest_ids == artifact_ids
    assert set(manifest["fast_case_ids"]) == FAST_CASE_IDS
    assert case_ids >= FAST_CASE_IDS
    assert manifest["fingerprint"] == fingerprint
    assert (snapshot_dir() / COMPLETE_MARKER).read_text().strip() == fingerprint
    # Freeze the scope of the documented fweight-singleton release delta. A
    # new formula, vcov, or SSC case must not silently inherit this omission.
    assert len(FWEIGHTS_SINGLETON_DELTA_CASE_IDS) == 108


@pytest.mark.parametrize("case", PARAMS)
def test_estimation_release_contract(case, release_contract):
    """Match the platform-local public-fit contract from pyfixest 0.60.0."""
    _, expected_cases = release_contract
    expected = expected_cases[case["id"]]
    actual = extract_snapshot(fit_case(case))
    _assert_snapshot(actual, expected, case)
