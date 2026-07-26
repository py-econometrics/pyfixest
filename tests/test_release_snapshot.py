"""
Snapshot tests pinning estimation output to the last released pyfixest.

The formula-parsing refactor rewrote how formulas are parsed and how model
matrices are built. Coefficients, standard errors, predictions and fixed effect
estimates must come out bit-for-bit comparable to the release for every
supported specification; the reference values in
`tests/data/release_snapshots.json` were produced by pyfixest 0.60.0.

The case list, the sample and the driver code live in
`tests/data/generate_release_snapshots.py` and are shared with the generator, so
a failure here is a genuine behaviour change rather than test drift. See that
file for how to regenerate.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_DATA_DIR = Path(__file__).parent / "data"
_GENERATOR_PATH = _DATA_DIR / "generate_release_snapshots.py"
_SNAPSHOT_PATH = _DATA_DIR / "release_snapshots.json"

# Estimates from closed-form OLS move only by accumulated floating point error,
# but the suite also covers models solved iteratively: alternating projections
# for the fixed effects and IRLS for Poisson/GLM. Those agree with the release
# to their convergence tolerance rather than to machine precision - the probit
# case below reproduces the release deviance to 1e-13 while its coefficient
# differs by 7e-7. `SOLVER_ATOL` covers fixed effect estimates and the
# predictions built from them, which come out of an lsqr solve run to 1e-6.
# A genuine behaviour change moves the third or fourth digit, far above either.
RTOL = 1e-6
ATOL = 1e-8
SOLVER_ATOL = 1e-5

# Released 0.60.0 refused `predict(newdata=...)` for these specifications: IV
# models still do, but the `i()` and `^` blocks were lifted by the stored
# ModelSpec, so a reference of `None` here is an improvement rather than a
# missing comparison. Listed explicitly so a silent re-block is caught.
PREDICT_UNBLOCKED = {"ols-i", "ols-i-ref", "ols-i-continuous", "fe-interacted"}

# `Feglm.resid()` gained a `type` argument defaulting to *response* residuals
# after 0.60.0 shipped; the release had no override and returned the working
# residuals it inherited from `Feols`. That landed on master independently of
# the formula-parsing work, so these residuals are not comparable across the
# two versions. Coefficients, standard errors and predictions still are.
RESID_REDEFINED_AFTER_RELEASE = {"glm-logit", "glm-probit-fe"}

# Specifications this branch currently gets wrong, each fixed by a follow-up PR
# in this series. `strict` means the marker has to be removed together with the
# fix, so none of these can quietly stay broken.
KNOWN_FAILURES = {
    "ols-identity-sum": (
        "any `+` on the left hand side is read as multiple dependent variables, "
        "including one nested inside a transform"
    ),
    "ols-categorical-transformed": (
        "unseen categorical levels are looked up in the factor's source "
        "columns, so every row of `C(np.floor(X2))` predicts NaN"
    ),
}


def _case_params() -> list[Any]:
    return [
        pytest.param(
            case,
            id=case["id"],
            marks=(
                [pytest.mark.xfail(strict=True, reason=KNOWN_FAILURES[case["id"]])]
                if case["id"] in KNOWN_FAILURES
                else []
            ),
        )
        for case in generator.CASES
    ]


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "release_snapshot_generator", _GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = _load_generator()


@pytest.fixture(scope="module")
def snapshots() -> dict[str, Any]:
    return json.loads(_SNAPSHOT_PATH.read_text())


@pytest.fixture(scope="module")
def data():
    return generator.snapshot_data()


def test_snapshot_reference_is_the_released_version(snapshots) -> None:
    assert snapshots["pyfixest_version"] == generator.REFERENCE_RELEASE


def test_snapshot_data_is_unchanged(snapshots, data) -> None:
    """A drifting DGP would silently invalidate every stored number."""
    assert generator.data_fingerprint(data) == snapshots["data_fingerprint"]


@pytest.mark.parametrize("case", _case_params())
def test_output_matches_released_pyfixest(case, snapshots, data) -> None:
    expected = snapshots["cases"][case["id"]]
    actual = generator.run_case(case, data)

    assert len(actual["models"]) == len(expected["models"]), (
        f"{case['id']}: expected {len(expected['models'])} model(s), "
        f"got {len(actual['models'])}"
    )
    for index, (got, want) in enumerate(
        zip(actual["models"], expected["models"], strict=True)
    ):
        _assert_model_matches(
            got, want, label=f"{case['id']}[{index}]", case_id=case["id"]
        )


def _assert_model_matches(got: dict, want: dict, label: str, case_id: str) -> None:
    assert got["coefnames"] == want["coefnames"], f"{label}: coefficient names"
    assert got["nobs"] == want["nobs"], f"{label}: number of observations"

    compared = ["coef", "se", "tstat", "pvalue"]
    if case_id not in RESID_REDEFINED_AFTER_RELEASE:
        compared.append("resid")
    for key in compared:
        _assert_allclose(got[key], want[key], label=f"{label}: {key}")
    for key in ("predict", "predict_newdata"):
        if want[key] is not None:
            _assert_allclose(got[key], want[key], label=f"{label}: {key}", solver=True)
        elif case_id in PREDICT_UNBLOCKED:
            assert got[key] is not None, f"{label}: {key} is blocked again"
        else:
            assert got[key] is None, f"{label}: {key} became supported"

    if want.get("fixef") is not None:
        _assert_fixef_matches(got["fixef"], want["fixef"], label=f"{label}: fixef")


def _assert_fixef_matches(got: list, want: list, label: str) -> None:
    """
    Every level the release estimated must keep its value.

    With two or more fixed effects the estimates are identified only up to a
    normalisation. The release omitted the reference level of the second and
    later fixed effects; it is now listed explicitly, pinned to zero.
    """
    got_by_level = {(row[0], row[1]): row[2] for row in got}
    want_by_level = {(row[0], row[1]): row[2] for row in want}

    missing = sorted(set(want_by_level) - set(got_by_level))
    assert not missing, f"{label}: levels dropped {missing}"

    shared = sorted(want_by_level)
    _assert_allclose(
        [got_by_level[level] for level in shared],
        [want_by_level[level] for level in shared],
        label=label,
        solver=True,
    )
    added = sorted(set(got_by_level) - set(want_by_level))
    assert all(got_by_level[level] == 0.0 for level in added), (
        f"{label}: added levels {added} must be reference levels pinned to zero"
    )


def _assert_allclose(got: list, want: list, label: str, solver: bool = False) -> None:
    got_array = np.asarray(got, dtype=float)
    want_array = np.asarray(want, dtype=float)
    assert got_array.shape == want_array.shape, f"{label}: shape"
    # None round-trips through JSON as nan; nan must line up on both sides.
    np.testing.assert_array_equal(
        np.isnan(got_array), np.isnan(want_array), err_msg=f"{label}: nan positions"
    )
    np.testing.assert_allclose(
        got_array,
        want_array,
        rtol=RTOL,
        atol=SOLVER_ATOL if solver else ATOL,
        equal_nan=True,
        err_msg=label,
    )
