"""
Generate reference estimation output from a *released* pyfixest.

`tests/test_release_snapshot.py` imports this module and re-runs `run_case()`
with the working tree's pyfixest, comparing the result against the JSON this
script writes. Both sides therefore execute identical driver code and only the
pyfixest version differs.

Regenerate only when the reference release changes -- not to make a failing
test pass. A diff in the stored numbers is a behaviour change that belongs in
the changelog.

```bash
python -m venv /tmp/pyfixest-release && /tmp/pyfixest-release/bin/pip install "pyfixest==0.60.0"
/tmp/pyfixest-release/bin/python tests/data/generate_release_snapshots.py
```
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SNAPSHOT_PATH = Path(__file__).parent / "release_snapshots.json"
REFERENCE_RELEASE = "0.60.0"

# Number of leading elements stored for vector-valued output. Enough to catch a
# reordering or an off-by-one row drop without bloating the JSON.
HEAD = 10
NEWDATA_ROWS = 25

# Iterative solvers are driven well past their defaults so both versions are
# compared at their converged solution. At the default `iwls_tol=1e-8` the two
# stop on different iterations and their standard errors differ in the fifth
# significant digit while agreeing exactly once converged; the stopping point is
# not a contract, the solution is.
IWLS_TOL = 1e-10
LSQR_TOL = 1e-10


def snapshot_data() -> pd.DataFrame:
    """
    Build the deterministic sample every snapshot case is estimated on.

    Self-contained on purpose: `pf.get_data()` is part of the surface under
    test, so the reference data must not come from it.
    """
    rng = np.random.default_rng(1234)
    n = 500

    Z1 = rng.normal(size=n)
    # Z1 is a relevant instrument for X2, and X2 is endogenous through `shared`.
    shared = rng.normal(size=n)
    X1 = rng.normal(size=n)
    X2 = 0.7 * Z1 + 0.5 * shared + rng.normal(size=n)
    f1 = rng.integers(0, 20, size=n)
    f2 = rng.integers(0, 10, size=n)
    f3 = rng.integers(0, 4, size=n)

    Y = (
        1.0
        + 1.5 * X1
        - 0.5 * X2
        + 0.3 * f1
        - 0.2 * f2
        + 0.8 * shared
        + rng.normal(size=n)
    )
    data = pd.DataFrame(
        {
            "Y": Y,
            "Y_count": rng.poisson(np.exp(0.3 * X1 - 0.1 * X2), size=n),
            "Y_binary": (np.median(Y) < Y).astype(float),
            "X1": X1,
            "X2": X2,
            "Z1": Z1,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "weights": rng.uniform(0.5, 2.0, size=n),
            "fweights": rng.integers(1, 4, size=n).astype(float),
        }
    )
    # Missing values exercise the NA-dropping path in model matrix creation.
    data.loc[[3, 17, 42], "X1"] = np.nan
    data.loc[[5, 91], "Y"] = np.nan
    return data


def data_fingerprint(data: pd.DataFrame) -> str:
    """Hash the sample so a drifting DGP fails loudly instead of silently."""
    digest = hashlib.sha256()
    digest.update(",".join(data.columns).encode())
    digest.update(np.ascontiguousarray(data.to_numpy(dtype=float, na_value=np.nan)))
    return digest.hexdigest()


# Each case is (id, kwargs for the estimation function). `method` selects the
# entry point; everything else is passed straight through.
CASES: list[dict[str, Any]] = [
    # --- plain OLS and operator expansion -----------------------------------
    {"id": "ols-simple", "fml": "Y ~ X1"},
    {"id": "ols-two-covariates", "fml": "Y ~ X1 + X2", "vcov": "hetero"},
    {"id": "ols-star", "fml": "Y ~ X1*X2"},
    {"id": "ols-colon", "fml": "Y ~ X1:X2"},
    {"id": "ols-caret-power", "fml": "Y ~ (X1 + X2)^2"},
    {"id": "ols-double-star", "fml": "Y ~ (X1 + X2)**2"},
    {"id": "ols-no-intercept", "fml": "Y ~ X1 - 1"},
    # --- transforms ----------------------------------------------------------
    {"id": "ols-poly", "fml": "Y ~ poly(X1, 2)"},
    {"id": "ols-log", "fml": "Y ~ X1 + np.log(np.abs(X2) + 1)"},
    {"id": "ols-identity-sum", "fml": "I(Y + X1) ~ X2"},
    {"id": "ols-categorical", "fml": "Y ~ C(f1)"},
    {"id": "ols-categorical-transformed", "fml": "Y ~ C(np.floor(X2))"},
    {"id": "ols-i", "fml": "Y ~ i(f1)"},
    {"id": "ols-i-ref", "fml": "Y ~ i(f1, ref=1)"},
    {"id": "ols-i-continuous", "fml": "Y ~ i(f1, X2)"},
    # --- fixed effects -------------------------------------------------------
    {"id": "fe-one", "fml": "Y ~ X1 | f1"},
    {"id": "fe-two", "fml": "Y ~ X1 + X2 | f1 + f2"},
    {"id": "fe-interacted", "fml": "Y ~ X1 | f1^f2"},
    {"id": "fe-three", "fml": "Y ~ X1 | f1 + f2 + f3"},
    {"id": "fe-with-categorical", "fml": "Y ~ X1 + C(f2) | f1"},
    {"id": "fe-demean-only", "fml": "Y ~ 1 | f1"},
    # --- vcov ----------------------------------------------------------------
    {"id": "vcov-crv1", "fml": "Y ~ X1 + X2", "vcov": {"CRV1": "f1"}},
    {"id": "vcov-crv1-fe", "fml": "Y ~ X1 | f1", "vcov": {"CRV1": "f2"}},
    {"id": "vcov-hetero-fe", "fml": "Y ~ X1 | f1", "vcov": "hetero"},
    {"id": "vcov-iid-fe", "fml": "Y ~ X1 | f1", "vcov": "iid"},
    # --- weights -------------------------------------------------------------
    {"id": "weights-aweights", "fml": "Y ~ X1 + X2", "weights": "weights"},
    {"id": "weights-aweights-fe", "fml": "Y ~ X1 | f1", "weights": "weights"},
    {
        "id": "weights-fweights",
        "fml": "Y ~ X1 + X2",
        "weights": "fweights",
        "weights_type": "fweights",
    },
    {
        "id": "weights-fweights-fe",
        "fml": "Y ~ X1 | f1",
        "weights": "fweights",
        "weights_type": "fweights",
    },
    {
        "id": "weights-crv1",
        "fml": "Y ~ X1 | f1",
        "weights": "weights",
        "vcov": {"CRV1": "f2"},
    },
    # --- ssc -----------------------------------------------------------------
    {"id": "ssc-no-adj", "fml": "Y ~ X1 | f1", "ssc_kwargs": {"k_adj": False}},
    {
        "id": "ssc-cluster-adj",
        "fml": "Y ~ X1 | f1",
        "vcov": {"CRV1": "f2"},
        "ssc_kwargs": {"G_adj": False},
    },
    # --- instrumental variables ---------------------------------------------
    {"id": "iv-simple", "fml": "Y ~ X1 | X2 ~ Z1"},
    {"id": "iv-fe", "fml": "Y ~ X1 | f1 | X2 ~ Z1"},
    {"id": "iv-crv1", "fml": "Y ~ X1 | X2 ~ Z1", "vcov": {"CRV1": "f1"}},
    # --- non-linear models ---------------------------------------------------
    {"id": "pois-simple", "method": "fepois", "fml": "Y_count ~ X1 + X2"},
    {"id": "pois-fe", "method": "fepois", "fml": "Y_count ~ X1 | f1"},
    {
        "id": "pois-fe-crv1",
        "method": "fepois",
        "fml": "Y_count ~ X1 | f1",
        "vcov": {"CRV1": "f2"},
    },
    {
        "id": "glm-logit",
        "method": "feglm",
        "fml": "Y_binary ~ X1 + X2",
        "family": "logit",
    },
    {
        "id": "glm-probit-fe",
        "method": "feglm",
        "fml": "Y_binary ~ X1 | f1",
        "family": "probit",
    },
    # --- multiple estimation -------------------------------------------------
    {"id": "multi-sw", "fml": "Y ~ sw(X1, X2)"},
    {"id": "multi-csw", "fml": "Y ~ csw(X1, X2)"},
    {"id": "multi-sw0", "fml": "Y ~ sw0(X1, X2)"},
    {"id": "multi-fe-sw", "fml": "Y ~ X1 | sw(f1, f2)"},
    {"id": "multi-both", "fml": "Y ~ sw(X1, X2) | sw(f1, f2)"},
    {"id": "multi-dependents", "fml": "Y + Y_count ~ X1"},
]


def run_case(case: dict[str, Any], data: pd.DataFrame) -> dict[str, Any]:
    """Estimate one case and reduce it to comparable, version-neutral output."""
    import pyfixest as pf

    kwargs = {
        key: value
        for key, value in case.items()
        if key not in ("id", "method", "ssc_kwargs")
    }
    if "ssc_kwargs" in case:
        kwargs["ssc"] = pf.ssc(**case["ssc_kwargs"])
    method = case.get("method", "feols")
    if method in ("fepois", "feglm"):
        kwargs["iwls_tol"] = IWLS_TOL
    estimator = {"feols": pf.feols, "fepois": pf.fepois, "feglm": pf.feglm}[method]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = estimator(data=data, **kwargs)
        models = (
            list(fitted.all_fitted_models.values())
            if hasattr(fitted, "all_fitted_models")
            else [fitted]
        )
        return {"models": [_summarise(model, data) for model in models]}


def _summarise(model: Any, data: pd.DataFrame) -> dict[str, Any]:
    """Reduce a fitted model to the output that must not change across versions."""
    summary: dict[str, Any] = {
        "coefnames": [str(name) for name in model.coef().index],
        "coef": _head(model.coef().to_numpy(), limit=None),
        "se": _head(model.se().to_numpy(), limit=None),
        "tstat": _head(model.tstat().to_numpy(), limit=None),
        "pvalue": _head(model.pvalue().to_numpy(), limit=None),
        "nobs": int(model._N),
        "resid": _head(model.resid()),
    }
    # `predict()` is unsupported for IV; `newdata` exercises the stored spec.
    try:
        summary["predict"] = _head(
            np.asarray(model.predict(atol=LSQR_TOL, btol=LSQR_TOL))
        )
        summary["predict_newdata"] = _head(
            np.asarray(
                model.predict(
                    newdata=data.head(NEWDATA_ROWS), atol=LSQR_TOL, btol=LSQR_TOL
                )
            ),
            limit=NEWDATA_ROWS,
        )
    except NotImplementedError:
        summary["predict"] = None
        summary["predict_newdata"] = None
    if getattr(model, "_has_fixef", False):
        try:
            summary["fixef"] = _normalise_fixef(
                model.fixef(atol=LSQR_TOL, btol=LSQR_TOL)
            )
        except NotImplementedError:
            # fixef() is unsupported for IV models.
            summary["fixef"] = None
    return summary


def _normalise_fixef(fixef: Any) -> list[list[Any]]:
    """
    Flatten fixed effect estimates to `[variable, level, coefficient]` rows.

    Released pyfixest returns `dict[str, dict[str, float]]` keyed by the encoded
    name (`C(f1)`); newer versions return a DataFrame. Both reduce to the same
    rows, so the *values* stay comparable across the return-type change.
    """
    rows: list[list[Any]] = []
    if isinstance(fixef, pd.DataFrame):
        for record in fixef.to_dict("records"):
            rows.append(
                [
                    _strip_encoding(str(record["variable"])),
                    _canonical_level(record["level"]),
                    round(float(record["coefficient"]), 8),
                ]
            )
    else:
        for variable, levels in fixef.items():
            for level, coefficient in levels.items():
                rows.append(
                    [
                        _strip_encoding(str(variable)),
                        _canonical_level(level),
                        round(float(coefficient), 8),
                    ]
                )
    return sorted(rows, key=lambda row: (row[0], row[1]))


def _strip_encoding(variable: str) -> str:
    """`C(f1)` and `f1` name the same fixed effect."""
    return (
        variable[2:-1]
        if variable.startswith("C(") and variable.endswith(")")
        else variable
    )


def _canonical_level(level: Any) -> str:
    """Levels are stored as ints, floats or strings depending on the version."""
    try:
        numeric = float(level)
    except (TypeError, ValueError):
        return str(level)
    return str(int(numeric)) if numeric.is_integer() else str(numeric)


def _head(values: Any, limit: int | None = HEAD) -> list[float | None]:
    flat = np.asarray(values, dtype=float).ravel()
    if limit is not None:
        flat = flat[:limit]
    return [None if np.isnan(value) else round(float(value), 8) for value in flat]


def main() -> None:
    import pyfixest as pf

    data = snapshot_data()
    snapshots = {
        "pyfixest_version": pf.__version__,
        "data_fingerprint": data_fingerprint(data),
        "cases": {case["id"]: run_case(case, data) for case in CASES},
    }
    SNAPSHOT_PATH.write_text(json.dumps(snapshots, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(CASES)} cases from pyfixest {pf.__version__} to {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
