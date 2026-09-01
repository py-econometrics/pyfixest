"""Shared case matrix and result extraction for release snapshots."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from _feols_test_cases import (
    FEOLS_FORMULA_F3_CASES,
    fixed_effect_interactions_to_legacy,
    glm_fmls,
    iv_fmls,
    ols_fmls,
    ssc_formula_vcov_dropna_cases,
)
from _feols_test_cases import (
    convert_f3 as _convert_f3,
)

SCHEMA_VERSION = 1
RELEASE_VERSION = "0.60.0"
DATA_SEED = 20260831
NOBS = 500
SAMPLE_SIZE = 5

VCOV_CASES: tuple[str | dict[str, str], ...] = (
    "iid",
    "hetero",
    {"CRV1": "group_id"},
)
WEIGHT_CASES = ((None, None), ("weights", "aweights"))
IV_WEIGHT_CASES = (*WEIGHT_CASES, ("fweights", "fweights"))

# Release 0.60.0 predates the new logit/probit starting values. Keep only the
# combinations whose complete numerical contract is comparable. Current GLM
# correctness remains covered by the live-R matrix.
FEGLM_FAMILY_VCOV_CASES = (
    ("logit", "iid"),
    # ("logit", "hetero"),
    # ("logit", {"CRV1": "group_id"}),
    # ("probit", "iid"),
    ("probit", "hetero"),
    ("probit", {"CRV1": "group_id"}),
)

QUANTREG_CASES = (
    ("Y ~ X1", 0.02, "fn", "nid"),
    ("Y ~ X1 + X2", 0.35, "pfn", "nid"),
    ("Y ~ X1", 0.5, "pfn", "nid"),
    ("Y ~ X1 + X2", 0.9, "fn", "nid"),
    ("Y ~ X1 + X2", 0.5, "fn", {"CRV1": "group_id"}),
)


def _case(group: str, index: int, **values: Any) -> dict[str, Any]:
    return {"id": f"{group}-{index:04d}", "group": group, **values}


def _vcov(vcov: str) -> str | dict[str, str]:
    return vcov if vcov in {"iid", "hetero"} else {"CRV1": vcov}


def build_case_groups() -> dict[str, list[dict[str, Any]]]:
    """Mirror the canonical fixest parametrizations without backend variation."""
    groups: dict[str, list[dict[str, Any]]] = {}

    groups["feols"] = [
        _case(
            "feols",
            index,
            estimator="feols",
            formula=formula,
            release_formula=fixed_effect_interactions_to_legacy(formula),
            data="linear",
            dropna=dropna,
            f3_type=f3_type,
            kwargs={"vcov": vcov, "weights": weights},
        )
        for index, (dropna, vcov, weights, (formula, f3_type)) in enumerate(
            product(
                (False, True),
                VCOV_CASES,
                (None, "weights"),
                FEOLS_FORMULA_F3_CASES,
            )
        )
    ]

    groups["fepois"] = [
        _case(
            "fepois",
            index,
            estimator="fepois",
            formula=formula,
            release_formula=fixed_effect_interactions_to_legacy(formula),
            data="count",
            dropna=False,
            f3_type="str",
            kwargs={
                "vcov": vcov,
                "weights": weights,
                "offset": "offset_var" if offset else None,
                "iwls_tol": 1e-10,
                "iwls_maxiter": 100,
            },
        )
        for index, (vcov, formula, weights, offset) in enumerate(
            product(VCOV_CASES, ols_fmls, (None, "weights"), (False, True))
        )
    ]

    groups["iv"] = [
        _case(
            "iv",
            index,
            estimator="feols",
            formula=formula,
            release_formula=fixed_effect_interactions_to_legacy(formula),
            data="linear",
            dropna=False,
            f3_type="str",
            kwargs={
                "vcov": vcov,
                "weights": weights,
                **({"weights_type": weights_type} if weights_type is not None else {}),
            },
        )
        for index, (vcov, formula, (weights, weights_type)) in enumerate(
            product(VCOV_CASES, iv_fmls, IV_WEIGHT_CASES)
        )
    ]

    groups["feglm"] = [
        _case(
            "feglm",
            index,
            estimator="feglm",
            formula=formula.replace("Y", "Y_bin", 1),
            release_formula=fixed_effect_interactions_to_legacy(
                formula.replace("Y", "Y_bin", 1)
            ),
            data="count",
            dropna=False,
            f3_type="str",
            kwargs={
                "family": family,
                "vcov": vcov,
                "iwls_tol": 1e-10,
                "iwls_maxiter": 100,
            },
        )
        for index, (formula, (family, vcov)) in enumerate(
            product(glm_fmls, FEGLM_FAMILY_VCOV_CASES)
        )
    ]

    groups["quantreg"] = [
        _case(
            "quantreg",
            index,
            estimator="quantreg",
            formula=formula,
            release_formula=formula,
            data="quantile",
            dropna=False,
            f3_type="str",
            kwargs={
                "quantile": quantile,
                "method": method,
                "vcov": vcov,
                "tol": 1e-6,
                "seed": 83838,
                "ssc": {"k_adj": False, "G_adj": False},
            },
        )
        for index, (formula, quantile, method, vcov) in enumerate(QUANTREG_CASES)
    ]

    groups["ssc"] = [
        _case(
            "ssc",
            index,
            estimator=model,
            formula=formula,
            release_formula=fixed_effect_interactions_to_legacy(formula),
            data=f"ssc_{model}",
            dropna=dropna,
            f3_type="str",
            kwargs={
                "vcov": _vcov(vcov),
                "weights": weights,
                "ssc": {
                    "k_adj": k_adj,
                    "G_adj": g_adj,
                    "G_df": "min",
                    "k_fixef": k_fixef,
                },
                **(
                    {"iwls_tol": 1e-10, "iwls_maxiter": 100}
                    if model == "fepois"
                    else {}
                ),
            },
        )
        for index, (
            (formula, dropna, vcov),
            weights,
            k_adj,
            g_adj,
            k_fixef,
            model,
        ) in enumerate(
            product(
                ssc_formula_vcov_dropna_cases,
                (None, "weights"),
                (True, False),
                (True, False),
                ("full", "none", "nonnested"),
                ("feols", "fepois"),
            )
        )
    ]
    return groups


def build_cases() -> list[dict[str, Any]]:
    return [case for cases in build_case_groups().values() for case in cases]


def build_data(data_id: str, *, dropna: bool, f3_type: str) -> pd.DataFrame:
    """Build deterministic data shared by the release and development fits."""
    seed = DATA_SEED + sum(map(ord, data_id))
    rng = np.random.default_rng(seed)
    f1 = rng.integers(0, 10, size=NOBS)
    f2 = rng.integers(0, 12, size=NOBS)
    f3 = rng.integers(0, 8, size=NOBS)
    z1 = rng.normal(size=NOBS)
    z2 = rng.normal(size=NOBS)
    x1 = 0.8 * z1 + rng.normal(size=NOBS)
    x2 = rng.normal(size=NOBS)
    linear = 5 + 0.7 * x1 - 0.4 * x2 + 0.15 * f1 - 0.1 * f2

    if data_id in {"linear", "ssc_feols"}:
        y = linear + rng.normal(scale=0.8, size=NOBS)
    elif data_id in {"count", "ssc_fepois"}:
        y = rng.poisson(np.exp(np.clip(-0.2 + 0.25 * x1 - 0.15 * x2, -3, 3)))
    elif data_id == "quantile":
        y = 1 + 2 * x1 + 3 * x2 - 0.5 * x2**2 + rng.normal(size=NOBS)
    else:  # pragma: no cover - closed case matrix
        raise ValueError(f"Unknown snapshot data id: {data_id}")

    data = pd.DataFrame(
        {
            "Y": y,
            "Y2": linear - 0.5 * x1 + rng.normal(size=NOBS),
            "Y_bin": (linear + rng.normal(size=NOBS) > np.median(linear)).astype(int),
            "X1": x1,
            "X2": x2,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "group_id": f1,
            "Z1": z1,
            "Z2": z2,
            "weights": 0.5 + rng.random(size=NOBS),
            "fweights": rng.integers(1, 5, size=NOBS),
            "offset_var": np.log(rng.uniform(0.5, 3.0, size=NOBS)),
        }
    )
    for row, column in enumerate(("Y", "X1", "X2", "Z1", "weights")):
        data.loc[row, column] = np.nan
    if dropna:
        data = data.dropna().copy()
    data.where(data != "nan", np.nan, inplace=True)
    return _convert_f3(data, f3_type)


def fit_case(case: dict[str, Any], *, release: bool) -> Any:
    """Fit one case through the public API using the requested formula spelling."""
    import pyfixest as pf

    data = build_data(
        str(case["data"]),
        dropna=bool(case["dropna"]),
        f3_type=str(case["f3_type"]),
    )
    kwargs = dict(case["kwargs"])
    if "ssc" in kwargs:
        kwargs["ssc"] = pf.ssc(**kwargs["ssc"])
    formula = case["release_formula"] if release else case["formula"]
    return getattr(pf, str(case["estimator"]))(formula, data=data, **kwargs)


def _number(value: object) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _named(series: pd.Series) -> dict[str, float | None]:
    return {str(name): _number(value) for name, value in series.sort_index().items()}


def extract_snapshot(fit: Any, *, estimator: str) -> dict[str, Any]:
    """Extract stable named results without serialising model internals."""
    coef = fit.coef()
    names = list(coef.index)
    sorted_names = sorted(names)
    order = [names.index(name) for name in sorted_names]
    vcov = np.asarray(fit._vcov)
    try:
        predict_sample = [
            _number(value) for value in np.asarray(fit.predict()).ravel()[:SAMPLE_SIZE]
        ]
        prediction_supported = True
    except NotImplementedError:
        predict_sample = []
        prediction_supported = False
    metadata: dict[str, Any] = {
        "coefnames": names,
        "nobs": int(fit._N),
        "df_k": int(fit._df_k),
        "df_t": int(fit._df_t),
        "dropped_terms": sorted(str(name) for name in getattr(fit, "_collin_vars", [])),
        "prediction_supported": prediction_supported,
    }
    if hasattr(fit, "convergence"):
        metadata["convergence"] = bool(fit.convergence)
    if hasattr(fit, "_has_converged"):
        metadata["convergence"] = bool(fit._has_converged)
    result: dict[str, Any] = {
        "coef": _named(coef),
        "vcov": {
            row: {
                column: _number(vcov[row_pos, column_pos])
                for column, column_pos in zip(sorted_names, order, strict=True)
            }
            for row, row_pos in zip(sorted_names, order, strict=True)
        },
        "se": _named(fit.se()),
        "tstat": _named(fit.tstat()),
        "pvalue": _named(fit.pvalue()),
        "metadata": metadata,
        "predict_sample": predict_sample,
    }
    # GLM residual semantics intentionally changed after 0.60.0.
    if estimator != "feglm":
        result["resid_sample"] = [
            _number(value) for value in np.asarray(fit.resid()).ravel()[:SAMPLE_SIZE]
        ]
    # Quantreg now uses its t reference distribution for confidence intervals.
    if estimator != "quantreg":
        confint = fit.confint().sort_index()
        result["confint"] = {
            str(name): {str(column): _number(value) for column, value in row.items()}
            for name, row in confint.to_dict(orient="index").items()
        }
    if getattr(fit, "deviance", None) is not None:
        result["deviance"] = _number(fit.deviance)
    return result


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
