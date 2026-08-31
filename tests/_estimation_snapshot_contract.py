"""Shared release-contract data, cases, extraction, and assertions.

The committed JSON artifacts are intentionally generated with a released wheel,
not with the editable checkout.  This module is dependency-light so the explicit
generator can import it from that isolated interpreter.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from _feols_test_cases import FEOLS_FORMULA_F3_CASES, convert_f3

SCHEMA_VERSION = 1
RELEASE_VERSION = "0.60.0"
DATA_SEED = 20260831
AUGMENTATION_SEED = DATA_SEED + 1
NOBS = 500
SAMPLE_SIZE = 5
SNAPSHOT_DIR = Path(__file__).parent / "data" / "estimation_snapshots"

# These are numerical regression tolerances, not release-to-R tolerances. They
# allow only normal floating-point variation between the release wheel and head.
TOLERANCES: dict[str, dict[str, tuple[float, float]]] = {
    "feols": {
        "coef": (1e-8, 1e-9),
        "vcov": (1e-7, 1e-9),
        "inference": (1e-7, 1e-9),
        # Frequency-weighted three-way FE t statistics differ by 1.54e-7
        # relative between the release wheel and head; other inference outputs
        # remain at the tighter generic bound.
        "tstat": (2e-7, 1e-9),
        "samples": (1e-6, 1e-9),
        "metrics": (1e-7, 1e-9),
    },
    "fepois": {
        "coef": (1e-6, 1e-8),
        "vcov": (1e-6, 1e-8),
        "inference": (1e-6, 1e-8),
        # Tiny absolute p-values require a small absolute allowance; all other
        # FEPoisson quantities retain the tighter 1e-6 relative tolerance.
        "pvalue": (2e-6, 5e-8),
        "confint": (3e-6, 5e-8),
        "samples": (1e-6, 1e-8),
        "metrics": (1e-10, 1e-10),
    },
    "feglm": {
        "coef": (1e-6, 1e-8),
        "vcov": (1e-6, 1e-8),
        "inference": (1e-6, 1e-8),
        "samples": (1e-6, 1e-8),
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


def build_data(data_id: str, *, variant: str, f3_type: str = "str") -> pd.DataFrame:
    """Build a deterministic, small public-API input frame for one estimator."""
    rng = np.random.default_rng(DATA_SEED)
    nobs = NOBS
    f1 = rng.integers(0, 10, size=nobs)
    f2 = rng.integers(0, 12, size=nobs)
    f3 = rng.integers(0, 8, size=nobs).astype(float)
    x1 = rng.normal(size=nobs)
    x2 = rng.normal(size=nobs)
    weights = 0.5 + rng.random(size=nobs)
    linear = 5.0 + 0.7 * x1 - 0.4 * x2 + 0.15 * f1 - 0.1 * f2

    if data_id == "linear":
        outcome = linear + rng.normal(scale=0.8, size=nobs)
    elif data_id == "count":
        outcome = 1 + rng.poisson(
            np.exp(np.clip(-0.2 + 0.25 * x1 - 0.15 * x2 + 0.04 * f1, -3, 3))
        )
    elif data_id == "binary":
        probability = 1 / (
            1 + np.exp(-np.clip(-0.1 + 0.55 * x1 - 0.35 * x2 + 0.1 * f1, -6, 6))
        )
        outcome = rng.binomial(1, probability)
    elif data_id == "quantile":
        outcome = 1 + 2 * x1 + 3 * x2 - 0.5 * x2**2 + rng.normal(size=nobs)
    else:  # pragma: no cover - closed case matrix
        raise ValueError(f"Unknown snapshot data id: {data_id}")

    # Keep the original base-data random stream fixed as this contract grows.
    # IV and frequency-weight coverage use a separate deterministic stream so
    # adding them cannot perturb established formula/SSC/GLM release cases.
    iv_rng = np.random.default_rng(AUGMENTATION_SEED)
    z1 = iv_rng.normal(size=nobs)
    endog_error = iv_rng.normal(size=nobs)
    x_endog = 0.9 * z1 + 0.6 * endog_error
    fweights = iv_rng.integers(1, 5, size=nobs)

    data = pd.DataFrame(
        {
            "Y": outcome,
            "Y_iv": linear + 0.8 * x_endog + 0.5 * endog_error,
            "X1": x1,
            "X2": x2,
            "X_endog": x_endog,
            "Z1": z1,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "weights": weights,
            "fweights": fweights,
        }
    )
    # This makes the complete-case SSC path observable even for formulas that
    # do not otherwise reference f3, matching the canonical SSC matrix setup.
    data.loc[data.index[:7], "f3"] = np.nan
    if variant == "complete":
        data = data.dropna().copy()
    elif variant != "full":  # pragma: no cover - closed case matrix
        raise ValueError(f"Unknown snapshot data variant: {variant}")
    return convert_f3(data, f3_type)


def _id(prefix: str, index: int, **parts: object) -> str:
    suffix = "-".join(f"{key}={value}" for key, value in parts.items())
    return f"{prefix}-{index:03d}-{suffix}" if suffix else f"{prefix}-{index:03d}"


def _ssc_cases() -> list[dict[str, Any]]:
    """Return the canonical exhaustive fit-time SSC matrix for feols/fepois."""
    formulas = (
        "Y ~ X1 + X2 + f1",
        "Y ~ X1 + X2 | f1",
        "Y ~ X1 + X2 | f2",
        "Y ~ X1 + X2 | f1 + f2",
        "Y ~ X1 + X2 | f1 + f2 + f3",
        # ``^`` is the release-compatible fixed-effect interaction spelling.
        # Current pyfixest still accepts it (with a deprecation warning).
        "Y ~ X1 + X2 | f1^f2",
    )
    cases: list[dict[str, Any]] = []
    index = 0
    for estimator, data_id in (("feols", "linear"), ("fepois", "count")):
        for formula_index, formula in enumerate(formulas):
            for variant in ("complete", "full"):
                for vcov_name in ("iid", "hetero", "f1", "f2", "f1+f2"):
                    if (
                        variant == "full"
                        and vcov_name not in {"iid", "hetero"}
                        and vcov_name not in formula
                    ):
                        continue
                    vcov: str | dict[str, str] = (
                        vcov_name
                        if vcov_name in {"iid", "hetero"}
                        else {"CRV1": vcov_name}
                    )
                    weight_options = (
                        (
                            (None, None),
                            ("weights", "aweights"),
                            ("fweights", "fweights"),
                        )
                        if estimator == "feols"
                        # 0.60.0 exposes fweights in the FEPoisson signature,
                        # but its IRLS dimensions fail (current issue #367).
                        # It remains outside this release contract.
                        else ((None, None), ("weights", "aweights"))
                    )
                    for weights, weights_type in weight_options:
                        for k_adj in (True, False):
                            for g_adj in (True, False):
                                for k_fixef in ("full", "none", "nonnested"):
                                    cases.append(
                                        {
                                            "id": _id(
                                                estimator,
                                                index,
                                                group="ssc",
                                                formula=formula_index,
                                                data=variant,
                                                vcov=vcov_name,
                                                weights=weights or "none",
                                                k_adj=k_adj,
                                                g_adj=g_adj,
                                                k_fixef=k_fixef,
                                            ),
                                            "estimator": estimator,
                                            "formula": formula,
                                            "data": data_id,
                                            "data_variant": variant,
                                            "kwargs": {
                                                "vcov": vcov,
                                                "weights": weights,
                                                **(
                                                    {"weights_type": weights_type}
                                                    if weights_type is not None
                                                    else {}
                                                ),
                                                "ssc": {
                                                    "k_adj": k_adj,
                                                    "G_adj": g_adj,
                                                    "G_df": "min",
                                                    "k_fixef": k_fixef,
                                                },
                                                **(
                                                    {
                                                        "iwls_tol": 1e-10,
                                                        "iwls_maxiter": 100,
                                                    }
                                                    if estimator == "fepois"
                                                    else {}
                                                ),
                                            },
                                        }
                                    )
                                    index += 1
    return cases


def build_cases() -> list[dict[str, Any]]:
    """Return every released, overlapping end-to-end public fit contract case."""
    cases: list[dict[str, Any]] = []
    release_formula_cases = tuple(
        (formula, f3_type)
        for formula, f3_type in FEOLS_FORMULA_F3_CASES
        # 0.60.0 cannot parse current canonical ``:`` fixed-effect interactions.
        # Formula interactions before the FE separator remain in the contract.
        if "|" not in formula or ":" not in formula.split("|", maxsplit=1)[1]
    )
    for index, (formula, f3_type) in enumerate(release_formula_cases):
        cases.append(
            {
                "id": _id("feols", index, group="formula", f3_type=f3_type),
                "estimator": "feols",
                "formula": formula,
                "data": "linear",
                "data_variant": "complete",
                "f3_type": f3_type,
                "kwargs": {"vcov": "hetero"},
            }
        )
    cases.extend(_ssc_cases())

    for index, (formula, vcov) in enumerate(
        (
            ("Y_iv ~ X1 + X2 | X_endog ~ Z1", "iid"),
            ("Y_iv ~ X1 + X2 | X_endog ~ Z1", "hetero"),
            ("Y_iv ~ X1 + X2 | X_endog ~ Z1", {"CRV1": "f1"}),
            ("Y_iv ~ X1 + X2 | f1 | X_endog ~ Z1", "iid"),
            ("Y_iv ~ X1 + X2 | f1 | X_endog ~ Z1", "hetero"),
            ("Y_iv ~ X1 + X2 | f1 | X_endog ~ Z1", {"CRV1": "f1"}),
        )
    ):
        for weights, weights_type in (
            (None, None),
            ("weights", "aweights"),
            ("fweights", "fweights"),
        ):
            cases.append(
                {
                    "id": _id(
                        "feols",
                        index,
                        group="iv",
                        fe=("yes" if "| f1 |" in formula else "no"),
                        vcov=("crv1" if isinstance(vcov, dict) else vcov),
                        weights=weights or "none",
                    ),
                    "estimator": "feols",
                    "formula": formula,
                    "data": "linear",
                    "data_variant": "complete",
                    "kwargs": {
                        "vcov": vcov,
                        "weights": weights,
                        **(
                            {"weights_type": weights_type}
                            if weights_type is not None
                            else {}
                        ),
                    },
                }
            )

    # 0.60.0 predates the development-head fepois-through-feglm API additions.
    # The overlapping feglm contract is therefore unweighted logit/probit only.
    for index, (family, formula, vcov) in enumerate(
        (
            ("logit", "Y ~ X1 + X2", "iid"),
            ("logit", "Y ~ X1 + X2 | f1", "hetero"),
            ("logit", "Y ~ X1 + X2 | f1", {"CRV1": "f1"}),
            ("probit", "Y ~ X1 + X2", "iid"),
            ("probit", "Y ~ X1 + X2 | f1", "hetero"),
            ("probit", "Y ~ X1 + X2 | f1", {"CRV1": "f1"}),
        )
    ):
        cases.append(
            {
                "id": _id(
                    "feglm",
                    index,
                    family=family,
                    vcov=("crv1" if isinstance(vcov, dict) else vcov),
                ),
                "estimator": "feglm",
                "formula": formula,
                "data": "binary",
                "data_variant": "complete",
                "kwargs": {
                    "family": family,
                    "vcov": vcov,
                    "iwls_tol": 1e-10,
                    "iwls_maxiter": 100,
                },
            }
        )

    for index, (formula, quantile, method, vcov) in enumerate(
        (
            ("Y ~ X1", 0.02, "fn", "nid"),
            ("Y ~ X1 + X2", 0.35, "pfn", "nid"),
            ("Y ~ X1", 0.5, "pfn", "nid"),
            ("Y ~ X1 + X2", 0.9, "fn", "nid"),
            ("Y ~ X1 + X2", 0.5, "fn", {"CRV1": "f1"}),
        )
    ):
        cases.append(
            {
                "id": _id(
                    "quantreg",
                    index,
                    quantile=quantile,
                    method=method,
                    vcov=("crv1" if isinstance(vcov, dict) else vcov),
                ),
                "estimator": "quantreg",
                "formula": formula,
                "data": "quantile",
                "data_variant": "complete",
                "kwargs": {
                    "quantile": quantile,
                    "method": method,
                    "vcov": vcov,
                    "tol": 1e-6,
                    "seed": 83838,
                    "ssc": {"k_adj": False, "G_adj": False},
                },
            }
        )
    return cases


def fast_case_ids(cases: list[dict[str, Any]]) -> frozenset[str]:
    """Select a small, reviewable edit tier without weakening the full contract."""
    formula_cases = (
        "Y~X1",
        "Y~X1+X2",
        "Y~X1|f2",
        "Y~X1|f2+f3",
        "Y ~ X1 + exp(X2)",
        "Y ~ X1 + C(f1)",
        "Y ~ X1 + i(f1, ref = 1)",
        "Y ~ X1 + X2:f1",
        "Y ~ X1 + i(f1,X2) | f2",
        "Y ~ X1 + I(X2 ** 2)",
        "Y ~ X1*X2 | f1+f2",
        "Y ~ X1 + poly(X2, 2) | f1",
        "log(Y) ~ X1",
    )
    selected: set[str] = set()
    for formula in formula_cases:
        selected.add(
            next(
                case["id"]
                for case in cases
                if case["estimator"] == "feols"
                and case["formula"] == formula
                and case.get("f3_type", "str") == "str"
            )
        )
    # One FE formula across all pandas f3 representations preserves that input
    # boundary without replaying every formula in every representation.
    for f3_type in ("str", "object", "int", "categorical", "float"):
        selected.add(
            next(
                case["id"]
                for case in cases
                if case["estimator"] == "feols"
                and case["formula"] == "Y~X1|f2+f3"
                and case.get("f3_type") == f3_type
            )
        )

    ssc_formulas = (
        "Y ~ X1 + X2 + f1",
        "Y ~ X1 + X2 | f1",
        "Y ~ X1 + X2 | f2",
        "Y ~ X1 + X2 | f1 + f2",
        "Y ~ X1 + X2 | f1 + f2 + f3",
        "Y ~ X1 + X2 | f1^f2",
    )
    ssc_variants = (
        ("complete", "iid", None, True, True, "full"),
        ("full", "hetero", "weights", False, False, "none"),
        ("complete", "f2", None, True, False, "nonnested"),
        ("complete", "f1+f2", "weights", False, True, "full"),
        ("complete", "f1", None, True, True, "none"),
        ("complete", "f1+f2", "weights", False, False, "nonnested"),
    )
    for estimator in ("feols", "fepois"):
        for formula, (variant, vcov_name, weights, k_adj, g_adj, k_fixef) in zip(
            ssc_formulas, ssc_variants, strict=True
        ):
            selected.add(
                next(
                    case["id"]
                    for case in cases
                    if case["estimator"] == estimator
                    and case["formula"] == formula
                    and case["data_variant"] == variant
                    and case["kwargs"]["weights"] == weights
                    and case["kwargs"]["ssc"]
                    == {
                        "k_adj": k_adj,
                        "G_adj": g_adj,
                        "G_df": "min",
                        "k_fixef": k_fixef,
                    }
                    and (
                        case["kwargs"]["vcov"] == vcov_name
                        or case["kwargs"]["vcov"] == {"CRV1": vcov_name}
                    )
                )
            )
        if estimator == "feols":
            # Keep released frequency weights in the fast tier without replaying
            # the entire SSC matrix. FEPoisson fweights are broken in 0.60.0.
            selected.add(
                next(
                    case["id"]
                    for case in cases
                    if case["estimator"] == estimator
                    and case["formula"] == ssc_formulas[0]
                    and case["kwargs"]["weights"] == "fweights"
                )
            )
    for weights in ("weights", "fweights"):
        selected.add(
            next(
                case["id"]
                for case in cases
                if case["estimator"] == "feols"
                and "group=iv" in case["id"]
                and case["kwargs"]["weights"] == weights
                and case["kwargs"]["vcov"] == "hetero"
            )
        )
    selected.update(
        case["id"] for case in cases if case["estimator"] in {"feglm", "quantreg"}
    )
    return frozenset(selected)


def fit_case(case: Mapping[str, Any]) -> Any:
    """Fit one declared case strictly through its public pyfixest entry point."""
    import pyfixest as pf

    data = build_data(
        str(case["data"]),
        variant=str(case["data_variant"]),
        f3_type=str(case.get("f3_type", "str")),
    )
    kwargs = dict(case["kwargs"])
    if "ssc" in kwargs:
        kwargs["ssc"] = pf.ssc(**kwargs["ssc"])
    return getattr(pf, str(case["estimator"]))(
        str(case["formula"]), data=data, **kwargs
    )


def _number(value: object) -> float | None:
    """Make non-finite released inference explicit and JSON-safe."""
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _named(series: pd.Series) -> dict[str, float | None]:
    return {str(name): _number(value) for name, value in series.sort_index().items()}


def extract_snapshot(fit: Any) -> dict[str, Any]:
    """Extract stable named values without serialising a fitted model or repr."""
    coef = fit.coef()
    confint = fit.confint().sort_index()
    names = list(coef.index)
    vcov = np.asarray(fit._vcov)
    order = [names.index(name) for name in sorted(names)]
    sorted_names = sorted(names)
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
    if getattr(fit, "deviance", None) is not None:
        metadata["deviance"] = float(fit.deviance)
    vcov_by_name = {
        row_name: {
            column_name: _number(vcov[row_position, column_position])
            for column_name, column_position in zip(sorted_names, order, strict=True)
        }
        for row_name, row_position in zip(sorted_names, order, strict=True)
    }
    return {
        "coef": _named(coef),
        "vcov": vcov_by_name,
        "se": _named(fit.se()),
        "tstat": _named(fit.tstat()),
        "pvalue": _named(fit.pvalue()),
        "confint": {
            str(name): {str(column): _number(value) for column, value in row.items()}
            for name, row in confint.to_dict(orient="index").items()
        },
        "metadata": metadata,
        "resid_sample": [
            _number(value) for value in np.asarray(fit.resid()).ravel()[:SAMPLE_SIZE]
        ],
        "predict_sample": predict_sample,
    }


def load_json(path: Path) -> dict[str, Any]:
    """Read one committed JSON artifact."""
    return json.loads(path.read_text())
