"""Wall-clock benchmarks for single-model estimation with the LSMR demeaner.

Each case fits one model on the simple and the difficult `base_dgp` panel
with the `within` LSMR demeaner under the additive and the diagonal
preconditioner. Run with `pixi run -e py312 bench-estimation`; compare a
branch against a saved run as described in `docs/developer/testing.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

# pytest puts `benchmarks/` on `sys.path`, where `benchmarks/benchmarks.py`
# shadows the `benchmarks` package, so import the DGP relative to it.
from modular.dgp_functions import base_dgp

import pyfixest as pf

SEED = 20261002
N_OBS = 100_000
N_OBS_POISSON = 50_000
N_OBS_OVERHEAD = 1_000
DGPS = ("simple", "difficult")
PRECONDITIONERS = ("additive", "diagonal")

FE2 = "indiv_id + year"
FE3 = "indiv_id + year + firm_id"
CRV1 = {"CRV1": "indiv_id"}


def _covariates(k: int) -> str:
    return " + ".join(f"x{j}" for j in range(1, k + 1))


@cache
def benchmark_data(dgp: str, n: int) -> pd.DataFrame:
    """Build the `base_dgp` panel plus an instrument for `x1` and weights."""
    data = base_dgp(n=n, type_=dgp, k=5, seed=SEED)
    rng = np.random.default_rng(SEED + 1)
    data["z1"] = data["x1"] + rng.standard_normal(len(data))
    data["weights"] = rng.uniform(0.5, 2.0, size=len(data))
    return data


@dataclass(frozen=True)
class Case:
    """One timed estimation call, fitted on every DGP and preconditioner."""

    id: str
    estimator: str
    fml: str
    vcov: str | dict[str, str]
    kwargs: dict[str, Any] = field(default_factory=dict)
    n: int = N_OBS


def _ols_cases() -> list[Case]:
    return [
        Case(f"ols-k{k}-{fe_id}-iid", "feols", f"y ~ {_covariates(k)} | {fe}", "iid")
        for k in (1, 5, 10)
        for fe_id, fe in (("fe2", FE2), ("fe3", FE3))
    ]


def _poisson_cases() -> list[Case]:
    return [
        Case(
            f"pois-k{k}-{fe_id}-iid",
            "fepois",
            f"negbin_y ~ {_covariates(k)} | {fe}",
            "iid",
            n=N_OBS_POISSON,
        )
        for k in (1, 5)
        for fe_id, fe in (("fe2", FE2), ("fe3", FE3))
    ]


CASES = [
    *_ols_cases(),
    # The one clustered case: the vcov step is a small share of every fit.
    Case("ols-k5-fe3-crv1", "feols", f"y ~ {_covariates(5)} | {FE3}", CRV1),
    *_poisson_cases(),
    Case(
        "logit-k5-fe2-iid",
        "feglm",
        f"binary_y ~ {_covariates(5)} | {FE2}",
        "iid",
        {"family": "logit"},
    ),
    Case("iv-fe2-iid", "feols", f"y ~ x2 + x3 + [x1 ~ z1] | {FE2}", "iid"),
    Case("iv-fe3-iid", "feols", f"y ~ x2 + x3 + [x1 ~ z1] | {FE3}", "iid"),
    Case(
        "wls-k5-fe2-iid",
        "feols",
        f"y ~ {_covariates(5)} | {FE2}",
        "iid",
        {"weights": "weights"},
    ),
    Case(
        "overhead-ols-k5-fe2-iid",
        "feols",
        f"y ~ {_covariates(5)} | {FE2}",
        "iid",
        n=N_OBS_OVERHEAD,
    ),
]
CASES_BY_ID = {case.id: case for case in CASES}
# OLS fits take milliseconds and need more rounds for a stable median than the
# IRLS fits, which take seconds.
ROUNDS = {"feols": 5, "fepois": 3, "feglm": 3}


@pytest.mark.parametrize("preconditioner", PRECONDITIONERS)
@pytest.mark.parametrize("dgp", DGPS)
@pytest.mark.parametrize("case_id", CASES_BY_ID)
def test_estimation(benchmark, case_id: str, dgp: str, preconditioner: str):
    """Time one fit after one warm-up round."""
    case = CASES_BY_ID[case_id]
    data = benchmark_data(dgp, case.n)
    estimator = getattr(pf, case.estimator)

    def fit():
        # A fresh demeaner per round, so no round reuses another's preconditioner.
        demeaner = pf.LsmrDemeaner(backend="within", preconditioner=preconditioner)
        return estimator(
            case.fml, data, vcov=case.vcov, demeaner=demeaner, **case.kwargs
        )

    benchmark.group = case_id
    benchmark.extra_info["fml"] = case.fml
    benchmark.pedantic(fit, rounds=ROUNDS[case.estimator], warmup_rounds=1)
