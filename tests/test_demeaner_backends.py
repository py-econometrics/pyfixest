from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from tests._feols_test_cases import FEOLS_FORMULA_F3_CASES, convert_f3
from tests._torch_test_utils import torch_param

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:The torch LSMR demeaner backend is experimental:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:No GPU available .* torch demeaning will run on CPU:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:Sparse CSR tensor support is in beta state:UserWarning"
    ),
]


@dataclass(frozen=True)
class BackendCase:
    name: str
    demeaner: pf.MapDemeaner | pf.LsmrDemeaner
    coef_tol: float
    predict_tol: float
    resid_tol: float
    inference_tol: float
    tstat_tol: float


REFERENCE_DEMEANER = pf.MapDemeaner(backend="rust")

BACKEND_CASES = [
    pytest.param(
        BackendCase(
            name="numba",
            demeaner=pf.MapDemeaner(backend="numba"),
            coef_tol=1e-8,
            predict_tol=1e-6,
            resid_tol=1e-6,
            inference_tol=1e-7,
            tstat_tol=1e-6,
        ),
        id="numba",
    ),
    pytest.param(
        BackendCase(
            name="within_additive",
            demeaner=pf.LsmrDemeaner(preconditioner="additive"),
            coef_tol=1e-8,
            predict_tol=1e-6,
            resid_tol=1e-6,
            # Full covariance matrices amplify LSMR stopping error in large
            # factor expansions more than the historical X1 diagonal check.
            inference_tol=1e-6,
            tstat_tol=1e-6,
        ),
        id="within_additive",
    ),
    pytest.param(
        BackendCase(
            name="within_diagonal",
            demeaner=pf.LsmrDemeaner(preconditioner="diagonal"),
            coef_tol=1e-8,
            predict_tol=1e-6,
            resid_tol=1e-6,
            # Full covariance matrices amplify LSMR stopping error in large
            # factor expansions more than the historical X1 diagonal check.
            inference_tol=1e-6,
            tstat_tol=1e-6,
        ),
        id="within_diagonal",
    ),
    torch_param(
        BackendCase(
            name="torch",
            demeaner=pf.LsmrDemeaner(backend="torch", device="auto"),
            coef_tol=1e-8,
            predict_tol=5e-5,
            resid_tol=5e-5,
            inference_tol=1e-6,
            tstat_tol=1e-5,
        ),
        id="torch",
    ),
    torch_param(
        BackendCase(
            name="torch_cpu",
            demeaner=pf.LsmrDemeaner(backend="torch", device="cpu"),
            coef_tol=1e-8,
            predict_tol=5e-5,
            resid_tol=5e-5,
            inference_tol=1e-6,
            tstat_tol=1e-5,
        ),
        id="torch_cpu",
    ),
    torch_param(
        BackendCase(
            name="torch_mps",
            demeaner=pf.LsmrDemeaner(
                backend="torch", precision="float32", device="mps"
            ),
            coef_tol=5e-6,
            predict_tol=2e-4,
            resid_tol=2e-4,
            inference_tol=1e-5,
            tstat_tol=1e-5,
        ),
        id="torch_mps",
        require="mps",
    ),
    torch_param(
        BackendCase(
            name="torch_cuda",
            demeaner=pf.LsmrDemeaner(backend="torch", device="cuda"),
            coef_tol=1e-8,
            predict_tol=5e-5,
            resid_tol=5e-5,
            inference_tol=1e-6,
            tstat_tol=1e-5,
        ),
        id="torch_cuda",
        require="cuda",
    ),
    torch_param(
        BackendCase(
            name="torch_cuda32",
            demeaner=pf.LsmrDemeaner(
                backend="torch", precision="float32", device="cuda"
            ),
            coef_tol=5e-6,
            predict_tol=2e-4,
            resid_tol=2e-4,
            inference_tol=1e-5,
            tstat_tol=1e-5,
        ),
        id="torch_cuda32",
        require="cuda",
    ),
]


@pytest.fixture(scope="module")
def backend_data() -> dict[tuple[bool, str], pd.DataFrame]:
    base = pf.get_data(
        N=1000,
        seed=76540251,
        beta_type="2",
        error_type="2",
        model="Feols",
    )
    variants = {}
    for dropna in (False, True):
        for f3_type in ("str", "object", "int", "categorical", "float"):
            data = base.dropna() if dropna else base.copy()
            data.where(data != "nan", np.nan, inplace=True)
            variants[(dropna, f3_type)] = convert_f3(data, f3_type)
    return variants


def _assert_backend_matches(
    actual,
    reference,
    case: BackendCase,
) -> None:
    context = f"backend={case.name}"

    assert actual._coefnames == reference._coefnames, context
    assert actual._collin_vars == reference._collin_vars, context
    assert actual._N == reference._N, context
    assert actual._df_k == reference._df_k, context
    assert actual._df_t == reference._df_t, context

    np.testing.assert_allclose(
        actual.coef(),
        reference.coef(),
        rtol=0,
        atol=case.coef_tol,
        err_msg=f"coefficients differ for {context}",
    )
    np.testing.assert_allclose(
        actual._vcov,
        reference._vcov,
        rtol=0,
        atol=case.inference_tol,
        err_msg=f"vcov differs for {context}",
    )
    np.testing.assert_allclose(
        actual.se(),
        reference.se(),
        rtol=0,
        atol=case.inference_tol,
        err_msg=f"standard errors differ for {context}",
    )
    np.testing.assert_allclose(
        actual.tstat(),
        reference.tstat(),
        rtol=0,
        atol=case.tstat_tol,
        err_msg=f"t statistics differ for {context}",
    )
    np.testing.assert_allclose(
        actual.pvalue(),
        reference.pvalue(),
        rtol=0,
        atol=case.inference_tol,
        err_msg=f"p-values differ for {context}",
    )
    np.testing.assert_allclose(
        actual.confint(),
        reference.confint(),
        rtol=0,
        atol=case.inference_tol,
        err_msg=f"confidence intervals differ for {context}",
    )
    np.testing.assert_allclose(
        actual.resid()[:5],
        reference.resid()[:5],
        rtol=0,
        atol=case.resid_tol,
        err_msg=f"residuals differ for {context}",
    )
    np.testing.assert_allclose(
        actual.predict()[:5],
        reference.predict()[:5],
        rtol=0,
        atol=case.predict_tol,
        err_msg=f"predictions differ for {context}",
    )
    np.testing.assert_allclose(
        [actual._r2, actual._adj_r2, actual._r2_within, actual._adj_r2_within],
        [
            reference._r2,
            reference._adj_r2,
            reference._r2_within,
            reference._adj_r2_within,
        ],
        rtol=0,
        atol=case.inference_tol,
        equal_nan=True,
        err_msg=f"fit statistics differ for {context}",
    )


@pytest.mark.parametrize("dropna", [False, True])
@pytest.mark.parametrize("inference", ["iid", "hetero", {"CRV1": "group_id"}])
@pytest.mark.parametrize("weights", [None, "weights"])
@pytest.mark.parametrize("case", BACKEND_CASES)
@pytest.mark.parametrize(
    "fml,f3_type",
    FEOLS_FORMULA_F3_CASES,
    ids=lambda value: str(value),
)
def test_feols_demeaner_backends_match_reference(
    backend_data,
    dropna,
    inference,
    weights,
    case,
    fml,
    f3_type,
):
    data = backend_data[(dropna, f3_type)]
    fit_kwargs = {
        "fml": fml,
        "data": data,
        "vcov": inference,
        "weights": weights,
        "ssc": pf.ssc(k_adj=True, G_adj=True),
    }

    reference = pf.feols(**fit_kwargs, demeaner=REFERENCE_DEMEANER)
    actual = pf.feols(**fit_kwargs, demeaner=case.demeaner)
    _assert_backend_matches(actual, reference, case)
