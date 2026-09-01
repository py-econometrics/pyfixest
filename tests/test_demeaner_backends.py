from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from tests._feols_test_cases import FEOLS_FORMULA_F3_CASES, build_feols_data_variants
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
    coef_rtol: float
    coef_tol: float
    predict_tol: float
    resid_tol: float
    inference_tol: float
    tstat_tol: float
    confint_rtol: float
    confint_tol: float


REFERENCE_DEMEANER = pf.MapDemeaner(backend="rust")

BACKEND_CASES = [
    pytest.param(
        BackendCase(
            name="numba",
            demeaner=pf.MapDemeaner(backend="numba"),
            coef_rtol=0,
            coef_tol=1e-8,
            predict_tol=1e-6,
            resid_tol=1e-6,
            inference_tol=1e-7,
            tstat_tol=1e-6,
            confint_rtol=0,
            confint_tol=1e-7,
        ),
        id="numba",
    ),
    pytest.param(
        BackendCase(
            name="within_additive",
            demeaner=pf.LsmrDemeaner(preconditioner="additive"),
            coef_rtol=0,
            coef_tol=1e-8,
            predict_tol=1e-6,
            resid_tol=1e-6,
            # Full covariance matrices amplify LSMR stopping error in large
            # factor expansions more than the historical X1 diagonal check.
            inference_tol=1e-6,
            tstat_tol=1e-6,
            confint_rtol=0,
            confint_tol=1e-6,
        ),
        id="within_additive",
    ),
    pytest.param(
        BackendCase(
            name="within_diagonal",
            demeaner=pf.LsmrDemeaner(preconditioner="diagonal"),
            coef_rtol=0,
            coef_tol=1e-8,
            predict_tol=1e-6,
            resid_tol=1e-6,
            # Full covariance matrices amplify LSMR stopping error in large
            # factor expansions more than the historical X1 diagonal check.
            inference_tol=1e-6,
            tstat_tol=1e-6,
            confint_rtol=0,
            confint_tol=1e-6,
        ),
        id="within_diagonal",
    ),
    torch_param(
        BackendCase(
            name="torch",
            demeaner=pf.LsmrDemeaner(backend="torch", device="auto"),
            coef_rtol=0,
            coef_tol=1e-8,
            predict_tol=5e-5,
            resid_tol=5e-5,
            inference_tol=1e-6,
            tstat_tol=1e-5,
            confint_rtol=0,
            confint_tol=1e-6,
        ),
        id="torch",
    ),
    torch_param(
        BackendCase(
            name="torch_cpu",
            demeaner=pf.LsmrDemeaner(backend="torch", device="cpu"),
            coef_rtol=0,
            coef_tol=1e-8,
            predict_tol=5e-5,
            resid_tol=5e-5,
            inference_tol=1e-6,
            tstat_tol=1e-5,
            confint_rtol=0,
            confint_tol=1e-6,
        ),
        id="torch_cpu",
    ),
    torch_param(
        BackendCase(
            name="torch_mps",
            demeaner=pf.LsmrDemeaner(
                backend="torch", precision="float32", device="mps"
            ),
            # Preserve a strict absolute guard near zero while allowing the
            # expected float32 rounding scale for large coefficients.
            coef_rtol=1e-6,
            # Float32 LSMR stopping error is amplified by large factor
            # expansions; the observed coefficient error is below 2e-4.
            coef_tol=2e-4,
            predict_tol=2e-4,
            resid_tol=2e-4,
            # Full covariance matrices and derived inference amplify the
            # float32 residualization error more than point estimates.
            inference_tol=2e-4,
            tstat_tol=2e-4,
            # Confidence bounds combine coefficient and standard-error error.
            confint_rtol=2e-6,
            confint_tol=3e-4,
        ),
        id="torch_mps",
        require="mps",
    ),
    torch_param(
        BackendCase(
            name="torch_cuda",
            demeaner=pf.LsmrDemeaner(backend="torch", device="cuda"),
            coef_rtol=0,
            coef_tol=1e-8,
            predict_tol=5e-5,
            resid_tol=5e-5,
            inference_tol=1e-6,
            tstat_tol=1e-5,
            confint_rtol=0,
            confint_tol=1e-6,
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
            coef_rtol=0,
            coef_tol=5e-6,
            predict_tol=2e-4,
            resid_tol=2e-4,
            inference_tol=1e-5,
            tstat_tol=1e-5,
            confint_rtol=0,
            confint_tol=1e-5,
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
    return build_feols_data_variants(base)


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
        rtol=case.coef_rtol,
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
        rtol=case.confint_rtol,
        atol=case.confint_tol,
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


def _assert_backend_prediction_paths_match(
    actual,
    reference,
    data: pd.DataFrame,
    case: BackendCase,
) -> None:
    context = f"backend={case.name}"
    newdata = data.iloc[:100]

    actual_prediction = actual.predict(newdata=newdata, atol=1e-12, btol=1e-12)
    reference_prediction = reference.predict(
        newdata=newdata,
        atol=1e-12,
        btol=1e-12,
    )
    np.testing.assert_array_equal(
        np.isnan(actual_prediction),
        np.isnan(reference_prediction),
        err_msg=f"new-data prediction missingness differs for {context}",
    )
    observed = ~np.isnan(reference_prediction)
    np.testing.assert_allclose(
        actual_prediction[observed][:5],
        reference_prediction[observed][:5],
        rtol=0,
        atol=case.predict_tol,
        err_msg=f"new-data predictions differ for {context}",
    )

    if not actual._has_fixef and not actual._has_weights:
        np.testing.assert_allclose(
            actual.predict(se_fit=True),
            reference.predict(se_fit=True),
            rtol=0,
            atol=case.predict_tol,
            err_msg=f"prediction standard errors differ for {context}",
        )
        np.testing.assert_allclose(
            actual.predict(newdata=newdata, se_fit=True),
            reference.predict(newdata=newdata, se_fit=True),
            rtol=0,
            atol=case.predict_tol,
            equal_nan=True,
            err_msg=f"new-data prediction standard errors differ for {context}",
        )

        actual_interval = actual.predict(interval="prediction")
        reference_interval = reference.predict(interval="prediction")
        for column in ["fit", "se_fit", "ci_low", "ci_high"]:
            np.testing.assert_allclose(
                actual_interval[column].to_numpy()[-4:],
                reference_interval[column].to_numpy()[-4:],
                rtol=0,
                atol=case.predict_tol,
                err_msg=(f"prediction intervals differ for {context}, column={column}"),
            )

        actual_interval = actual.predict(
            newdata=newdata,
            interval="prediction",
        )
        reference_interval = reference.predict(
            newdata=newdata,
            interval="prediction",
        )
        for column in ["fit", "se_fit", "ci_low", "ci_high"]:
            np.testing.assert_allclose(
                actual_interval[column].to_numpy()[-4:],
                reference_interval[column].to_numpy()[-4:],
                rtol=0,
                atol=case.predict_tol,
                equal_nan=True,
                err_msg=(
                    "new-data prediction intervals differ for "
                    f"{context}, column={column}"
                ),
            )
    else:
        with pytest.raises(NotImplementedError):
            actual.predict(se_fit=True)
        with pytest.raises(NotImplementedError):
            actual.predict(interval="prediction")
        with pytest.raises(NotImplementedError):
            actual.predict(newdata=newdata, se_fit=True)
        with pytest.raises(NotImplementedError):
            actual.predict(newdata=newdata, interval="prediction")


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
    if inference == "iid":
        _assert_backend_prediction_paths_match(
            actual=actual,
            reference=reference,
            data=data,
            case=case,
        )
