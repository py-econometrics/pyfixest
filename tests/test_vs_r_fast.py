import numpy as np
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import pyfixest as pf

fixest = importr("fixest")
quantreg_r = importr("quantreg")
stats = importr("stats")

pytestmark = [pytest.mark.against_r_core, pytest.mark.fast_r]

# Keep these representative bounds aligned with the estimator-specific
# contracts in test_vs_fixest.py: coefficients are strict, while IRLS and
# derived inference allow the slightly larger numerical differences documented
# by the canonical matrix.
COEFFICIENT_ATOL = 1e-8
FEOLS_INFERENCE_ATOL = 1e-7
GLM_INFERENCE_ATOL = 1e-6

# The compact fixest matrix covers no-FE/IID, weighted-FE/hetero, and FE/CRV1
# paths across feols, fepois, and feglm. Quantreg separately covers both
# supported solvers and low, interior, median, and high quantiles; it does not
# support fixed effects.
FIXEST_CASES = [
    pytest.param("feols", "Y ~ X1 + X2", None, "iid", None, id="feols-iid"),
    pytest.param(
        "feols",
        "Y ~ X1 + X2 | f1 + f2",
        None,
        "hetero",
        "weights",
        id="feols-fe-hetero-weighted",
    ),
    pytest.param(
        "feols",
        "Y ~ X1 + X2 | f1 + f2",
        None,
        {"CRV1": "group_id"},
        None,
        id="feols-fe-clustered",
    ),
    pytest.param("fepois", "Y ~ X1 + X2", None, "iid", None, id="fepois-iid"),
    pytest.param(
        "fepois",
        "Y ~ X1 + X2 | f1",
        None,
        "hetero",
        "weights",
        id="fepois-fe-hetero-weighted",
    ),
    pytest.param(
        "fepois",
        "Y ~ X1 + X2 | f1",
        None,
        {"CRV1": "group_id"},
        None,
        id="fepois-fe-clustered",
    ),
    pytest.param(
        "feglm", "Y_bin ~ X1 + X2", "logit", "iid", None, id="feglm-logit-iid"
    ),
    pytest.param(
        "feglm",
        "Y_bin ~ X1 + X2 | f1",
        "probit",
        "hetero",
        "weights",
        id="feglm-probit-fe-weighted",
    ),
    pytest.param(
        "feglm",
        "Y ~ X1 + X2 | f1",
        "poisson",
        {"CRV1": "group_id"},
        None,
        id="feglm-poisson-fe-clustered",
    ),
]
FIXEST_TOLERANCES = {
    "feols": (FEOLS_INFERENCE_ATOL, 1e-6, 1e-6, {"link": 1e-6}),
    "fepois": (
        GLM_INFERENCE_ATOL,
        1e-6,
        1e-7,
        {"link": 1e-6, "response": 1e-5},
    ),
    "feglm": (
        GLM_INFERENCE_ATOL,
        1e-6,
        1e-7,
        {"link": 1e-6, "response": 1e-5},
    ),
}
QUANTREG_CASES = [
    pytest.param("Y ~ X1", 0.02, "fn", id="fn-low"),
    pytest.param("Y ~ X1 + X2", 0.35, "pfn", id="pfn-two-regressors"),
    pytest.param("Y ~ X1", 0.5, "pfn", id="pfn-median"),
    pytest.param("Y ~ X1 + X2", 0.9, "fn", id="fn-high-two-regressors"),
]


@pytest.fixture(scope="module")
def linear_data():
    return pf.get_data(N=500, seed=76540251, model="Feols").dropna()


@pytest.fixture(scope="module")
def count_data():
    data = pf.get_data(N=500, seed=7651, model="Fepois").dropna()
    rng = np.random.default_rng(20260825)
    latent = 0.5 + 0.8 * data["X1"] - 0.4 * data["X2"]
    data["Y_bin"] = rng.binomial(1, 1 / (1 + np.exp(-latent)))
    return data


@pytest.fixture(scope="module")
def quantile_data():
    data = pf.get_data(N=5_000, seed=3131)
    rng = np.random.default_rng(3993)
    data["Y"] = 1 + 2 * data["X1"] + 3 * data["X2"] + rng.normal(size=len(data))
    return data


def _assert_close(actual, reference, *, atol, quantity, rtol=0):
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(reference),
        rtol=rtol,
        atol=atol,
        err_msg=f"{quantity} differ from the R reference",
    )


def _r_coefficient_names(r_fit) -> list[str]:
    ro.globalenv[".pyfixest_fast_fit"] = r_fit
    names = ro.r("names(coef(.pyfixest_fast_fit))")
    return ["Intercept" if name == "(Intercept)" else str(name) for name in names]


def _r_inference(inference):
    if isinstance(inference, dict):
        return ro.Formula(f"~{inference['CRV1']}")
    return inference


def _r_ssc():
    return fixest.ssc(True, "nonnested", False, True, "min", "min")


def _assert_fixest_contract(py_fit, r_fit, *, inference_atol, derived_atol):
    r_names = _r_coefficient_names(r_fit)
    r_table = np.asarray(r_fit.rx2("coeftable"))
    py_positions = [py_fit.coef().index.get_loc(name) for name in r_names]

    assert set(py_fit.coef().index) == set(r_names), (
        "estimated coefficient names differ from R fixest"
    )
    _assert_close(
        py_fit.coef().loc[r_names],
        r_table[:, 0],
        atol=COEFFICIENT_ATOL,
        quantity="coefficient estimates",
    )
    _assert_close(
        py_fit._vcov[np.ix_(py_positions, py_positions)],
        stats.vcov(r_fit),
        atol=inference_atol,
        quantity="covariance matrices",
    )
    _assert_close(
        py_fit.se().loc[r_names],
        r_table[:, 1],
        atol=inference_atol,
        quantity="standard errors",
    )
    _assert_close(
        py_fit.tstat().loc[r_names],
        r_table[:, 2],
        atol=derived_atol,
        quantity="test statistics",
    )
    _assert_close(
        py_fit.pvalue().loc[r_names],
        r_table[:, 3],
        atol=derived_atol,
        quantity="p-values",
    )
    _assert_close(
        py_fit.confint().loc[r_names],
        np.asarray(stats.confint(r_fit)).T,
        atol=derived_atol,
        quantity="confidence intervals",
    )

    ro.globalenv[".pyfixest_fast_fit"] = r_fit
    assert int(py_fit._df_k) == int(
        ro.r('attr(.pyfixest_fast_fit$cov.scaled, "df.K")')[0]
    ), "model degrees of freedom differ from R fixest"
    assert int(py_fit._df_t) == int(
        ro.r('attr(.pyfixest_fast_fit$cov.scaled, "df.t")')[0]
    ), "inference degrees of freedom differ from R fixest"
    assert int(stats.nobs(r_fit)[0]) == py_fit._N, (
        "observation count differs from R fixest"
    )


def _assert_fit_samples(
    py_fit, r_fit, *, residual_atol, prediction_atols: dict[str, float]
):
    _assert_close(
        py_fit.resid()[:5],
        np.asarray(stats.residuals(r_fit))[:5],
        atol=residual_atol,
        quantity="first five residuals",
    )
    for prediction_type, prediction_atol in prediction_atols.items():
        _assert_close(
            py_fit.predict(type=prediction_type)[:5],
            np.asarray(stats.predict(r_fit, type=prediction_type))[:5],
            atol=prediction_atol,
            quantity=f"first five {prediction_type} predictions",
        )


def _fit_fixest_pair(estimator, fml, family, inference, weights, data):
    py_kwargs = {
        "fml": fml,
        "data": data,
        "vcov": inference,
        "weights": weights,
        "ssc": pf.ssc(k_adj=True, G_adj=True),
    }
    r_kwargs = {
        "data": data,
        "vcov": _r_inference(inference),
        "ssc": _r_ssc(),
    }
    if estimator in ("fepois", "feglm"):
        py_kwargs.update(iwls_tol=1e-10, iwls_maxiter=100)
        r_kwargs.update(glm_tol=1e-10, glm_iter=100)
    if family is not None:
        py_kwargs["family"] = family
        r_kwargs["family"] = {
            "logit": stats.binomial(link="logit"),
            "probit": stats.binomial(link="probit"),
            "poisson": stats.poisson(),
        }[family]
    if weights is not None:
        r_kwargs["weights"] = ro.Formula(f"~{weights}")
    py_fit = getattr(pf, estimator)(**py_kwargs)
    r_fit = getattr(fixest, estimator)(ro.Formula(fml), **r_kwargs)
    return py_fit, r_fit


@pytest.mark.parametrize(
    ("estimator", "fml", "family", "inference", "weights"), FIXEST_CASES
)
def test_fast_against_fixest(request, estimator, fml, family, inference, weights):
    data_fixture = "linear_data" if estimator == "feols" else "count_data"
    data = request.getfixturevalue(data_fixture)
    py_fit, r_fit = _fit_fixest_pair(
        estimator=estimator,
        fml=fml,
        family=family,
        inference=inference,
        weights=weights,
        data=data,
    )
    inference_atol, derived_atol, residual_atol, prediction_atols = FIXEST_TOLERANCES[
        estimator
    ]
    if weights is not None and estimator != "feols":
        derived_atol = 1e-5
    _assert_fixest_contract(
        py_fit, r_fit, inference_atol=inference_atol, derived_atol=derived_atol
    )
    _assert_fit_samples(
        py_fit,
        r_fit,
        residual_atol=residual_atol,
        prediction_atols=prediction_atols,
    )


@pytest.mark.parametrize(("fml", "quantile", "method"), QUANTREG_CASES)
def test_quantreg_fast_against_r_quantreg(quantile_data, fml, quantile, method):
    # Match the solver-specific contract in test_quantreg.py rather than the
    # tighter absolute-error contract used for fixest estimators above.
    py_fit = pf.quantreg(
        fml=fml,
        data=quantile_data,
        quantile=quantile,
        method=method,
        vcov="nid",
        tol=1e-6,
        ssc=pf.ssc(k_adj=False, G_adj=False),
        seed=83838,
    )
    r_fit = quantreg_r.rq(
        ro.Formula(fml),
        data=quantile_data,
        tau=quantile,
        method=method,
        eps=1e-6,
    )
    r_names = _r_coefficient_names(r_fit)
    assert set(py_fit.coef().index) == set(r_names), (
        "estimated coefficient names differ from R quantreg"
    )
    _assert_close(
        py_fit.coef().loc[r_names],
        r_fit.rx2("coefficients"),
        rtol=1e-3,
        atol=1e-6,
        quantity="quantile-regression coefficients",
    )
    r_summary = ro.r["summary"](r_fit, se="nid")
    _assert_close(
        py_fit.se().loc[r_names],
        np.asarray(r_summary.rx2("coefficients"))[:, 1],
        rtol=1e-3,
        atol=1e-6,
        quantity="quantile-regression standard errors",
    )
    if method == "fn":
        r_residuals = np.asarray(r_fit.rx2("residuals"))
        _assert_close(
            py_fit.resid()[:5],
            r_residuals[:5],
            rtol=1e-3,
            atol=1e-8,
            quantity="first five quantile-regression residuals",
        )
        r_objective = np.sum(np.abs(r_residuals) * (quantile - (r_residuals < 0)))
        _assert_close(
            py_fit.objective_value,
            r_objective,
            rtol=1e-6,
            atol=1e-8,
            quantity="quantile-regression objective",
        )
