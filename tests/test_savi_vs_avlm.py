import numpy as np
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.estimation import feols
from pyfixest.estimation.post_estimation.savi import optimal_mixture_precision
from pyfixest.estimation.post_estimation.wald import _wald_statistic
from pyfixest.utils.check_r_install import check_r_install

_HAS_AVLM = check_r_install("avlm", strict=False)
if _HAS_AVLM:
    avlm = importr("avlm")
    # avlm prints the overall F-test p-value but does not store it in summary().
    _R_AVLM_RESULTS = ro.r(
        """
        function(formula, data, g, vcov_estimator, alpha) {
            fit <- stats::lm(formula, data = data)
            if (is.null(vcov_estimator)) {
                av_fit <- avlm::av(fit, g = g)
            } else {
                av_fit <- avlm::av(
                    fit,
                    g = g,
                    vcov_estimator = vcov_estimator
                )
            }
            av_summary <- summary(av_fit)
            f_statistic <- unname(av_summary$fstatistic)
            log_g_f <- getFromNamespace("log_G_f", "avlm")
            p_g_f <- getFromNamespace("p_G_f", "avlm")
            f_pvalue <- p_g_f(
                log_g_f(
                    f_statistic[1],
                    f_statistic[2],
                    f_statistic[3],
                    stats::nobs(fit),
                    g
                )
            )

            list(
                pvalues = unname(av_summary$coefficients[, 4]),
                confint = unname(confint(av_fit, level = 1 - alpha)),
                f_statistic = f_statistic,
                f_pvalue = unname(f_pvalue)
            )
        }
        """
    )


pytestmark = [
    pytest.mark.against_r_extended,
    pytest.mark.skipif(not _HAS_AVLM, reason="R package avlm is not installed."),
]

_FORMULAS = ["Y ~ X1 + X2", "Y ~ X1 + X2 + Z1"]
_VCOV_TYPES = ["iid", "HC1"]
_MIXTURE_PRECISIONS = [1.0, 2.5, "optimal"]
# Cross-language linear algebra agrees to near machine precision on this data.
_INFERENCE_RTOL = 1e-10
# R and SciPy use different bounded optimizers with slightly different stopping rules.
_OPTIMIZER_RTOL = 1e-6


@pytest.fixture(scope="module")
def savi_data():
    return (
        pf.get_data()
        .dropna(subset=["Y", "X1", "X2", "Z1"])
        .loc[:, ["Y", "X1", "X2", "Z1"]]
        .head(200)
        .reset_index(drop=True)
    )


@pytest.mark.parametrize("formula", _FORMULAS)
@pytest.mark.parametrize("vcov", _VCOV_TYPES)
@pytest.mark.parametrize("mixture_precision", _MIXTURE_PRECISIONS)
def test_savi_coefficient_inference_matches_avlm(
    savi_data, formula, vcov, mixture_precision
):
    alpha = 0.05
    fit = feols(formula, savi_data, vcov=vcov)
    if mixture_precision == "optimal":
        mixture_precision = optimal_mixture_precision(fit._N, fit._k, alpha)

    r_vcov = ro.NULL if vcov == "iid" else vcov
    r_results = _R_AVLM_RESULTS(
        ro.Formula(formula),
        savi_data,
        mixture_precision,
        r_vcov,
        alpha,
    )

    np.testing.assert_allclose(
        fit.pvalue(
            inference_type="savi",
            mixture_precision=mixture_precision,
        ),
        np.asarray(r_results.rx2("pvalues")),
        rtol=_INFERENCE_RTOL,
    )
    np.testing.assert_allclose(
        fit.confint(
            alpha=alpha,
            inference_type="savi",
            mixture_precision=mixture_precision,
        ),
        np.asarray(r_results.rx2("confint")),
        rtol=_INFERENCE_RTOL,
    )

    if vcov == "iid":
        restriction = np.eye(fit._k)[1:]
        wald_statistic, dfn = _wald_statistic(
            beta_hat=fit._beta_hat,
            vcov=fit._vcov,
            R=restriction,
        )
        f_statistic = wald_statistic / dfn
        r_f_statistic = np.asarray(r_results.rx2("f_statistic"))

        np.testing.assert_allclose(
            f_statistic,
            r_f_statistic[0],
            rtol=_INFERENCE_RTOL,
        )
        assert dfn == r_f_statistic[1]
        assert fit._df_t == r_f_statistic[2]
        np.testing.assert_allclose(
            fit.sequential_pvalue(
                R=restriction,
                mixture_precision=mixture_precision,
            ),
            np.asarray(r_results.rx2("f_pvalue")),
            rtol=_INFERENCE_RTOL,
        )


@pytest.mark.parametrize(
    "nobs, number_of_coefficients, alpha",
    [(100, 3, 0.05), (1_000, 5, 0.10), (10_000, 10, 0.01)],
)
def test_optimal_mixture_precision_matches_avlm(nobs, number_of_coefficients, alpha):
    pyfixest_result = optimal_mixture_precision(
        nobs=nobs,
        number_of_coefficients=number_of_coefficients,
        alpha=alpha,
    )
    avlm_result = avlm.optimal_g(nobs, number_of_coefficients, alpha)

    np.testing.assert_allclose(
        pyfixest_result,
        avlm_result,
        rtol=_OPTIMIZER_RTOL,
    )
