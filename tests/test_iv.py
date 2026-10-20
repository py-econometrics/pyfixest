import copy

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from pyfixest.errors import IVIdentificationError
from pyfixest.estimation import feols
from pyfixest.utils.check_r_install import check_r_install
from pyfixest.utils.utils import get_data, ssc

fixest = importr("fixest")
stats = importr("stats")

# Extend R packages
if import_check := check_r_install("ivDiag", strict=False):
    ivDiag = importr("ivDiag")


@pytest.fixture(scope="module")
def r_results():
    np.random.seed(1)

    # Number of observations
    n = 500

    # Simulate the data
    # Instrumental variable
    z = np.random.binomial(1, 0.5, size=n)
    z2 = np.random.binomial(1, 0.5, size=n)
    # Endogenous variable
    d = 0.5 * z + 1.5 * z2 + np.random.normal(size=n)

    # Control variables
    c1 = np.random.normal(size=n)
    c2 = np.random.normal(size=n)

    # Outcome variable
    y = 1.0 + 1.5 * d + 0.8 * c1 + 0.5 * c2 + np.random.normal(size=n)

    # Cluster variable
    cluster = np.random.randint(1, 50, size=n)

    # Sampling weights (random uniform distribution between 1 and 3 for example)
    weights = np.random.uniform(1, 3, size=n)

    # Create a DataFrame
    data = pd.DataFrame(
        {
            "d": d,
            "y": y,
            "z": z,
            "z2": z2,
            "c1": c1,
            "c2": c2,
            "cluster": cluster,
            "weights": weights,
        }
    )

    # Convert the DataFrame to an R DataFrame
    data_r = pandas2ri.py2rpy(data)

    # Define the variables
    Y = "y"
    D = "d"  # Endogenous treatment
    Z = "z"  # Instrumental variable
    controls = ["c1", "c2"]  # Covariates of control variables
    cl = "cluster"

    # Convert the variables and controls to R objects
    Y_r = ro.StrVector([Y])
    D_r = ro.StrVector([D])
    Z_r = ro.StrVector([Z])
    controls_r = ro.StrVector(controls)
    cl_r = ro.StrVector([cl])

    # set to True to run ivDiag R package
    run_r = False
    if run_r:
        # Call the ivDiag function from the ivDiag package
        F_stat_weights = ivDiag.ivDiag(
            Y=Y_r,
            D=D_r,
            Z=Z_r,
            controls=controls_r,
            data=data_r,
            weights="weights",
            cl=cl_r,
            run_AR=False,
            parallel=False,
            bootstrap=False,
        ).rx2("F_stat")

        F_stat_no_weights = ivDiag.ivDiag(
            Y=Y_r,
            D=D_r,
            Z=Z_r,
            controls=controls_r,
            data=data_r,
            cl=cl_r,
            run_AR=False,
            parallel=False,
            bootstrap=False,
        ).rx2("F_stat")

    else:
        F_stat_weights = np.array([20.1279, 18.9532, 17.3067, 17.3067])
        F_stat_no_weights = np.array([19.7981, 19.9658, 17.0545, 17.0545])

    return {
        "with_weights": F_stat_weights,
        "without_weights": F_stat_no_weights,
        "data": data,
    }


@pytest.mark.skipif(import_check is False, reason="R package ivDiag not installed.")
@pytest.mark.against_r_extended
@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.parametrize("adj_vcov", ["iid", "hetero", {"CRV1": "cluster"}])
def test_iv_Fstat_ivDiag(has_weight, adj_vcov, r_results):
    # Compare weak iv test result(naive, robust, and clustered F stats )
    # with ivDiag package.
    # Set random seed for reproducibility

    data = r_results["data"]
    if has_weight:
        weight_detail_py = "weights"
        result = r_results["with_weights"]
    else:
        weight_detail_py = None
        result = r_results["without_weights"]
    """
    if adj_vcov == 0.0:
        vcov_detail = {"CRV1": "cluster"}
    elif adj_vcov == 1.0:
        vcov_detail = "iid"
    elif adj_vcov == 2.0:
        vcov_detail = "hetero"
    """
    fit_iv = feols(
        "y ~ 1 + c1 + c2 | d ~ z", data=data, vcov=adj_vcov, weights=weight_detail_py
    )
    F_stat_pf = fit_iv.first_stage.diagnostics.f_stat
    fit_iv.IV_Diag()
    F_stat_eff_pf = fit_iv.first_stage.diagnostics.eff_f

    F_naive = result[0]
    F_hetero = result[1]
    F_cl = result[2]

    # Note that we are not putting arbitrary values into
    # F_eff_R. This is for saving computing times
    # Note that Effective F stat is equal to cluster robust F
    # when clusteres are set up. If not set up,
    # then effective F is equal to hetero-roboust F.
    _N = fit_iv.sample_info.n_obs
    if adj_vcov == {"CRV1": "cluster"}:
        F_stat_R = F_cl
        F_eff_R = result[3]
    elif adj_vcov == "iid":
        F_stat_R = F_naive
        F_eff_R = result[1]  # * _N / (_N - 1)
    elif adj_vcov == "hetero":
        F_stat_R = F_hetero  # * _N / (_N - 1)
        F_eff_R = result[1]  # * _N / (_N - 1)

    np.testing.assert_allclose(
        F_stat_pf,
        F_stat_R,
        rtol=1e-5,
        atol=1e-5,
        err_msg="First stage F stats estimate mismatch between pyfixest and IV_Diag packages",
    )
    np.testing.assert_allclose(
        F_stat_eff_pf,
        F_eff_R,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Effective F stats estimate mismatch between pyfixest and IV_Diag packages",
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "vcov, vcov_kwargs, vcov_r",
    [
        ("iid", None, "iid"),
        ("hetero", None, "hetero"),
        ({"CRV1": "f2"}, None, "~f2"),
        ({"CRV1": "f2+f3"}, None, "~f2+f3"),
        (
            "NW",
            {"lag": 2, "time_id": "year", "panel_id": "unit"},
            ("NW", {"unit": "unit", "time": "year", "lag": 2}),
        ),
        (
            "DK",
            {"lag": 2, "time_id": "year", "panel_id": "unit"},
            ("DK", {"time": "year", "lag": 2}),
        ),
    ],
    ids=["iid", "hetero", "CRV1", "CRV1-twoway", "NW", "DK"],
)
@pytest.mark.parametrize("k_adj", [True, False])
@pytest.mark.parametrize(
    "fml, fml_r, drop_intercept",
    [
        ("Y ~ X2 + [X1 ~ Z1 + Z2] | f1", "Y ~ X2 | f1 | X1 ~ Z1 + Z2", False),
        # fixest has no `drop_intercept`; `-1` removes the intercept from both
        # stages.
        ("Y ~ X2 + [X1 ~ Z1 + Z2]", "Y ~ -1 + X2 | X1 ~ Z1 + Z2", True),
        ("Y ~ -1 + X2 | X1 ~ Z1 + Z2", "Y ~ -1 + X2 | X1 ~ Z1 + Z2", False),
        ("Y ~ 0 + X2 | X1 ~ Z1 + Z2", "Y ~ 0 + X2 | X1 ~ Z1 + Z2", False),
        ("Y ~ X2 - 1 | X1 ~ Z1 + Z2", "Y ~ X2 - 1 | X1 ~ Z1 + Z2", False),
        ("Y ~ 0 | X1 ~ Z1 + Z2", "Y ~ 0 | X1 ~ Z1 + Z2", False),
        ("Y ~ -1 + X2 + [X1 ~ Z1 + Z2]", "Y ~ -1 + X2 | X1 ~ Z1 + Z2", False),
        ("Y ~ 0 + X2 + [X1 ~ Z1 + Z2]", "Y ~ 0 + X2 | X1 ~ Z1 + Z2", False),
        ("Y ~ X2 - 1 + [X1 ~ Z1 + Z2]", "Y ~ X2 - 1 | X1 ~ Z1 + Z2", False),
        ("Y ~ 0 + [X1 ~ Z1 + Z2]", "Y ~ 0 | X1 ~ Z1 + Z2", False),
        ("Y ~ X2 | X1 ~ Z1 + Z2", "Y ~ X2 | X1 ~ Z1 + Z2", False),
        ("Y ~ X2 + [X1 ~ Z1 + Z2]", "Y ~ X2 | X1 ~ Z1 + Z2", False),
        ("Y ~ 0 + X2 | X1 ~ 0 + Z1 + Z2", "Y ~ 0 + X2 | X1 ~ Z1 + Z2", False),
        ("Y ~ 0 + X2 + [X1 ~ 0 + Z1 + Z2]", "Y ~ 0 + X2 | X1 ~ Z1 + Z2", False),
        ("Y ~ 0 + X2 | f1 | X1 ~ Z1 + Z2", "Y ~ 0 + X2 | f1 | X1 ~ Z1 + Z2", False),
        (
            "Y ~ 0 + X2 + [X1 ~ Z1 + Z2] | f1",
            "Y ~ 0 + X2 | f1 | X1 ~ Z1 + Z2",
            False,
        ),
        (
            "Y ~ 0 + i(category) | X1 ~ Z1 + Z2",
            "Y ~ 0 + i(category) | X1 ~ Z1 + Z2",
            False,
        ),
        (
            "Y ~ 0 + i(category) + [X1 ~ Z1 + Z2]",
            "Y ~ 0 + i(category) | X1 ~ Z1 + Z2",
            False,
        ),
        # Explicit references align categorical instrument coding with fixest.
        (
            "Y ~ 0 + X2 | X1 ~ Z1 + i(category, ref='a')",
            "Y ~ 0 + X2 | X1 ~ Z1 + i(category, ref='a')",
            False,
        ),
        (
            "Y ~ 0 + X2 + [X1 ~ Z1 + i(category, ref='a')]",
            "Y ~ 0 + X2 | X1 ~ Z1 + i(category, ref='a')",
            False,
        ),
    ],
)
def test_first_stage_vs_fixest(
    vcov, vcov_kwargs, vcov_r, k_adj, fml, fml_r, drop_intercept
):
    # The first stage is fitted on the second stage's rows, with its
    # small-sample correction, intercept choice, and covariance estimator. The
    # data keep missing values and add a complete singleton row, which
    # `fixef_rm="none"` keeps in both stages.
    data = get_data()
    data["category"] = pd.Categorical(np.take(["a", "b", "c"], data.index % 3))
    complete_row = data[["Y", "X1", "X2", "Z1", "Z2", "f1"]].notna().all(axis=1)
    data.loc[complete_row.idxmax(), "f1"] = 999
    # a balanced panel of 50 units over 20 years for the HAC estimators
    data["unit"] = data.index // 20
    data["year"] = data.index % 20

    if isinstance(vcov_r, tuple):
        vcov_fun, vcov_args = vcov_r
        vcov_r = (fixest.vcov_NW if vcov_fun == "NW" else fixest.vcov_DK)(**vcov_args)
    elif vcov_r.startswith("~"):
        vcov_r = ro.Formula(vcov_r)

    fit = feols(
        fml,
        data=data,
        vcov=vcov,
        vcov_kwargs=vcov_kwargs,
        ssc=ssc(k_adj=k_adj),
        fixef_rm="none",
        drop_intercept=drop_intercept,
    )
    fit_r = fixest.feols(
        ro.Formula(fml_r),
        data=pandas2ri.py2rpy(data),
        vcov=vcov_r,
        ssc=fixest.ssc(k_adj, "nonnested", False, True, "min", "min"),
        fixef_rm="none",
        panel_time_step=1,
    )
    _assert_iv_stages_match_fixest(fit=fit, fit_r=fit_r)


@pytest.mark.against_r_core
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Pre-existing: default i() instrument coding differs from fixest without an intercept (#1917).",
)
@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ 0 + X2 | X1 ~ Z1 + i(category)",
        "Y ~ 0 + X2 + [X1 ~ Z1 + i(category)]",
    ],
    ids=["pipe", "bracket"],
)
def test_no_intercept_iv_default_categorical_instrument_vs_fixest(fml):
    """Without a reference, fixest omits a level that pyfixest retains."""
    data = get_data()
    data["category"] = pd.Categorical(np.take(["a", "b", "c"], data.index % 3))
    fit = feols(fml, data=data, vcov="iid")
    fit_r = fixest.feols(
        ro.Formula("Y ~ 0 + X2 | X1 ~ Z1 + i(category)"),
        data=pandas2ri.py2rpy(data),
        vcov="iid",
    )
    _assert_iv_stages_match_fixest(fit=fit, fit_r=fit_r)


@pytest.mark.against_r_core
@pytest.mark.parametrize("vcov", ["iid", "hetero"])
@pytest.mark.parametrize(
    "fml",
    ["Y ~ 0 + X2 | X1 ~ Z1 + Z2", "Y ~ 0 + X2 + [X1 ~ Z1 + Z2]"],
    ids=["pipe", "bracket"],
)
def test_weighted_no_intercept_iv_vs_fixest(fml, vcov):
    """Analytic weights preserve the absent intercept in both IV stages."""
    data = get_data()
    fit = feols(fml, data=data, weights="weights", vcov=vcov)
    fit_r = fixest.feols(
        ro.Formula("Y ~ 0 + X2 | X1 ~ Z1 + Z2"),
        data=pandas2ri.py2rpy(data),
        weights=ro.Formula("~weights"),
        vcov=vcov,
    )
    _assert_iv_stages_match_fixest(fit=fit, fit_r=fit_r)


def _assert_iv_stages_match_fixest(fit, fit_r):
    """Compare both IV stages and instrument inference against the R reference."""
    first_stage_r = ro.r("function(fit) summary(fit, stage = 1)")(fit_r)
    first_stage = fit.first_stage.model

    names_r = list(ro.r("function(fit) names(coef(fit))")(first_stage_r))
    has_intercept_r = "(Intercept)" in names_r
    assert ("Intercept" in fit.model_matrix.instruments.columns) == has_intercept_r
    assert ("Intercept" in first_stage.coef().index) == has_intercept_r
    second_stage_names_r = list(ro.r("function(fit) names(coef(fit))")(fit_r))
    second_stage_names = [
        name.replace("(Intercept)", "Intercept").removeprefix("fit_")
        for name in second_stage_names_r
    ]
    assert sorted(fit.coef().index) == sorted(second_stage_names)
    np.testing.assert_allclose(
        fit.coef()[second_stage_names],
        np.asarray(stats.coef(fit_r)),
        rtol=0,
        atol=1e-8,
        err_msg="second-stage coefficients != fixest",
    )

    n_obs_r = int(stats.nobs(fit_r)[0])
    assert fit.sample_info.n_obs == n_obs_r, "second-stage n_obs != fixest"
    assert first_stage.sample_info.n_obs == n_obs_r, "first-stage n_obs != fixest"

    names_r = [name.replace("(Intercept)", "Intercept") for name in names_r]
    assert sorted(first_stage.coef().index) == sorted(names_r), "coefnames differ"
    np.testing.assert_allclose(
        first_stage.coef()[names_r],
        np.asarray(stats.coef(first_stage_r)),
        rtol=0,
        atol=1e-8,
        err_msg="first-stage coefficients != fixest",
    )
    np.testing.assert_allclose(
        first_stage.se()[names_r],
        np.asarray(fixest.se(first_stage_r)),
        rtol=0,
        atol=1e-7,
        err_msg="first-stage standard errors != fixest",
    )
    # fixest's `ivwald` is the Wald test of the excluded instruments under the
    # first stage's covariance, the statistic pyfixest reports as `f_stat`. It
    # is O(100-1000) here, so it is compared on a relative scale.
    ivwald_r = ro.r("function(fit) fixest::fitstat(fit, 'ivwald')[[1]]$stat")(fit_r)
    np.testing.assert_allclose(
        fit.first_stage.diagnostics.f_stat,
        np.asarray(ivwald_r)[0],
        rtol=1e-8,
        err_msg="first-stage F statistic != fixest ivwald",
    )


@pytest.mark.against_r_core
def test_fixest_rejects_redundant_endogenous():
    """Unlike pyfixest's term deduplication, fixest rejects redundant declarations."""
    with pytest.raises(
        RRuntimeError, match="Endogenous variables should not be used as exogenous"
    ):
        fixest.feols(ro.Formula("Y ~ X2 | X2 ~ Z1"), data=pandas2ri.py2rpy(get_data()))


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "formula, reference_formula, weights",
    [
        ("Y ~ X1 + X2_hat | X2 ~ Z1", "Y ~ X1 + X2_hat | X2 ~ Z1", None),
        ("Y ~ X1 + X2_hat + X2 + [X2 ~ Z1]", "Y ~ X1 + X2_hat | X2 ~ Z1", None),
        ("Y ~ [X2 ~ Z1] + X1 + X2_hat + X2", "Y ~ X1 + X2_hat | X2 ~ Z1", None),
        ("Y ~ X1 + X2_hat + [X2:Z2 ~ Z1]", "Y ~ X1 + X2_hat | X2:Z2 ~ Z1", None),
        (
            "Y ~ X1 + X2_hat + X2 + [X2 ~ Z1] | f1",
            "Y ~ X1 + X2_hat | f1 | X2 ~ Z1",
            "weights",
        ),
        (
            "Y ~ X1 + X2_hat + [X2:Z2 ~ Z1] | f1",
            "Y ~ X1 + X2_hat | f1 | X2:Z2 ~ Z1",
            "weights",
        ),
    ],
    ids=[
        "pipe",
        "covariate-before-block",
        "covariate-after-block",
        "interacted-endogenous",
        "weighted-fe",
        "weighted-fe-interacted-endogenous",
    ],
)
def test_iv_generated_name_collision_against_fixest(
    formula, reference_formula, weights
):
    """User covariates survive placeholder names and redundant endogenous terms."""
    data = get_data().dropna().assign(X2_hat=lambda frame: frame.Z2)
    fit = feols(formula, data=data, weights=weights, vcov="iid")
    reference = fixest.feols(
        ro.Formula(reference_formula),
        data=pandas2ri.py2rpy(data),
        weights=ro.NULL if weights is None else ro.Formula(f"~{weights}"),
        vcov="iid",
    )
    first_stage_r = ro.r("function(fit) summary(fit, stage = 1)")(reference)
    for model, model_r in (
        (fit, reference),
        (fit.first_stage.model, first_stage_r),
    ):
        assert "X2_hat" in model.coef().index
        names_r = list(ro.r("function(fit) names(coef(fit))")(model_r))
        names = [
            column.removeprefix("fit_").replace("(Intercept)", "Intercept")
            for column in names_r
        ]
        assert sorted(model.coef().index) == sorted(names)
        np.testing.assert_allclose(
            model.coef()[names], stats.coef(model_r), rtol=0, atol=1e-8
        )
        np.testing.assert_allclose(
            model.se()[names], fixest.se(model_r), rtol=0, atol=1e-7
        )


@pytest.fixture
def iv_identification_data():
    return (
        get_data()
        .dropna()
        .assign(
            D=lambda frame: frame.X1, C=lambda frame: frame.X2, Z=lambda frame: frame.Z1
        )
    )


@pytest.fixture(
    params=[(None, ""), ("weights", ""), ("weights", "f1")],
    ids=["unweighted", "weighted", "weighted-fe"],
)
def iv_identification_options(request):
    return request.param


def _assert_iv_identification_error_matches_fixest(
    *, data, covariates, instrument, weights, fixed_effects, message, reference_message
):
    suffix = f" | {fixed_effects}" if fixed_effects else ""
    with pytest.raises(IVIdentificationError, match=message):
        feols(
            f"Y ~ {covariates} + [D ~ {instrument}]" + suffix,
            data=data,
            weights=weights,
        )

    reference_formula = (
        f"Y ~ {covariates} | "
        + (f"{fixed_effects} | " if fixed_effects else "")
        + f"D ~ {instrument}"
    )
    reference_error = ro.r("""function(fml, data, weights) {
        tryCatch({
            fixest::feols(fml, data=data, weights=weights)
            "NO ERROR"
        }, error=function(e) conditionMessage(e))
    }""")(
        ro.Formula(reference_formula),
        pandas2ri.py2rpy(data),
        ro.NULL if weights is None else ro.Formula(f"~{weights}"),
    )[0]
    assert reference_message in reference_error


@pytest.mark.against_r_core
def test_iv_identification_perfect_covariate_against_fixest(
    iv_identification_data, iv_identification_options
):
    data = iv_identification_data
    weights, fixed_effects = iv_identification_options
    data["C"] = data.D - data.D.mean()
    _assert_iv_identification_error_matches_fixest(
        data=data,
        covariates="C",
        instrument="Z",
        weights=weights,
        fixed_effects=fixed_effects,
        message="fully explained",
        reference_message="fully explained",
    )


@pytest.mark.against_r_core
def test_iv_identification_perfect_instrument_against_fixest(
    iv_identification_data, iv_identification_options
):
    data = iv_identification_data
    weights, fixed_effects = iv_identification_options
    data["Z"] = data.D
    _assert_iv_identification_error_matches_fixest(
        data=data,
        covariates="C",
        instrument="Z",
        weights=weights,
        fixed_effects=fixed_effects,
        message="fully explained",
        reference_message="fully explained",
    )


@pytest.mark.against_r_core
def test_iv_identification_perfect_instrument_without_covariates_against_fixest(
    iv_identification_data, iv_identification_options
):
    data = iv_identification_data
    weights, fixed_effects = iv_identification_options
    data["Z"] = data.D
    _assert_iv_identification_error_matches_fixest(
        data=data,
        covariates="0",
        instrument="Z",
        weights=weights,
        fixed_effects=fixed_effects,
        message="fully explained",
        reference_message="fully explained",
    )


@pytest.mark.against_r_core
def test_iv_identification_irrelevant_instrument_against_fixest(
    iv_identification_data, iv_identification_options
):
    data = iv_identification_data
    weights, fixed_effects = iv_identification_options
    suffix = f" | {fixed_effects}" if fixed_effects else ""
    # OLS residuals have no remaining projection onto C or Z on this sample.
    data["D"] = feols("D ~ C + Z" + suffix, data=data, weights=weights).resid()
    _assert_iv_identification_error_matches_fixest(
        data=data,
        covariates="C",
        instrument="Z",
        weights=weights,
        fixed_effects=fixed_effects,
        message="no explanatory power",
        reference_message="0 explanatory power",
    )


@pytest.mark.against_r_core
def test_iv_identification_irrelevant_instrument_without_covariates_against_fixest(
    iv_identification_data, iv_identification_options
):
    data = iv_identification_data
    weights, fixed_effects = iv_identification_options
    suffix = f" | {fixed_effects}" if fixed_effects else ""
    data["D"] = feols("D ~ C + Z" + suffix, data=data, weights=weights).resid()
    _assert_iv_identification_error_matches_fixest(
        data=data,
        covariates="0",
        instrument="Z",
        weights=weights,
        fixed_effects=fixed_effects,
        message="no explanatory power",
        reference_message="0 explanatory power",
    )


@pytest.mark.against_r_core
def test_iv_identification_small_instrument_against_fixest(
    iv_identification_data, iv_identification_options
):
    data = iv_identification_data
    weights, fixed_effects = iv_identification_options
    # lstsq alone retains this column, but the estimation rank check drops it.
    data["Z"] *= 1e-9
    _assert_iv_identification_error_matches_fixest(
        data=data,
        covariates="C",
        instrument="Z",
        weights=weights,
        fixed_effects=fixed_effects,
        message="no explanatory power",
        reference_message="0 explanatory power",
    )


@pytest.mark.against_r_core
def test_iv_identification_small_covariate_against_fixest(
    iv_identification_data, iv_identification_options
):
    data = iv_identification_data
    weights, fixed_effects = iv_identification_options
    suffix = f" | {fixed_effects}" if fixed_effects else ""
    data["D"] = feols("D ~ C + Z" + suffix, data=data, weights=weights).resid()
    data["C_small"] = 1e-9 * data.D
    _assert_iv_identification_error_matches_fixest(
        data=data,
        covariates="C + C_small",
        instrument="Z",
        weights=weights,
        fixed_effects=fixed_effects,
        message="no explanatory power",
        reference_message="0 explanatory power",
    )


@pytest.mark.against_r_core
def test_iv_identification_identical_instrument_covariate_against_fixest(
    iv_identification_data, iv_identification_options
):
    weights, fixed_effects = iv_identification_options
    _assert_iv_identification_error_matches_fixest(
        data=iv_identification_data,
        covariates="C",
        instrument="C",
        weights=weights,
        fixed_effects=fixed_effects,
        message="no explanatory power",
        reference_message="0 explanatory power",
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize("weights", [None, "weights"])
@pytest.mark.parametrize("control_scale", [1.0, 1e14], ids=["original", "rescaled"])
def test_iv_control_rescaling_against_fixest(
    iv_identification_data, weights, control_scale
):
    """Control units must not trigger a different rank cutoff in IV validation."""
    data = iv_identification_data
    data["C"] *= control_scale
    fit = feols("Y ~ C + [D ~ Z]", data=data, weights=weights, vcov="iid")
    reference = fixest.feols(
        ro.Formula("Y ~ C | D ~ Z"),
        data=pandas2ri.py2rpy(data),
        weights=ro.NULL if weights is None else ro.Formula("~weights"),
        vcov="iid",
    )
    names_r = list(ro.r("function(fit) names(coef(fit))")(reference))
    names = [
        name.removeprefix("fit_").replace("(Intercept)", "Intercept")
        for name in names_r
    ]
    assert sorted(fit.coef().index) == sorted(names)
    assert fit.sample_info.n_obs == int(stats.nobs(reference)[0])
    # Compare C in its original units so absolute tolerances do not mask errors
    # in its tiny coefficient or standard error after rescaling.
    original_units = np.array([control_scale if name == "C" else 1.0 for name in names])
    np.testing.assert_allclose(
        fit.coef()[names] * original_units,
        np.asarray(stats.coef(reference)) * original_units,
        rtol=0,
        atol=1e-8,
        err_msg="IV coefficients with a rescaled control != fixest",
    )
    np.testing.assert_allclose(
        fit.se()[names] * original_units,
        np.asarray(fixest.se(reference)) * original_units,
        rtol=0,
        atol=1e-7,
        err_msg="IV standard errors with a rescaled control != fixest",
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize("weights", [None, "weights"])
@pytest.mark.parametrize(
    "covariates, reference_covariates",
    [("C", "1"), ("C + X2", "X2")],
    ids=["all-controls-absorbed", "one-control-retained"],
)
def test_iv_identification_with_absorbed_covariate_against_fixest(
    weights, covariates, reference_covariates
):
    """Dropping an absorbed control keeps the second-stage projection aligned."""
    data = (
        get_data()
        .dropna()
        .assign(
            D=lambda frame: frame.X1, C=lambda frame: frame.f1, Z=lambda frame: frame.Z1
        )
    )
    with pytest.warns(UserWarning, match="dropped due to multicollinearity"):
        fit = feols(
            f"Y ~ {covariates} + [D ~ Z] | f1",
            data=data,
            weights=weights,
            vcov="iid",
        )
    # C is fully absorbed. fixest 0.14.0 fails internally when every weighted
    # covariate is absorbed, so compare the equivalent design with C omitted.
    reference = fixest.feols(
        ro.Formula(f"Y ~ {reference_covariates} | f1 | D ~ Z"),
        data=pandas2ri.py2rpy(data),
        weights=ro.NULL if weights is None else ro.Formula("~weights"),
        vcov="iid",
    )
    names_r = list(ro.r("function(fit) names(coef(fit))")(reference))
    names = [name.removeprefix("fit_") for name in names_r]
    assert sorted(fit.coef().index) == sorted(names)
    assert "C" in fit.collinearity.dropped_coef_names
    assert fit.sample_info.n_obs == int(stats.nobs(reference)[0])
    np.testing.assert_allclose(
        fit.coef()[names],
        stats.coef(reference),
        rtol=0,
        atol=1e-8,
        err_msg="IV coefficients with an absorbed covariate != fixest",
    )
    np.testing.assert_allclose(
        fit.se()[names],
        fixest.se(reference),
        rtol=0,
        atol=1e-7,
        err_msg="IV standard errors with an absorbed covariate != fixest",
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "formula, reference_formula",
    [
        ("Y ~ I(X1**2) + [X1 ~ Z1]", "Y ~ I(X1^2) | X1 ~ Z1"),
        ("Y ~ X1 + [I(X1**2) ~ Z1]", "Y ~ X1 | I(X1^2) ~ Z1"),
        ("Y ~ X1 + [X2 ~ Z1 + I(X1**2)]", "Y ~ X1 | X2 ~ Z1 + I(X1^2)"),
        ("Y ~ X1 + [X2 ~ Z1 + Z1:X1]", "Y ~ X1 | X2 ~ Z1 + Z1:X1"),
        ("Y ~ X1 + [X2 ~ Z1 + X1]", "Y ~ X1 | X2 ~ Z1 + X1"),
    ],
)
@pytest.mark.parametrize("weights, fixed_effects", [(None, ""), ("weights", "f1")])
def test_iv_transformed_terms_against_fixest(
    formula, reference_formula, weights, fixed_effects
):
    """Numerically valid transformations and redundant instruments are allowed."""
    data = get_data().dropna()
    if fixed_effects:
        formula += f" | {fixed_effects}"
        reference_formula = reference_formula.replace(
            " | ", f" | {fixed_effects} | ", 1
        )
    fit = feols(formula, data=data, weights=weights, vcov="iid")
    reference = fixest.feols(
        ro.Formula(reference_formula),
        data=pandas2ri.py2rpy(data),
        weights=ro.NULL if weights is None else ro.Formula("~weights"),
        vcov="iid",
    )
    names_r = list(ro.r("function(fit) names(coef(fit))")(reference))
    names = [
        name.removeprefix("fit_")
        .replace("(Intercept)", "Intercept")
        .replace("I(I(X1^2))", "I(X1 ** 2)")
        for name in names_r
    ]
    assert sorted(fit.coef().index) == sorted(names)
    assert fit.sample_info.n_obs == int(stats.nobs(reference)[0])
    np.testing.assert_allclose(
        fit.coef()[names],
        stats.coef(reference),
        rtol=0,
        atol=1e-8,
        err_msg="IV transformed-term coefficients != fixest",
    )
    np.testing.assert_allclose(
        fit.se()[names],
        fixest.se(reference),
        rtol=0,
        atol=1e-7,
        err_msg="IV transformed-term standard errors != fixest",
    )


@pytest.mark.parametrize("seed", [293, 912])
@pytest.mark.parametrize("sd", [0.1, 0.2, 0.3])
@pytest.mark.parametrize("has_weight", [0.0, 1.0])
@pytest.mark.parametrize("adj_vcov", [0.0, 1.0, 2.0, 3.0, 4.0])
def test_1st_stage_iv(seed, sd, has_weight, adj_vcov):
    # Test 1st stage regression result in 2SLS estimator.
    rng = np.random.default_rng(seed)
    data = get_data().dropna()
    data["Z1"] = data["Z1"] + rng.normal(0, sd, size=len(data))

    # Compute test statistics of IV and OLS respectively

    weight_detail = "weights" if has_weight == 1.0 else None

    if adj_vcov == 0.0:
        vcov_detail = {"CRV1": "f1"}
    elif adj_vcov == 1.0:
        vcov_detail = "iid"
    elif adj_vcov == 2.0:
        vcov_detail = "hetero"
    elif adj_vcov == 3.0:
        vcov_detail = "HC1"
    elif adj_vcov == 4.0:
        vcov_detail = None

    fit_iv = feols(
        "Y ~ 1 | f1 | X1 ~ Z1 ", vcov=vcov_detail, data=data, weights=weight_detail
    )
    fit_ols = feols("X1 ~  Z1 | f1", vcov=vcov_detail, data=data, weights=weight_detail)

    wald_ols = fit_ols.wald_test()

    first_stage = fit_iv.first_stage
    _pi_hat_iv = first_stage.coefficients
    _X_hat_iv = first_stage.fitted_values
    _v_hat_iv = first_stage.residuals
    _F_stat_iv = first_stage.diagnostics.f_stat
    _F_pval_iv = first_stage.diagnostics.p_value

    _pi_hat_ols = fit_ols._beta_hat
    _X_hat_ols = fit_ols.within_data.design @ fit_ols._beta_hat
    _v_hat_ols = fit_ols._u_hat
    _F_stat_ols = wald_ols.f_statistic
    _F_pval_ols = wald_ols.pvalue

    # Assert that the parameter estimates and predicted values are c
    # lose between IV and OLS
    np.testing.assert_allclose(
        _pi_hat_iv,
        _pi_hat_ols,
        rtol=1e-5,
        atol=1e-8,
        err_msg="First stage coefficient estimate mismatch between IV and OLS",
    )

    np.testing.assert_allclose(
        _X_hat_iv,
        _X_hat_ols,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Predicted X values mismatch in first stage between IV and OLS",
    )

    np.testing.assert_allclose(
        _v_hat_iv,
        _v_hat_ols,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Residuals mismatch in first stage between IV and OLS",
    )

    np.testing.assert_allclose(
        _F_stat_iv,
        _F_stat_ols,
        rtol=1e-5,
        atol=1e-8,
        err_msg="F-Stats mismatch in first stage between IV and OLS",
    )

    np.testing.assert_allclose(
        _F_pval_iv,
        _F_pval_ols,
        rtol=1e-5,
        atol=1e-8,
        err_msg="F-Stats p-value mismatch in first stage between IV and OLS",
    )


@pytest.mark.parametrize("weights_type", ["aweights", "fweights"])
@pytest.mark.parametrize("k_adj", [True, False])
def test_iv_diag_does_not_relabel_vcov_type(weights_type, k_adj):
    # The effective F stat uses heteroskedasticity-robust first-stage inference,
    # but neither the outer model nor the retained first stage should change.
    data = get_data()
    if weights_type == "fweights":
        data["weights"] = np.maximum(1, np.rint(data["weights"])).astype(int)

    fit_iid = feols(
        "Y ~ X2 + [X1 ~ Z1 + Z2]",
        data=data,
        weights="weights",
        weights_type=weights_type,
        vcov="iid",
        ssc=ssc(k_adj=k_adj),
    )
    vcov_type_detail_before = fit_iid.variance_covariance.spec.vcov_type_detail
    se_before = fit_iid.se().copy()
    first_stage_model = fit_iid.first_stage.model
    first_stage_vcov_before = first_stage_model.variance_covariance
    first_stage_se_before = first_stage_model.se().copy()
    first_stage_f_before = fit_iid.first_stage.diagnostics.f_stat
    first_stage_p_value_before = fit_iid.first_stage.diagnostics.p_value

    reference_first_stage = copy.deepcopy(first_stage_model)
    reference_first_stage.vcov("hetero")

    fit_iid.IV_Diag()

    assert vcov_type_detail_before == "iid"
    assert fit_iid.variance_covariance.spec.vcov_type_detail == "iid", (
        "IV_Diag() relabelled the main model's covariance type"
    )
    np.testing.assert_allclose(
        fit_iid.se(),
        se_before,
        rtol=1e-12,
        atol=1e-12,
        err_msg="IV_Diag() changed the main model's standard errors",
    )
    assert fit_iid.first_stage.model.variance_covariance is first_stage_vcov_before
    np.testing.assert_allclose(first_stage_model.se(), first_stage_se_before)
    assert fit_iid.first_stage.diagnostics.f_stat == first_stage_f_before
    assert fit_iid.first_stage.diagnostics.p_value == first_stage_p_value_before
    assert np.isfinite(fit_iid.first_stage.diagnostics.eff_f)

    instruments = list(fit_iid.first_stage.instruments)
    iv_positions = [
        list(first_stage_model._coefnames).index(instrument)
        for instrument in instruments
    ]
    Z = reference_first_stage.within_data.design[:, iv_positions]
    observation_weights = reference_first_stage.observation_weights.values
    Q_zz = (
        Z.T @ Z
        if observation_weights is None
        else Z.T @ (observation_weights[:, None] * Z)
    )
    pi_hat = np.array(reference_first_stage.coef()[instruments])
    Sigma = reference_first_stage.variance_covariance.vcov[
        np.ix_(iv_positions, iv_positions)
    ]
    expected_eff_f = (pi_hat.T @ Q_zz @ pi_hat) / np.sum(np.diag(Sigma @ Q_zz))
    np.testing.assert_allclose(fit_iid.first_stage.diagnostics.eff_f, expected_eff_f)

    # The effective F is computed from the heteroskedasticity-robust first
    # stage, so it does not depend on the outer model's covariance type.
    fit_hetero = feols(
        "Y ~ X2 + [X1 ~ Z1 + Z2]",
        data=data,
        weights="weights",
        weights_type=weights_type,
        vcov="hetero",
        ssc=ssc(k_adj=k_adj),
    )
    fit_hetero.IV_Diag()

    np.testing.assert_allclose(
        fit_iid.first_stage.diagnostics.eff_f,
        fit_hetero.first_stage.diagnostics.eff_f,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Effective F differs between iid and hetero specifications",
    )


def test_eff_f_does_not_require_stored_data():
    # Since eff_F() no longer refits the first stage to switch its covariance,
    # it does not need the raw data retained for an iid-vcov fit.
    data = get_data()
    fit_iid = feols("Y ~ X2 + [X1 ~ Z1 + Z2]", data=data, vcov="iid", store_data=False)
    fit_iid.IV_Diag()
    assert np.isfinite(fit_iid.first_stage.diagnostics.eff_f)
