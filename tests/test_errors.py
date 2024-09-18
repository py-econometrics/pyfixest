import numpy as np
import pandas as pd
import pytest
from formulaic.errors import FactorEvaluationError

import pyfixest as pf
from pyfixest.errors import (
    DuplicateKeyError,
    EndogVarsAsCovarsError,
    FeatureDeprecationError,
    InstrumentsAsCovarsError,
    MultiEstNotSupportedError,
    NanInClusterVarError,
    UnderDeterminedIVError,
    VcovTypeNotSupportedError,
)
from pyfixest.estimation.estimation import feols, fepois
from pyfixest.estimation.FormulaParser import FixestFormulaParser
from pyfixest.estimation.multcomp import rwolf
from pyfixest.report.summarize import etable, summary
from pyfixest.utils.utils import get_data, ssc


@pytest.fixture
def data():
    return pf.get_data()


def test_formula_parser2():
    with pytest.raises(DuplicateKeyError):
        FixestFormulaParser("y ~ sw(a, b) +  sw(c, d)| sw(X3, X4))")


def test_formula_parser3():
    with pytest.raises(DuplicateKeyError):
        FixestFormulaParser("y ~ sw(a, b) +  csw(c, d)| sw(X3, X4))")


# def test_formula_parser2():
#    with pytest.raises(FixedEffectInteractionError):
#        FixestFormulaParser('y ~ X1 + X2 | X3:X4')

# def test_formula_parser3():
#    with pytest.raises(CovariateInteractionError):
#        FixestFormulaParser('y ~ X1 + X2^X3')


def test_cluster_na():
    """Test if a nan value in a cluster variable raises an error."""
    data = get_data()
    data = data.dropna()
    data["f3"] = data["f3"].astype("int64")
    data["f3"][5] = np.nan

    with pytest.raises(NanInClusterVarError):
        feols(fml="Y ~ X1", data=data, vcov={"CRV1": "f3"})


def test_cluster_but_no_data():
    """Test if AttributeError if self._data is not stored."""
    data = get_data()
    fit = feols("Y ~ X1", data=data, store_data=False)
    with pytest.raises(AttributeError):
        fit.vcov({"CRV1": "f2"})


def test_error_hc23_fe():
    """
    Test if HC2 & HC3 inference with fixed effects regressions raises an error.

    Notes
    -----
    Currently not supported.
    """
    data = get_data().dropna()

    with pytest.raises(VcovTypeNotSupportedError):
        feols(fml="Y ~ X1 | f2", data=data, vcov="HC2")

    with pytest.raises(VcovTypeNotSupportedError):
        feols(fml="Y ~ X1 | f2", data=data, vcov="HC3")


def test_depvar_numeric():
    """Test if feols() throws an error when the dependent variable is not numeric."""
    data = get_data()
    data["Y"] = data["Y"].astype("str")
    data["Y"] = pd.Categorical(data["Y"])

    with pytest.raises(TypeError):
        feols(fml="Y ~ X1", data=data)


def test_iv_errors():
    data = get_data()

    # under determined
    with pytest.raises(UnderDeterminedIVError):
        feols(fml="Y ~ X1 | Z1 + Z2 ~ 24 ", data=data)
    # instrument specified as covariate
    with pytest.raises(InstrumentsAsCovarsError):
        feols(fml="Y ~ X1 | Z1  ~ X1 + X2", data=data)
    # endogenous variable specified as covariate
    with pytest.raises(EndogVarsAsCovarsError):
        feols(fml="Y ~ Z1 | Z1  ~ X1", data=data)

    # instrument specified as covariate
    # with pytest.raises(InstrumentsAsCovarsError):
    #    fixest.feols('Y ~ X1 | Z1 + Z2 ~ X3 + X4')
    # underdetermined IV
    # with pytest.raises(UnderDeterminedIVError):
    #    fixest.feols('Y ~ X1 + X2 | X1 + X2 ~ X4 ')
    # with pytest.raises(UnderDeterminedIVError):
    #    fixest.feols('Y ~ X1 | Z1 + Z2 ~ X2 + X3 ')
    # CRV3 inference
    with pytest.raises(VcovTypeNotSupportedError):
        feols(fml="Y ~ 1 | Z1 ~ X1 ", vcov={"CRV3": "group_id"}, data=data)
    # wild bootstrap
    with pytest.raises(NotImplementedError):
        feols(fml="Y ~ 1 | Z1 ~ X1 ", data=data).wildboottest(param="Z1", reps=999)
    # multi estimation error
    with pytest.raises(MultiEstNotSupportedError):
        feols(fml="Y + Y2 ~ 1 | Z1 ~ X1 ", data=data)
    with pytest.raises(MultiEstNotSupportedError):
        feols(fml="Y  ~ 1 | sw(f2, f3) | Z1 ~ X1 ", data=data)
    with pytest.raises(MultiEstNotSupportedError):
        feols(fml="Y  ~ 1 | csw(f2, f3) | Z1 ~ X1 ", data=data)
    # unsupported HC vcov
    with pytest.raises(VcovTypeNotSupportedError):
        feols(fml="Y  ~ 1 | Z1 ~ X1", vcov="HC2", data=data)
    with pytest.raises(VcovTypeNotSupportedError):
        feols(fml="Y  ~ 1 | Z1 ~ X1", vcov="HC3", data=data)


@pytest.mark.skip("Not yet implemented.")
def test_poisson_devpar_count():
    """Check that the dependent variable is a count variable."""
    data = get_data()
    # under determined
    with pytest.raises(AssertionError):
        fepois(fml="Y ~ X1 | X4", data=data)


def test_all_variables_multicollinear():
    data = get_data()
    with pytest.raises(ValueError):
        fit = feols("Y ~ f1 | f1", data=data)  # noqa: F841


def test_wls_errors():
    data = get_data()

    with pytest.raises(AssertionError):
        feols(fml="Y ~ X1", data=data, weights="weights2")

    with pytest.raises(ValueError):
        feols("Y ~ X1", data=data, weights=[1, 2])

    data.loc[0, "weights"] = np.nan
    with pytest.raises(NotImplementedError):
        feols("Y ~ X1", data=data, weights="weights").wildboottest(
            cluster="f1", param="X1", reps=999, seed=12
        )

    # test for ValueError when weights are not positive
    data.loc[10, "weights"] = -1
    with pytest.raises(ValueError):
        feols("Y ~ X1", data=data, weights="weights", vcov="iid")

    # test for ValueError when weights are not numeric
    data["weights"] = data["weights"].astype("str")
    data.loc[10, "weights"] = "a"
    with pytest.raises(ValueError):
        feols("Y ~ X1", data=data, weights="weights", vcov="iid")

    data = get_data()
    with pytest.raises(NotImplementedError):
        feols("Y ~ X1", data=data, weights="weights", vcov="iid").wildboottest(reps=999)


def test_multcomp_errors():
    data = get_data().dropna()

    # param not in model
    fit1 = feols("Y + Y2 ~ X1 | f1", data=data)
    with pytest.raises(ValueError):
        rwolf(fit1.to_list(), param="X2", reps=999, seed=92)


def test_multcomp_sampling_errors():
    data = get_data().dropna()
    # Sampling method not supported in "rwolf"
    fit1 = feols("Y + Y2 ~ X1 | f1", data=data)
    with pytest.raises(ValueError):
        rwolf(fit1.to_list(), param="X1", reps=999, seed=92, sampling_method="abc")


def test_rwolf_error():
    rng = np.random.default_rng(123)

    data = get_data()
    data["f1"] = rng.choice(range(5), len(data), True)
    fit = feols("Y + Y2 ~ X1 | f1", data=data)

    # test for full enumeration warning
    with pytest.warns(UserWarning):
        pf.rwolf(fit.to_list(), "X1", reps=9999, seed=123)


def test_wildboottest_errors():
    data = get_data()
    fit = feols("Y ~ X1", data=data)
    with pytest.raises(ValueError):
        fit.wildboottest(param="X2", reps=999, seed=213)


def test_summary_errors():
    "Test for appropriate errors when providing objects of type FixestMulti."
    data = get_data()
    fit1 = feols("Y + Y2 ~ X1 | f1", data=data)
    fit2 = feols("Y ~ X1 + X2 | f1", data=data)

    with pytest.raises(TypeError):
        etable([fit1, fit2])
    with pytest.raises(TypeError):
        summary([fit1, fit2])


def test_errors_etable():
    data = get_data()
    fit1 = feols("Y ~ X1", data=data)
    fit2 = feols("Y ~ X1 + X2 | f1", data=data)

    with pytest.raises(AssertionError):
        etable([fit1, fit2], signif_code=[0.01, 0.05])

    with pytest.raises(AssertionError):
        etable([fit1, fit2], signif_code=[0.2, 0.05, 0.1])

    with pytest.raises(AssertionError):
        etable([fit1, fit2], signif_code=[0.1, 0.5, 1.5])

    with pytest.raises(AssertionError):
        etable(
            models=[fit1, fit2],
            custom_stats={
                "conf_int_lb": [
                    fit2._conf_int[0]
                ],  # length of customized statistics not equal to the number of models
                "conf_int_ub": [fit2._conf_int[1]],
            },
            coef_fmt="b se\n[conf_int_lb, conf_int_ub]",
        )

    with pytest.raises(AssertionError):
        etable(
            models=[fit1, fit2],
            custom_stats={
                "conf_int_lb": [
                    [0.1, 0.1, 0.1],
                    fit2._conf_int[0],
                ],  # length of customized statistics not equal to length of model
                "conf_int_ub": [fit1._conf_int[1], fit2._conf_int[1]],
            },
            coef_fmt="b [conf_int_lb, conf_int_ub]",
        )

    with pytest.raises(ValueError):
        etable(
            models=[fit1, fit2],
            custom_stats={
                "b": [
                    fit2._conf_int[0],
                    fit2._conf_int[0],
                ],  # preserved keyword cannot be used as a custom statistic
            },
            coef_fmt="b [se]",
        )


def test_errors_ccv():
    data = get_data().dropna()
    data["D"] = np.random.choice([0, 1], size=len(data))

    # error when D is not binary
    fit = feols("Y ~ X1", data=data, vcov={"CRV1": "f1"})
    with pytest.raises(AssertionError):
        fit.ccv(treatment="X1", pk=0.05, qk=0.5, n_splits=10, seed=929)

    # error when fixed effects in estimation
    fit = feols("Y ~ D | f1", data=data)
    with pytest.raises(NotImplementedError):
        fit.ccv(treatment="D", pk=0.05, qk=0.5, n_splits=10, seed=929)

    # error when treatment not found
    fit = feols("Y ~ D", data=data)
    with pytest.raises(ValueError):
        fit.ccv(treatment="X2", pk=0.05, qk=0.5, n_splits=10, seed=929)

    # error when no cluster variable found
    fit = feols("Y ~ D", data=data)
    with pytest.raises(ValueError):
        fit.ccv(treatment="D", pk=0.05, qk=0.5, n_splits=10, seed=929)

    # error when cluster variable not in data.frame
    with pytest.raises(ValueError):
        fit.ccv(treatment="D", pk=0.05, qk=0.5, n_splits=10, seed=929, cluster="e8")

    # error when two-way clustering
    fit = feols("Y ~ D", data=data, vcov={"CRV1": "f1+f2"})
    with pytest.raises(ValueError):
        fit.ccv(treatment="D", pk=0.05, qk=0.5, n_splits=10, seed=929)

    # error when ccv is attempted on fepois
    pois_data = get_data(model="Fepois").dropna()
    pois_data["D"] = np.random.choice([0, 1], size=len(pois_data))
    fit = fepois("Y ~ D", data=pois_data)
    with pytest.raises(AssertionError):
        fit.ccv(treatment="D", pk=0.05, qk=0.5, n_splits=10, seed=929)

    # same for IV
    fit = feols("Y ~ 1 | D ~ Z1", data=data)
    with pytest.raises(AssertionError):
        fit.ccv(treatment="D", pk=0.05, qk=0.5, n_splits=10, seed=929)


def test_errors_confint():
    data = get_data()
    fit = feols("Y ~ X1", data=data)
    with pytest.raises(ValueError):
        fit.confint(alpha=0.5, keep=["abababa"])


def test_deprecation_errors():
    data = get_data()
    with pytest.raises(FeatureDeprecationError):
        feols("Y ~ i(f1)", data, i_ref1=1)
    with pytest.raises(FeatureDeprecationError):
        fepois("Y ~ i(f1)", data, i_ref1=1)


def test_i_error():
    data = get_data()
    data["f2"] = pd.Categorical(data["f2"])

    with pytest.raises(ValueError):
        feols("Y ~ i(f1, f2)", data)

    data["f2"] = data["f2"].astype("object")
    with pytest.raises(ValueError):
        feols("Y ~ i(f1, f2)", data)

    with pytest.raises(FactorEvaluationError):
        feols("Y ~ i(f1, X1, ref=a)", data)


def test_plot_error():
    df = get_data()
    fit = feols("Y ~ X1", data=df)
    with pytest.raises(
        ValueError, match="plot_backend must be either 'lets_plot' or 'matplotlib'."
    ):
        fit.coefplot(plot_backend="plotnine")

    fit_multi = feols("Y + Y2 ~ i(f1)", data=df)
    with pytest.raises(ValueError):
        fit_multi.coefplot(joint=True)

    with pytest.raises(ValueError):
        fit_multi.iplot(joint="both")


def test_ritest_error(data):
    fit = pf.feols("Y ~ X1", data=data)

    with pytest.raises(ValueError, match="X2 not found in the model's coefficients."):
        fit.ritest(resampvar="X2", reps=1000)

    with pytest.raises(ValueError, match="CLUST is not found in the data"):
        fit.ritest(resampvar="X1", reps=1000, cluster="CLUST")

    with pytest.raises(
        ValueError, match="type must be 'randomization-t' or 'randomization-c."
    ):
        fit.ritest(resampvar="X1", reps=1000, type="a")

    with pytest.raises(AssertionError):
        fit.ritest(resampvar="X1", reps=100.4)

    with pytest.raises(ValueError):
        fit.ritest(resampvar="X1", cluster="f1", reps=100)

    with pytest.raises(NotImplementedError):
        fit_iv = pf.feols("Y ~ 1 | X1 ~ Z1", data=data)
        fit_iv.ritest(resampvar="X1", reps=100)

    with pytest.raises(NotImplementedError):
        fit_wls = pf.feols("Y ~ X1", data=data, weights="weights")
        fit_wls.ritest(resampvar="X1", reps=100)

    with pytest.raises(ValueError):
        "Incorrect plot backend."
        fit.ritest(resampvar="X1", reps=100, store_ritest_statistics=True)
        fit.plot_ritest(plot_backend="a")

    with pytest.raises(ValueError):
        "No test_statistics found in the model."
        fit = pf.feols("Y ~ X1", data=data)
        fit.ritest(resampvar="X1", reps=100)
        fit.plot_ritest()


def test_wald_test_invalid_distribution():
    data = pd.read_csv("pyfixest/did/data/df_het.csv")
    data = data.iloc[1:3000]

    fml = "dep_var ~ treat"
    fit = feols(fml, data, vcov={"CRV1": "year"}, ssc=ssc(adj=False))

    with pytest.raises(ValueError):
        fit.wald_test(R=np.array([[1, -1]]), distribution="abc")


def test_wald_test_R_q_column_consistency():
    data = pd.read_csv("pyfixest/did/data/df_het.csv")
    data = data.iloc[1:3000]
    fml = "dep_var ~ treat"
    fit = feols(fml, data, vcov={"CRV1": "year"}, ssc=ssc(adj=False))

    # Test with R.size[1] == number of coeffcients
    with pytest.raises(ValueError):
        fit.wald_test(R=np.array([[1, 0, 0]]))

    # Test with q type
    with pytest.raises(ValueError):
        fit.wald_test(R=np.array([[1, 0]]), q="invalid type q")

    # Test with q being a one-dimensional array or a scalar.
    with pytest.raises(ValueError):
        fit.wald_test(R=np.array([[1, 0], [0, 1]]), q=np.array([[0, 1]]))

    # q must have the same number of rows as R
    with pytest.raises(ValueError):
        fit.wald_test(
            R=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), q=np.array([[0, 1]])
        )


def setup_feiv_instance():
    # Setup necessary data for Feiv

    data = pf.get_data()
    return pf.feols("Y ~ 1 | X1 ~ Z1", data=data)


def test_IV_first_stage_invalid_model_type():
    class NotFeols:
        # Dummy class for testing invalid model type
        pass

    invalid_model = NotFeols()

    with pytest.raises(TypeError):
        feiv_instance = setup_feiv_instance()
        feiv_instance.first_stage(invalid_model)  # This should raise TypeError


def test_IV_Diag_unsupported_statistics():
    feiv_instance = setup_feiv_instance()

    unsupported_statistics = ["unsupported_stat"]

    with pytest.raises(ValueError):
        feiv_instance.IV_Diag(statistics=unsupported_statistics)
