import numpy as np
import pandas as pd
import pytest

from pyfixest.estimation import feols, fepois
from pyfixest.exceptions import (
    DuplicateKeyError,
    EndogVarsAsCovarsError,
    InstrumentsAsCovarsError,
    InvalidReferenceLevelError,
    MultiEstNotSupportedError,
    NanInClusterVarError,
    UnderDeterminedIVError,
    VcovTypeNotSupportedError,
)
from pyfixest.FormulaParser import FixestFormulaParser
from pyfixest.multcomp import rwolf
from pyfixest.summarize import etable, summary
from pyfixest.utils import get_data


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


def test_i_ref():
    data = get_data()

    with pytest.raises(ValueError):
        feols(fml="y ~ i(X1, X2, ref = -1)", data=data, vcov="iid")


def test_cluster_na():
    """Test if a nan value in a cluster variable raises an error."""
    data = get_data()
    data = data.dropna()
    data["f3"] = data["f3"].astype("int64")
    data["f3"][5] = np.nan

    with pytest.raises(NanInClusterVarError):
        feols(fml="Y ~ X1", data=data, vcov={"CRV1": "f3"})


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
        feols(fml="Y ~ 1 | Z1 ~ X1 ", data=data).wildboottest(param="Z1", B=999)
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


def test_i_interaction_errors():
    data = get_data()

    with pytest.raises(InvalidReferenceLevelError):
        # "a" not a level in f1
        feols(fml="Y ~ i(f1, X1)", data=data, i_ref1="a")

    with pytest.raises(
        InvalidReferenceLevelError
    ):  # incorrect type - int is provided but float required
        # "a" not a level in f1
        feols(fml="Y ~ i(f1, X1)", data=data, i_ref1=1)


def test_all_variables_multicollinear():
    data = get_data()
    with pytest.raises(ValueError):
        fit = feols("Y ~ f1 | f1", data=data)  # noqa: F841


def test_wls_errors():
    data = get_data()

    with pytest.raises(AssertionError):
        feols(fml="Y ~ X1", data=data, weights="weights2")

    with pytest.raises(AssertionError):
        feols("Y ~ X1", data=data, weights=[1, 2])

    data.loc[0, "weights"] = np.nan
    with pytest.raises(VcovTypeNotSupportedError):
        feols("Y ~ X1", data=data, weights="weights", vcov={"CRV3": "group_id"})

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
        feols("Y ~ X1", data=data, weights="weights", vcov="iid").wildboottest(B=999)


def test_multcomp_errors():
    data = get_data().dropna()

    # param not in model
    fit1 = feols("Y + Y2 ~ X1 | f1", data=data)
    with pytest.raises(ValueError):
        rwolf(fit1.to_list(), param="X2", B=999, seed=92)


def test_wildboottest_errors():
    data = get_data()
    fit = feols("Y ~ X1", data=data)
    with pytest.raises(ValueError):
        fit.wildboottest(param="X2", B=999, seed=213)


def test_summary_errors():
    data = get_data()
    fit1 = feols("Y + Y2 ~ X1 | f1", data=data)
    fit2 = feols("Y ~ X1 + X2 | f1", data=data)

    with pytest.raises(TypeError):
        etable(fit1)
    with pytest.raises(TypeError):
        etable([fit1, fit2])
    with pytest.raises(TypeError):
        summary(fit1)
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

    with pytest.raises(ValueError):
        etable([fit1, fit2], coef_fmt="b (se)\nt [p]", type="tex")

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
