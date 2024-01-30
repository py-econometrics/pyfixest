import pytest
import numpy as np
import pandas as pd
from pyfixest.utils import get_data
from pyfixest.exceptions import (
    DuplicateKeyError,
    EndogVarsAsCovarsError,
    InstrumentsAsCovarsError,
    UnderDeterminedIVError,
    VcovTypeNotSupportedError,
    MultiEstNotSupportedError,
    NanInClusterVarError,
    InvalidReferenceLevelError,
)
from pyfixest.estimation import feols, fepois
from pyfixest.FormulaParser import FixestFormulaParser


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
    """
    test if a nan value in a cluster variable raises
    an error
    """

    data = get_data()
    data = data.dropna()
    data["f3"] = data["f3"].astype("int64")
    data["f3"][5] = np.nan

    with pytest.raises(NanInClusterVarError):
        feols(fml="Y ~ X1", data=data, vcov={"CRV1": "f3"})


def test_error_hc23_fe():
    """
    test if HC2&HC3 inference with fixed effects regressions raises an error (currently not supported)
    """
    data = get_data().dropna()

    with pytest.raises(VcovTypeNotSupportedError):
        feols(fml="Y ~ X1 | f2", data=data, vcov="HC2")

    with pytest.raises(VcovTypeNotSupportedError):
        feols(fml="Y ~ X1 | f2", data=data, vcov="HC3")


def test_depvar_numeric():
    """
    test if feols() throws an error when the dependent variable is not numeric
    """

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
    # endogeneous variable specified as covariate
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
    """
    check that the dependent variable is a count variable
    """

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
        fit = feols("Y ~ f1 | f1", data=data)


def test_wls_errors():
    data = get_data()

    with pytest.raises(AssertionError):
        feols(fml="Y ~ X1", data=data, weights="weights2")

    with pytest.raises(AssertionError):
        feols("Y ~ X1", data=data, weights=[1, 2])

    data["weights"].iloc[0] = np.nan
    with pytest.raises(VcovTypeNotSupportedError):
        feols("Y ~ X1", data=data, weights="weights", vcov={"CRV3": "group_id"})

    # test for ValueError when weights are not positive
    data["weights"].iloc[10] = -1
    with pytest.raises(ValueError):
        feols("Y ~ X1", data=data, weights="weights", vcov="iid")

    # test for ValueError when weights are not numeric
    data["weights"].iloc[10] = "a"
    with pytest.raises(ValueError):
        feols("Y ~ X1", data=data, weights="weights", vcov="iid")

    data = get_data()
    with pytest.raises(NotImplementedError):
        feols("Y ~ X1", data=data, weights="weights", vcov="iid").wildboottest(B = 999)
