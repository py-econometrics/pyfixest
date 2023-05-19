import pytest
import numpy as np
import pandas as pd
from pyfixest import Fixest
from pyfixest.utils import get_data

from pyfixest.FormulaParser import FixestFormulaParser, DuplicateKeyError, FixedEffectInteractionError, CovariateInteractionError

def test_formula_parser():
    with pytest.raises(DuplicateKeyError):
        FixestFormulaParser('y ~ i(X1, X2) + i(X3, X4)')

def test_formula_parser1():
    with pytest.raises(DuplicateKeyError):
        FixestFormulaParser('y ~ i(X1, X2) + i(X3, X4) | X5')

def  test_formula_parser2():
    with pytest.raises(DuplicateKeyError):
        FixestFormulaParser('y ~ sw(a, b) +  sw(c, d)| sw(X3, X4))')

def test_formula_parser3():
    with pytest.raises(DuplicateKeyError):
        FixestFormulaParser('y ~ sw(a, b) +  csw(c, d)| sw(X3, X4))')

#def test_formula_parser2():
#    with pytest.raises(FixedEffectInteractionError):
#        FixestFormulaParser('y ~ X1 + X2 | X3:X4')

#def test_formula_parser3():
#    with pytest.raises(CovariateInteractionError):
#        FixestFormulaParser('y ~ X1 + X2^X3')

def test_i_ref():

    data = get_data()
    fixest = Fixest(data)

    with pytest.raises(ValueError):
        fixest.feols('y ~ i(X1, X2, ref = -1)', vcov = 'iid')

def test_cluster_na():

    '''
    test if a nan value in a cluster variable raises
    an error
    '''

    data = get_data()
    #data = data.dropna()
    data['X3'] = data['X3'].astype('int64')
    data['X3'][5] = np.nan

    fixest = Fixest(data)
    with pytest.raises(ValueError):
        fixest.feols('Y ~ X1', vcov = {'CRV1': 'X3'})

def test_error_hc23_fe():

    '''
    test if HC2&HC3 inference with fixed effects regressions raises an error (currently not supported)
    '''
    data = get_data().dropna()

    fixest = Fixest(data)
    with pytest.raises(ValueError):
        fixest.feols('Y ~ X1 | X2', vcov = "HC2")

    with pytest.raises(ValueError):
        fixest.feols('Y ~ X1 | X2', vcov = "HC3")



def test_depvar_numeric():

    '''
    test if feols() throws an error when the dependent variable is not numeric
    '''

    data = get_data()
    data['Y'] = data['Y'].astype('str')
    data['Y'] = pd.Categorical(data['Y'])

    fixest = Fixest(data)
    with pytest.raises(ValueError):
        fixest.feols('Y ~ X1')


def test_iv_errors():

    data = get_data()
    data["Z1"] = data["X1"] + np.random.normal(0, 1, data.shape[0])
    data["Z2"] = data["X2"] + np.random.normal(0, 1, data.shape[0])


    fixest = Fixest(data)
    # under determined
    with pytest.raises(ValueError):
        fixest.feols('Y ~ X1 | Z1 + Z2 ~ X1 ')
    # instrument specified as covariate
    with pytest.raises(ValueError):
        fixest.feols('Y ~ X1 | Z1  ~ X1 + X2')
    # endogeneous variable specified as covariate
    with pytest.raises(ValueError):
        fixest.feols('Y ~ Z1 | Z1  ~ X1')
    # instrument specified as covariate
    with pytest.raises(ValueError):
        fixest.feols('Y ~ X1 | Z1 + Z2 ~ X1 + X2')
    # CRV3 inference
    with pytest.raises(ValueError):
        fixest.feols('Y ~ 1 | Z1 ~ X1 ', vcov = {"CRV3":"group_id"})
    # wild bootstrap
    with pytest.raises(ValueError):
        fixest.feols('Y ~ 1 | Z1 ~ X1 ').wildboottest(param = "Z1", B = 999)





