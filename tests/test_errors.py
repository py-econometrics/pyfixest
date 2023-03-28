import pytest
import numpy as np
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

def test_error_crv3_fe():

    '''
    test if CRV3 inference with fixed effects regressions raises an error (currently not supported)
    '''
    data = get_data()
    data["group_id"][9] = np.nan

    fixest = Fixest(data)
    with pytest.raises(AssertionError):
        fixest.feols('Y ~ X1 | X2', vcov = {'CRV3': 'group_id'})

