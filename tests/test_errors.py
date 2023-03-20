import pytest
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





