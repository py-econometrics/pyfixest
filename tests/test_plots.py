import pytest
import pandas as pd
from pyfixest.fixest import Fixest
from pyfixest.utils import get_data
import matplotlib

matplotlib.use("Agg")


@pytest.fixture
def data():
    data = get_data()
    data["f2"] = pd.Categorical(data["f2"])
    return data


def test_iplot(data):
    fixest = Fixest(data)
    fixest.feols("Y ~ i(f2, X1) | f1", vcov="iid").iplot()
    fixest.feols("Y ~ i(f2, X1, ref = 1)", vcov="iid").iplot()
    with pytest.raises(ValueError):
        fixest.feols("Y ~ X1", vcov="iid").iplot()


def test_coefplot(data):
    fixest = Fixest(data)
    fixest.feols("Y ~ i(f2, X1) | f1", vcov="iid").coefplot()

    fixest2 = Fixest(data)
    fixest2.feols("Y ~ i(f2, X1) | f1", vcov="iid").coefplot()

    fixest3 = Fixest(data)
    fixest3.feols("Y ~ X1 + X2", vcov="iid").coefplot()
