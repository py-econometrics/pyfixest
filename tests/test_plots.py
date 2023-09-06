import pytest
import pandas as pd
from pyfixest.utils import get_data
import matplotlib
from pyfixest.estimation import feols

matplotlib.use("Agg")


@pytest.fixture
def data():
    data = get_data()
    data["f2"] = pd.Categorical(data["f2"])
    return data


def test_iplot(data):
    feols(fml="Y ~ i(f2, X1) | f1", data=data, vcov="iid").iplot()
    feols(fml="Y ~ i(f2, X1, ref = 1.0)", data=data, vcov="iid").iplot()
    with pytest.raises(ValueError):
        feols(fml="Y ~ X1", data=data, vcov="iid").iplot()


def test_coefplot(data):
    feols(fml="Y ~ i(f2, X1) | f1", vcov="iid", data=data).coefplot()

    feols(fml="Y ~ i(f2, X1) | f1", vcov="iid", data=data).coefplot()

    feols(fml="Y ~ X1 + X2", vcov="iid", data=data).coefplot()
