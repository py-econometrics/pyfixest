import pandas as pd
import pytest

from pyfixest.estimation.estimation import feols
from pyfixest.report.visualize import coefplot, iplot
from pyfixest.utils.utils import get_data


@pytest.fixture
def data():
    data = get_data()
    data["f2"] = pd.Categorical(data["f2"])
    return data


def test_iplot(data):
    fit1 = feols(fml="Y ~ i(f2, X1) | f1", data=data, vcov="iid")
    fit2 = feols(fml="Y ~ i(f2, X1) | f2", data=data, vcov="iid")
    fit3 = feols(fml="Y ~ i(f2, X1)", data=data, vcov="iid", i_ref1=1.0)

    fit1.iplot()
    fit2.iplot()
    fit1.iplot(yintercept=0)

    iplot(fit1)
    iplot([fit1, fit2])
    iplot([fit1, fit2], yintercept=0)

    with pytest.raises(ValueError):
        fit3 = feols(fml="Y ~ X1", data=data, vcov="iid")
        fit3.iplot()
        iplot(fit3)

    fit_multi = feols(fml="Y + Y2 ~ i(f2, X1)", data=data)
    fit_multi.iplot()


def test_coefplot(data):
    fit1 = feols(fml="Y ~ i(f2, X1) | f1", vcov="iid", data=data)
    fit2 = feols(fml="Y ~ i(f2, X1) | f1", vcov="iid", data=data)
    fit3 = feols(fml="Y ~ X1 + X2", vcov="iid", data=data)

    fit1.coefplot()
    fit2.coefplot()
    fit3.coefplot()
    coefplot(fit1)
    coefplot([fit1, fit2])
    coefplot([fit1, fit2], yintercept=0)

    fit_multi = feols(fml="Y + Y2 ~ i(f2, X1)", data=data)
    fit_multi.coefplot()
