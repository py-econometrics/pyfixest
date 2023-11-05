from pyfixest.estimation import feols, fepois
from pyfixest.utils import get_data
from pyfixest.summarize import summary, etable


def test_summary():
    """Just run etable() and summary() on a few models."""

    df1 = get_data()
    df2 = get_data(model="Fepois")

    fit1 = feols("Y ~ X1 + X2 | f1", data=df1)
    fit2 = fepois("Y ~ X1 + X2 | f1", data=df2)

    summary(fit1)
    summary(fit2)
    summary([fit1, fit2])

    etable(fit1)
    etable(fit2)
    etable([fit1, fit2])
