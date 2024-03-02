import pyfixest as pf
from pyfixest.utils.utils import get_data


def test_api():
    df1 = get_data()
    df2 = get_data(model="Fepois")

    fit1 = pf.feols("Y ~ X1 + X2 | f1", data=df1)
    fit2 = pf.estimation.fepois(
        "Y ~ X1 + X2 + f2 | f1", data=df2, vcov={"CRV1": "f1+f2"}
    )
    pf.summary(fit1)
    pf.report.summary(fit2)
    pf.etable([fit1, fit2])
    pf.coefplot([fit1, fit2])
