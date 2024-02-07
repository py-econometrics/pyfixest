from pyfixest.estimation import feols, fepois
from pyfixest.utils import get_data
from pyfixest.visualize import coefplot, iplot


def test_visualize():

    data = get_data()
    fit1 = feols("Y ~ X1 + X2 | f1", data=data)
    coefplot(fit1)

    data_pois = get_data(model="Fepois")
    fit2 = fepois("Y ~ X1 + X2 + f2 | f1", data=data_pois, vcov={"CRV1": "f1+f2"})
    coefplot(fit2)

    coefplot([fit1, fit2])

    # FixestMulti
    fit3 = feols("Y + Y2 ~ X1 + X2 | f1", data=data)
    coefplot(fit3.to_list())

    fit3.coefplot()

    # iplot
    fit5 = feols("Y ~ i(f1)", data=data)
    iplot(fit5)
