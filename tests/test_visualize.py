import pandas as pd

from pyfixest.did.visualize import panelview
from pyfixest.estimation.estimation import feols, fepois
from pyfixest.report.visualize import coefplot, iplot
from pyfixest.utils.utils import get_data


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

    # FixestMulti
    fit6 = feols("Y + Y2 ~ X1 + X2 | f1", data=data)
    fit6.coefplot()


def test_panelview():
    df_het = pd.read_csv("pyfixest/did/data/df_het.csv")
    df_het.head()

    panelview(
        data=df_het,
        unit="unit",
        time="year",
        treat="treat",
        subsamp=50,
        title="Treatment Assignment",
    )
