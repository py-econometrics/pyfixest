from pyfixest.estimation import feols, fepois
from pyfixest.utils import get_data
import polars as pl


def test_polars_input():
    data = get_data()
    data_pl = pl.from_pandas(data)
    fit = feols("Y ~ X1", data=data)

    data = get_data(model="Fepois")
    data_pl = pl.from_pandas(data)
    fit = fepois("Y ~ X1", data=data_pl)
