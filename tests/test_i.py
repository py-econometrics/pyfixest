import pytest
import numpy as np
import pandas as pd
from pyfixest.estimation import feols

def test_i():

    df_het = pd.read_csv("pyfixest/experimental/data/df_het.csv")
    df_het["X"] = np.random.normal(size = len(df_het))

    df_het["rel_year"] = df_het["rel_year"].astype("category")
    df_het["treat"] = df_het["treat"].astype("int")

    feols("dep_var~i(rel_year, treat)", df_het, i_ref1 = 1.0)
    feols("dep_var~i(rel_year, year)", df_het, i_ref1 = 1.0)
    feols("dep_var~i(rel_year)", df_het, i_ref1 = 1.0)

    feols("dep_var~i(rel_year, treat)", df_het, i_ref1 = [1.0, 2.0])
    feols("dep_var~i(rel_year, year)", df_het, i_ref1 = [1.0, 2.0])
    feols("dep_var~i(rel_year)", df_het, i_ref1 = [1.0, 2.0])


    feols("dep_var~i(rel_year, treat)", df_het)
    feols("dep_var~i(rel_year, year)", df_het)
    feols("dep_var~i(rel_year)", df_het)



