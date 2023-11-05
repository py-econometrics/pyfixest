import pytest
import numpy as np
import pandas as pd
from pyfixest.estimation import feols
from pyfixest.exceptions import InvalidReferenceLevelError


def test_i():
    df_het = pd.read_csv("pyfixest/experimental/data/df_het.csv")
    df_het["X"] = np.random.normal(size=len(df_het))

    if (
        "C(rel_year)[T.1.0]"
        in feols("dep_var~i(rel_year)", df_het, i_ref1=1.0)._coefnames
    ):
        raise AssertionError("C(rel_year)[T.1.0] should not be in the column names.")
    if (
        "C(rel_year)[T.-2.0]"
        in feols("dep_var~i(rel_year)", df_het, i_ref1=-2.0)._coefnames
    ):
        raise AssertionError("C(rel_year)[T.-2.0] should not be in the column names.")
    if (
        "C(rel_year)[T.1.0]"
        in feols("dep_var~i(rel_year)", df_het, i_ref1=[1.0, 2.0])._coefnames
    ):
        raise AssertionError("C(rel_year)[T.1.0] should not be in the column names.")
    if (
        "C(rel_year)[T.2.0]"
        in feols("dep_var~i(rel_year)", df_het, i_ref1=[1.0, 2.0])._coefnames
    ):
        raise AssertionError("C(rel_year)[T.2.0] should not be in the column names.")

    if (
        "C(rel_year)[T.1.0]:treat"
        in feols("dep_var~i(rel_year, treat)", df_het, i_ref1=1.0)._coefnames
    ):
        raise AssertionError(
            "C(rel_year)[T.1.0]:treat should not be in the column names."
        )
    if (
        "C(rel_year)[T.-2.0]:treat"
        in feols("dep_var~i(rel_year, treat)", df_het, i_ref1=-2.0)._coefnames
    ):
        raise AssertionError(
            "C(rel_year)[T.-2.0]:treat should not be in the column names."
        )
    if (
        "C(rel_year)[T.1.0]:treat"
        in feols("dep_var~i(rel_year, treat)", df_het, i_ref1=[1.0, 2.0])._coefnames
    ):
        raise AssertionError(
            "C(rel_year)[T.1.0]:treat should not be in the column names."
        )
    if (
        "C(rel_year)[T.2.0]:treat"
        in feols("dep_var~i(rel_year, treat)", df_het, i_ref1=[1.0, 2.0])._coefnames
    ):
        raise AssertionError(
            "C(rel_year)[T.2.0]:treat should not be in the column names."
        )

    with pytest.raises(InvalidReferenceLevelError):
        feols("dep_var~i(rel_year, treat)", df_het, i_ref1="1.0")
    with pytest.raises(InvalidReferenceLevelError):
        feols("dep_var~i(rel_year, treat)", df_het, i_ref1=[1])
    with pytest.raises(InvalidReferenceLevelError):
        feols("dep_var~i(rel_year, X)", df_het, i_ref1=[1, 2])
    with pytest.raises(AssertionError):
        feols("dep_var~i(rel_year, X)", df_het, i_ref1=[1.0, "a"])

    # i_ref2 currently not supported
    with pytest.raises(AssertionError):
        feols("dep_var~i(rel_year, treat)", df_het, i_ref2="1.0")
