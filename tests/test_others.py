import numpy as np
import pandas as pd
import polars as pl

from pyfixest.estimation.estimation import feols, fepois
from pyfixest.report.utils import (
    rename_categoricals,
    rename_event_study_coefs,
)
from pyfixest.utils.utils import capture_context, get_data, ssc


def test_multicol_overdetermined_iv():
    data = get_data()
    fit = feols(
        fml="Y ~ X2 +  f1| f1 | X1 ~ Z1 + Z2",
        data=data,
        ssc=ssc(k_adj=False),
        vcov={"CRV1": "f1"},
    )

    assert fit._collin_vars == ["f1"]
    assert fit._collin_vars_z == ["f1"]

    np.testing.assert_allclose(
        fit._beta_hat[::-1], np.array([-0.993607, -0.174227], dtype=float), rtol=1e-5
    )
    np.testing.assert_allclose(fit._se[::-1], np.array([0.104009, 0.018416]), rtol=1e-5)


def test_polars_input():
    data = get_data()
    data_pl = pl.from_pandas(data)
    fit = feols("Y ~ X1", data=data)
    fit.predict(newdata=data_pl)

    data = get_data(model="Fepois")
    data_pl = pl.from_pandas(data)
    fit = fepois("Y ~ X1", data=data_pl)


def test_integer_XY():
    # Create a random number generator
    rng = np.random.default_rng()

    N = 1000
    X = rng.normal(0, 1, N)
    f = rng.choice([0, 1], N)
    Y = 2 * X + rng.normal(0, 1, N) + f * 2
    Y = np.round(Y).astype(np.int64)
    X = np.round(X).astype(np.int64)

    df = pd.DataFrame({"Y": Y, "X": X, "f": f})

    fit1 = feols("Y ~ X | f", data=df, vcov="iid")
    fit2 = feols("Y ~ X + C(f)", data=df)

    np.testing.assert_allclose(fit1.coef().xs("X"), fit2.coef().xs("X"))


def test_coef_update():
    data = get_data()
    data_subsample = data.sample(frac=0.5)
    m = feols("Y ~ X1 + X2", data=data_subsample)
    new_points_id = np.random.choice(
        list(set(data.index) - set(data_subsample.index)), 5
    )
    X_new, y_new = (
        np.c_[
            np.ones(len(new_points_id)), data.loc[new_points_id][["X1", "X2"]].values
        ],
        data.loc[new_points_id]["Y"].values,
    )
    updated_coefs = m.update(X_new, y_new)
    full_coefs = (
        feols(
            "Y ~ X1 + X2",
            data=data.loc[data_subsample.index.append(pd.Index(new_points_id))],
        )
        .coef()
        .values
    )

    np.testing.assert_allclose(updated_coefs, full_coefs)


def test_coef_update_inplace():
    data = get_data()
    data_subsample = data.sample(frac=0.3)
    m = feols("Y ~ X1 + X2", data=data_subsample)
    new_points_id = np.random.choice(
        list(set(data.index) - set(data_subsample.index)), 5
    )
    X_new, y_new = (
        np.c_[
            data.loc[new_points_id][
                ["X1", "X2"]
            ].values  # only pass columns; let `update` add the intercept
        ],
        data.loc[new_points_id]["Y"].values,
    )
    m.update(X_new, y_new, inplace=True)
    full_coefs = (
        feols(
            "Y ~ X1 + X2",
            data=data.loc[data_subsample.index.append(pd.Index(new_points_id))],
        )
        .coef()
        .values
    )
    np.testing.assert_allclose(m.coef().values, full_coefs)


def test_rename_categoricals():
    coefnames = ["C(var)[T.1]", "C(var)[T.2]", "C(var2)[T.1]", "C(var2)[T.2]"]
    renamed = rename_categoricals(coefnames)
    assert renamed == {
        "C(var)[T.1]": "var::1",
        "C(var)[T.2]": "var::2",
        "C(var2)[T.1]": "var2::1",
        "C(var2)[T.2]": "var2::2",
    }

    # with strings:
    coefnames = ["Intercept", "C(f4)[T.B]", "C(f4)[T.C]"]
    renamed = rename_categoricals(coefnames)
    assert renamed == {
        "Intercept": "Intercept",
        "C(f4)[T.B]": "f4::B",
        "C(f4)[T.C]": "f4::C",
    }

    # with reference levels:
    coefnames = [
        "Intercept",
        "C(f4, contr.treatment(base='A'))[T.B]",
        "C(f4, contr.treatment(base='A'))[T.C]",
    ]
    renamed = rename_categoricals(coefnames)
    assert renamed == {
        "Intercept": "Intercept",
        "C(f4, contr.treatment(base='A'))[T.B]": "f4::B",
        "C(f4, contr.treatment(base='A'))[T.C]": "f4::C",
    }

    # without 'T.' in the categorical notation:
    coefnames = [
        "C(f4)[B]",
        "C(f4)[C]",
    ]
    renamed = rename_categoricals(coefnames)
    assert renamed == {
        "C(f4)[B]": "f4::B",
        "C(f4)[C]": "f4::C",
    }

    # without C() and no 'T.' notation
    coefnames = [
        "f4[B]",
        "f4[C]",
    ]
    renamed = rename_categoricals(coefnames)
    assert renamed == {
        "f4[B]": "f4::B",
        "f4[C]": "f4::C",
    }

    # with categoricals:
    coefnames = ["Intercept", "variable1[T.value1]", "variable1[T.value2]"]
    renamed = rename_categoricals(coefnames)
    assert renamed == {
        "Intercept": "Intercept",
        "variable1[T.value1]": "variable1::value1",
        "variable1[T.value2]": "variable1::value2",
    }

    # Test with labels
    coefnames = ["C(variable1)[T.value1]", "variable2[T.value2]"]
    labels = {"variable1": "var1", "variable2": "var2"}
    renamed = rename_categoricals(coefnames, labels=labels)
    assert renamed == {
        "C(variable1)[T.value1]": "var1::value1",
        "variable2[T.value2]": "var2::value2",
    }

    # Test with custom template
    coefnames = ["C(variable1)[T.value1]", "variable2[T.value2]"]
    template = "{variable}--{value}"
    renamed = rename_categoricals(coefnames, template=template)
    assert renamed == {
        "C(variable1)[T.value1]": "variable1--value1",
        "variable2[T.value2]": "variable2--value2",
    }


def test_rename_event_study_coefs():
    coefnames = [
        "C(rel_year, contr.treatment(base=-1.0))[T.-20.0]",
        "C(rel_year, contr.treatment(base=-1.0))[T.-19.0]",
        "Intercept",
    ]

    renamed = rename_event_study_coefs(coefnames)
    assert renamed == {
        "C(rel_year, contr.treatment(base=-1.0))[T.-20.0]": "rel_year::-20.0",
        "C(rel_year, contr.treatment(base=-1.0))[T.-19.0]": "rel_year::-19.0",
        "Intercept": "Intercept",
    }


def _foo():
    "Simulate a callable for testing context capture behavior."
    ...


def test_context_capture():
    # `_foo` is in caller's stack frame, if should be captured
    # call with -1 to account for adding one more frame inside the function
    context = capture_context(-1)
    assert "_foo" in context

    # `_foo` is in caller's stack frame, but we ask for a deeper stack, `_foo` should not be captured
    context = capture_context(1)
    assert "_foo" not in context

    context = capture_context({})
    assert context == {}

    context = capture_context({"_foo": _foo})
    assert context == {"_foo": _foo}
