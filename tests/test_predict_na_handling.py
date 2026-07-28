"""Tests for NaN and singleton handling in Feols.predict (issue #1236).

These tests pin down the documented contract of `Feols.predict`:

* ``newdata=None`` predicts on the estimation sample, where rows with NaNs and
  singleton fixed effect levels were already dropped during fitting. The result
  therefore has length ``_N`` and contains no NaNs.
* ``newdata=<df>`` predicts on every row of the passed data. Rows that cannot be
  predicted (NaN in a right hand side covariate, or an unseen fixed effect level)
  return NaN, and ``predict`` warns about them.

No R / rpy2 dependency so these run in the default test environment.
"""

import warnings

import numpy as np
import pytest

import pyfixest as pf


def test_predict_none_drops_nas_newdata_keeps_rows():
    """newdata=None returns the estimation sample; newdata keeps every row."""
    data = pf.get_data(N=1_000, seed=0, model="Feols")
    fit = pf.feols("Y ~ X1", data=data)

    pred_none = fit.predict(newdata=None)
    pred_full = fit.predict(newdata=data)

    # newdata=None: estimation sample only, NaN rows already dropped.
    assert len(pred_none) == fit._N
    assert data.shape[0] > fit._N
    assert not np.isnan(pred_none).any()

    # newdata=data: one prediction per input row.
    assert len(pred_full) == data.shape[0]

    # a row with a NaN covariate cannot be predicted -> NaN.
    nan_x1 = np.where(data["X1"].isna().to_numpy())[0]
    assert nan_x1.size > 0
    assert np.isnan(pred_full[nan_x1]).all()


def test_predict_newdata_covariate_nan_warns():
    """predict(newdata) must warn when a covariate NaN forces NaN predictions."""
    data = pf.get_data(N=1_000, seed=0, model="Feols")
    fit = pf.feols("Y ~ X1", data=data)

    assert data["X1"].isna().any()

    with pytest.warns(UserWarning, match="cannot be predicted"):
        pred = fit.predict(newdata=data)

    assert np.isnan(pred).any()


def test_predict_newdata_no_missing_does_not_warn():
    """No warning when newdata has no rows that fail to predict."""
    data = pf.get_data(N=1_000, seed=0, model="Feols").dropna()
    fit = pf.feols("Y ~ X1", data=data)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        pred = fit.predict(newdata=data)

    assert len(pred) == data.shape[0]
    assert not np.isnan(pred).any()
    assert not [w for w in record if issubclass(w.category, UserWarning)]
