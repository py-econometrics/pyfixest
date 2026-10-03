import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.estimation import feols
from pyfixest.estimation.formula.transforms.misc import log


def test_log_accepts_nullable_pandas_series_with_missing_values():
    with pytest.warns(UserWarning, match="1 rows with infinite values detected"):
        result = log(pd.Series([1, pd.NA, 3], dtype="Int64"))

    np.testing.assert_allclose(result, [0.0, np.nan, np.log(3)], equal_nan=True)


def test_log_preserves_numpy_array_shape():
    array = np.array([[1.0, 4.0], [9.0, 16.0]])

    result = log(array)

    np.testing.assert_allclose(result, np.log(array))
    assert result.shape == array.shape


@pytest.mark.parametrize("dtype", ["Float64", "Int64"])
def test_feols_log_matches_numpy_log_for_nullable_columns(dtype):
    data = pf.get_data().dropna().copy()
    data["X3"] = pd.Series(np.arange(1, len(data) + 1), index=data.index).astype(dtype)
    data.loc[data.index[[7, 8]], "X3"] = pd.NA

    with pytest.warns(UserWarning, match="rows with infinite values detected"):
        fit_log = feols("Y ~ log(X3)", data=data)
    fit_np_log = feols("Y ~ np.log(X3)", data=data)

    assert fit_log.sample_info.n_obs == fit_np_log.sample_info.n_obs
    np.testing.assert_allclose(
        fit_log.coef().to_numpy(),
        fit_np_log.coef().to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
