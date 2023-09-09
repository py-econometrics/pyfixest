import pytest
import numpy as np
import pandas as pd
from pyfixest.estimation import fepois


def test_separation():
    """
    Test separation detection.
    """

    y = np.array([0, 0, 0, 1, 2, 3])
    df1 = np.array(["a", "a", "b", "b", "b", "c"])
    df2 = np.array(["c", "c", "d", "d", "d", "e"])
    x = np.random.normal(0, 1, 6)

    df = pd.DataFrame({"Y": y, "fe1": df1, "fe2": df2, "x": x})

    with pytest.warns(
        UserWarning, match="2 observations removed because of only 0 outcomes"
    ):
        mod = fepois("Y ~ x  | fe1", data=df, vcov="hetero")
    # mod._check_for_separation()

    # np.allclose(mod.separation_na, np.array([0, 1]))
    # np.allclose(mod.n_separation_na, 2)
