import numpy as np
import pandas as pandas

from pyfixest.estimation import feols
from pyfixest.utils import get_data


def test_confint():
    """Test the confint method of the feols class."""
    data = get_data()
    fit = feols("Y ~ X1 + X2 + C(f1)", data=data)
    confint = fit.confint()

    np.testing.assert_allclose(confint, fit.confint(alpha=0.05))
    assert np.all(confint.loc[:, "0.025%"] == fit.confint(alpha=0.05).loc[:, "0.025%"])
    assert np.all(confint.loc[:, "0.975%"] == fit.confint(alpha=0.05).loc[:, "0.975%"])
    assert np.all(confint.loc[:, "0.025%"] < fit.confint(alpha=0.10).loc[:, "0.05%"])
    assert np.all(confint.loc[:, "0.975%"] > fit.confint(alpha=0.10).loc[:, "0.95%"])

    # simultaneous CIs:
    for x in range(25):
        assert np.all(
            confint.loc[:, "0.025%"]
            > fit.confint(alpha=0.05, joint_cis=True).loc[:, "0.025%"]
        )
        assert np.all(
            confint.loc[:, "0.975%"]
            < fit.confint(alpha=0.05, joint_cis=True).loc[:, "0.975%"]
        )

    # test seeds
    confint1 = fit.confint(joint_cis=True, seed=1)
    confint2 = fit.confint(joint_cis=True, seed=1)
    confint3 = fit.confint(joint_cis=True, seed=2)

    assert np.all(confint1 == confint2)
    assert np.all(confint1 != confint3)
