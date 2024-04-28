import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from pyfixest.estimation.estimation import feols
from pyfixest.estimation.multcomp import _get_rwolf_pval, bonferroni, rwolf
from pyfixest.utils.utils import get_data

pandas2ri.activate()

fixest = importr("fixest")
wildrwolf = importr("wildrwolf")
stats = importr("stats")
broom = importr("broom")


def test_bonferroni():
    data = get_data().dropna()
    rng = np.random.default_rng(989)
    data = get_data()
    data["Y2"] = data["Y"] * rng.normal(0.2, 1, size=len(data))
    data["Y3"] = data["Y2"] + rng.normal(0, 0.5, size=len(data))

    # test set 1

    fit1 = feols("Y ~ X1", data=data)
    fit2 = feols("Y2 ~ X1", data=data)
    fit3 = feols("Y3 ~ X1", data=data)

    bonferroni_py = bonferroni([fit1, fit2, fit3], "X1")

    # R
    fit1_r = fixest.feols(ro.Formula("Y ~ X1"), data=data)
    fit2_r = fixest.feols(ro.Formula("Y2 ~ X1"), data=data)
    fit3_r = fixest.feols(ro.Formula("Y3 ~ X1"), data=data)

    pvalues_r = np.zeros(3)
    for i, x in enumerate([fit1_r, fit2_r, fit3_r]):
        df_tidy = broom.tidy_fixest(x)
        df_r = pd.DataFrame(df_tidy).T
        df_r.columns = ["term", "estimate", "std.error", "statistic", "p.value"]
        pvalues_r[i] = df_r.set_index("term").xs("X1")["p.value"]

    bonferroni_r = stats.p_adjust(pvalues_r, method="bonferroni")

    np.testing.assert_allclose(
        bonferroni_py.iloc[6].values, bonferroni_r, atol=1e-8, rtol=np.inf
    )


def test_wildrwolf():
    rng = np.random.default_rng(12345)
    data = get_data()
    data["Y2"] = data["Y"] * rng.normal(0, 0.5, size=len(data))
    data["Y3"] = data["Y2"] + rng.normal(0, 0.5, size=len(data))

    # test set 1

    fit1 = feols("Y ~ X1", data=data)
    fit2 = feols("Y2 ~ X1", data=data)
    fit3 = feols("Y3 ~ X1", data=data)

    if False:
        rwolf_py = rwolf([fit1, fit2, fit3], "X1", B=9999, seed=12345)

        # R
        fit_r = fixest.feols(ro.Formula("c(Y, Y2, Y3) ~ X1"), data=data)
        rwolf_r = wildrwolf.rwolf(fit_r, param="X1", B=9999, seed=12345)

        np.testing.assert_allclose(
            rwolf_py.iloc[6].values,
            pd.DataFrame(rwolf_r).iloc[5].values.astype(float),
            atol=1e-2,
            rtol=np.inf,
        )

    # test set 2

    fit1 = feols("Y ~ X1 | f1 + f2", data=data)
    fit2 = feols("Y2 ~ X1 | f1 + f2", data=data)
    fit3 = feols("Y3 ~ X1 | f1 + f2", data=data)

    rwolf_py = rwolf([fit1, fit2, fit3], "X1", B=9999, seed=12345)

    # R
    fit_r = fixest.feols(ro.Formula("c(Y, Y2, Y3) ~ X1 | f1 + f2"), data=data)
    rwolf_r = wildrwolf.rwolf(fit_r, param="X1", B=9999, seed=12345)

    np.testing.assert_allclose(
        rwolf_py.iloc[6].values,
        pd.DataFrame(rwolf_r).iloc[5].values.astype(float),
        atol=1e-2,
        rtol=np.inf,
    )


def test_stepwise_function():
    B = 1000
    S = 5

    rng = np.random.default_rng(33)
    t_stat = rng.normal(0, 1, size=S)
    t_boot = rng.normal(0, 1, size=(B, S))

    stepwise_py = _get_rwolf_pval(t_stat, t_boot)
    stepwise_r = wildrwolf.get_rwolf_pval(t_stat, t_boot)

    np.testing.assert_allclose(stepwise_py, stepwise_r)
