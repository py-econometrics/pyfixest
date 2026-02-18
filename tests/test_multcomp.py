import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.estimation import feols
from pyfixest.estimation.post_estimation.multcomp import _get_rwolf_pval, bonferroni, rwolf
from pyfixest.utils.check_r_install import check_r_install
from pyfixest.utils.utils import get_data

pandas2ri.activate()

# Core R packages
fixest = importr("fixest")
stats = importr("stats")
broom = importr("broom")
# Extended R packages
if import_check := check_r_install("wildrwolf", strict=False):
    wildrwolf = importr("wildrwolf")


@pytest.mark.against_r_core
@pytest.mark.extended
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

    assert np.all(np.abs(bonferroni_py.iloc[6].values - bonferroni_r) < 0.01), (
        "bonferroni failed"
    )


@pytest.mark.skipif(import_check is False, reason="R package wildrwolf not installed.")
@pytest.mark.against_r_extended
@pytest.mark.extended
@pytest.mark.parametrize("seed", [293, 912, 831])
@pytest.mark.parametrize("sd", [0.5, 1.0, 1.5])
def test_wildrwolf_hc(seed, sd):
    rng = np.random.default_rng(seed)
    data = get_data(N=1_000, seed=seed)
    data["f1"] = rng.choice(range(100), len(data), True)
    data["Y2"] = data["Y"] * rng.normal(0, sd, size=len(data))
    data["Y3"] = data["Y2"] + rng.normal(0, sd, size=len(data))

    # test set 1

    fit = feols("Y + Y2 + Y3~ X1", data=data)
    rwolf_py = rwolf(fit.to_list(), "X1", reps=9999, seed=seed + 2)

    # R
    fit_r = fixest.feols(ro.Formula("c(Y, Y2, Y3) ~ X1"), data=data)
    rwolf_r = wildrwolf.rwolf(fit_r, param="X1", B=9999, seed=seed + 2)

    try:
        np.testing.assert_allclose(
            rwolf_py.iloc[6].values,
            pd.DataFrame(rwolf_r).iloc[5].values.astype(float),
            rtol=0,
            atol=0.01,
            err_msg="rwolf 1 failed",
        )
    except AssertionError:
        rwolf_py = rwolf(fit.to_list(), "X1", reps=29999, seed=seed + 2)
        rwolf_r = wildrwolf.rwolf(fit_r, param="X1", B=29999, seed=seed + 2)

        np.testing.assert_allclose(
            rwolf_py.iloc[6].values,
            pd.DataFrame(rwolf_r).iloc[5].values.astype(float),
            rtol=0,
            atol=0.01,
            err_msg="rwolf 1 failed",
        )


@pytest.mark.skipif(import_check is False, reason="R package wildrwolf not installed.")
@pytest.mark.against_r_extended
@pytest.mark.extended
@pytest.mark.parametrize("seed", [9391])
@pytest.mark.parametrize("sd", [0.5, 1.5])
def test_wildrwolf_crv(seed, sd):
    rng = np.random.default_rng(seed)
    data = get_data(N=4_000, seed=seed)
    data["f1"] = rng.choice(range(100), len(data), True)
    data["Y2"] = data["Y"] * rng.normal(0, sd, size=len(data))
    data["Y3"] = data["Y2"] + rng.normal(0, sd, size=len(data))

    # test set 2

    fit = feols("Y + Y2 + Y3 ~ X1 | f1", data=data)

    rwolf_py = rwolf(fit.to_list(), "X1", reps=9999, seed=seed + 3)

    # R
    fit_r = fixest.feols(ro.Formula("c(Y, Y2, Y3) ~ X1 | f1"), data=data)
    rwolf_r = wildrwolf.rwolf(fit_r, param="X1", B=9999, seed=seed + 3)

    try:
        np.testing.assert_allclose(
            rwolf_py.iloc[6].values,
            pd.DataFrame(rwolf_r).iloc[5].values.astype(float),
            rtol=0,
            atol=0.025,
            err_msg="rwolf 2 failed",
        )

    except AssertionError:
        rwolf_py = rwolf(fit.to_list(), "X1", reps=19999, seed=seed + 3)
        rwolf_r = wildrwolf.rwolf(fit_r, param="X1", B=19999, seed=seed + 3)

        np.testing.assert_allclose(
            rwolf_py.iloc[6].values,
            pd.DataFrame(rwolf_r).iloc[5].values.astype(float),
            rtol=0,
            atol=0.025,
            err_msg="rwolf 2 failed",
        )


@pytest.mark.skipif(import_check is False, reason="R package wildrwolf not installed.")
@pytest.mark.against_r_extended
@pytest.mark.extended
def test_stepwise_function():
    B = 1000
    S = 5

    rng = np.random.default_rng(33)
    t_stat = rng.normal(0, 1, size=S)
    t_boot = rng.normal(0, 1, size=(B, S))

    stepwise_py = _get_rwolf_pval(t_stat, t_boot)
    stepwise_r = wildrwolf.get_rwolf_pval(t_stat, t_boot)

    np.testing.assert_allclose(stepwise_py, stepwise_r)


# Import data from pyfixest


@pytest.mark.extended
@pytest.mark.parametrize("seed", [453])
@pytest.mark.parametrize("reps", [499])
def test_sampling_scheme(seed, reps):
    # Compare RW adjusted p-values from RI and WB resampling methods
    # The p-values should be "largely" similar regardless of resampling methods

    data = get_data().dropna()
    rng = np.random.default_rng(seed)

    # Perturb covariates(not treatment variable)
    data["Y2"] = data["Y"] * rng.normal(0.2, 1, size=len(data))
    data["X2"] = rng.normal(size=len(data))

    fit1 = feols("Y ~ X1 + X2", data=data)
    fit2 = feols("Y ~ X1 + X2 + Y2", data=data)

    # Run rwolf with "ri" sampling method
    rwolf_df_ri = rwolf([fit1, fit2], "X1", reps=reps, seed=seed, sampling_method="ri")

    ri_pval = rwolf_df_ri["est1"]["RW Pr(>|t|)"]

    # Run rwolf with "wild-bootstrap" sampling method
    rwolf_df_wb = rwolf(
        [fit1, fit2], "X1", reps=reps, seed=seed, sampling_method="wild-bootstrap"
    )

    wb_pval = rwolf_df_wb["est1"]["RW Pr(>|t|)"]

    # Calculate percentage difference in p-values
    percent_diff = 100 * (wb_pval - ri_pval) / ri_pval

    print(
        f"Percentage difference in p-values (seed={seed}, reps={reps}): {percent_diff}"
    )

    # Assert that the percentage difference is within an acceptable range
    assert np.abs(percent_diff) < 1.0, (
        f"Percentage difference is too large: {percent_diff}%"
    )


@pytest.mark.extended
def test_multi_vs_list():
    "Test that lists of models and FixestMulti input produce identical results."
    seed = 1232
    reps = 100
    data = pf.get_data(N=100)

    fit_all = pf.feols("Y + Y2 ~ X1 + X2", data=data)
    fit1 = pf.feols("Y ~ X1 + X2", data=data)
    fit2 = pf.feols("Y2 ~ X1 + X2", data=data)

    assert bonferroni(fit_all, "X1").equals(bonferroni([fit1, fit2], "X1"))
    assert rwolf(fit_all, "X1", seed=seed, reps=reps).equals(
        rwolf([fit1, fit2], "X1", seed=seed, reps=reps)
    )


@pytest.mark.extended
@pytest.mark.parametrize("fml", ["Y ~ X1", "Y ~ X1 + X2"])
@pytest.mark.parametrize("seed", [199])
@pytest.mark.parametrize("sampling_method", ["ri", "wild-bootstrap"])
def test_rwolf_vs_wyoung(fml, seed, sampling_method):
    data = pf.get_data(N=100)
    fml1 = fml
    fml2 = f"{fml1} + f1"

    fit1 = pf.feols(fml1, data=data)
    fit2 = pf.feols(fml2, data=data)

    rwolf_output = pf.rwolf(
        [fit1, fit2], "X1", reps=99, seed=seed, sampling_method=sampling_method
    )
    wyoung_output = pf.wyoung(
        [fit1, fit2], "X1", reps=99, seed=seed, sampling_method=sampling_method
    )

    # test that the two pandas dfs are close
    assert np.allclose(rwolf_output, wyoung_output, atol=0.01)
