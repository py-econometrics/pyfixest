import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# rpy2 imports
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.utils.utils import ssc
from pyfixest.utils.dgps import get_sharkfin


pandas2ri.activate()

fixest = importr("fixest")
stats = importr("stats")
broom = importr("broom")

# note: tolerances are lowered below for
# fepois inference as it is not as precise as feols
# effective tolerances for fepois are 1e-04 and 1e-03
# (the latter only for CRV inferece)
rtol = 1e-08
atol = 1e-08

ols_fmls = [
    ("Y~treat"),
    ("Y~treat + unit"),
    ("Y~treat | unit"),
    ("Y~treat | unit + year"),
]


@pytest.fixture(scope="module")
def data_panel(N=100,T=15,seed=42):
    np.random.seed(seed)
    units=np.repeat(np.arange(N),T)
    time=np.tile(np.arange(T),N)
    treated_units=np.random.choice(N,size=N//2,replace=False)
    treat=np.zeros(N*T,dtype=int)
    midpoint=T//2
    treat[(np.isin(units,treated_units))&(time>=midpoint)]=1
    ever_treated=np.isin(units,treated_units).astype(int)
    alpha=np.random.normal(0,1,N)
    gamma=np.random.normal(0,0.5,T)
    epsilon=np.random.normal(0,0.5,N*T)
    Y=alpha[units]+gamma[time]+treat+epsilon
    weights=np.random.uniform(0,1,N*T)
    return pd.DataFrame({"unit":units,"year":time,"treat":treat, "ever_treated":ever_treated,"Y":Y,"weights":weights})

@pytest.fixture(scope="module")
def data_time():

    N = 200
    rng = np.random.default_rng(9291)
    data = pd.DataFrame({
        "unit": rng.normal(0, 1, N),
        "year": np.arange(N),
        "treat": rng.choice([0, 1], N),
        "weights": rng.uniform(0, 1, N),
    })
    data["Y"] = data["unit"] - data["year"] + 0.5 *data["treat"] + rng.normal(0, 1, N)
    return data

def check_absolute_diff(x1, x2, tol, msg=None):
    "Check for absolute differences."
    if isinstance(x1, (int, float)):
        x1 = np.array([x1])
    if isinstance(x2, (int, float)):
        x2 = np.array([x2])
        msg = "" if msg is None else msg

    # handle nan values
    nan_mask_x1 = np.isnan(x1)
    nan_mask_x2 = np.isnan(x2)

    if not np.array_equal(nan_mask_x1, nan_mask_x2):
        raise AssertionError(f"{msg}: NaN positions do not match")

    valid_mask = ~nan_mask_x1  # Mask for non-NaN elements (same for x1 and x2)
    assert np.all(np.abs(x1[valid_mask] - x2[valid_mask]) < tol), msg


def na_omit(arr):
    mask = ~np.isnan(arr)
    return arr[mask]


def check_relative_diff(x1, x2, tol, msg=None):
    msg = "" if msg is None else msg
    assert np.all(np.abs(x1 - x2) / np.abs(x1) < tol), msg

ALL_F3 = ["str", "object", "int", "categorical", "float"]
SINGLE_F3 = ALL_F3[0]
BACKEND_F3 = [
    *[("numba", t) for t in ALL_F3],
    # *[(b, SINGLE_F3) for b in ("jax", "rust")],
]


@pytest.mark.against_r_core
@pytest.mark.parametrize("dropna", [False, True])
@pytest.mark.parametrize("inference", ["NW"])
@pytest.mark.parametrize(
    "vcov_kwargs",
    [
        {"lag": 2, "time_id": "year"},
        {"lag": 8, "time_id": "year"},
        # now add panel id
        {"lag": 2, "time_id": "year", "panel_id": "unit"},
        {"lag": 8, "time_id": "year", "panel_id": "unit"},
        # lag not required when panel_id is provided
        {"time_id": "year", "panel_id": "unit"},
    ],
)
@pytest.mark.parametrize("weights", [None, "weights"])
@pytest.mark.parametrize("fml", ols_fmls)
def test_single_fit_feols_hac_panel(
    data_panel,
    data_time,
    dropna,
    inference,
    vcov_kwargs,
    weights,
    fml,
):

    adj = False
    cluster_adj = False
    ssc_ = ssc(adj=adj, cluster_adj=cluster_adj)

    lag = vcov_kwargs.get("lag", None)
    time_id = vcov_kwargs.get("time_id", None)
    panel_id = vcov_kwargs.get("panel_id", None)
    data = data_panel if panel_id is not None else data_time

    if "|" in fml and panel_id is None:
        pytest.skip("Don't run fixed effect test when data is not a panel.")

    r_panel_kwars = (
        ({"time": time_id} if time_id is not None else {}) |
        ({"lag": lag} if lag is not None else {}) |
        ({"unit": panel_id} if panel_id is not None else {})
    )

    r_fixest = fixest.feols(
        ro.Formula(fml),
        vcov=fixest.vcov_NW(
            **r_panel_kwars
        ),
        data=data,
        ssc=fixest.ssc(adj, "nested", cluster_adj, "min", "min", False),
        **({"weights": ro.Formula(f"~{weights}")} if weights is not None else {}),
    )

    mod = pf.feols(
        fml=fml,
        data=data,
        vcov=inference,
        vcov_kwargs=vcov_kwargs,
        weights=weights,
        ssc=ssc_,
    )


    # r_fixest to global r env, needed for
    # operations as in dof.K
    ro.globalenv["r_fixest"] = r_fixest

    py_vcov = mod._vcov[0, 0]
    r_vcov = stats.vcov(r_fixest)[0, 0]

    check_absolute_diff(py_vcov, r_vcov, 1e-08, "py_vcov != r_vcov")

@pytest.mark.against_r_core
def test_vcov_updating(data_panel):
    fit_hetero = pf.feols("Y ~ treat", data=data_panel, vcov="hetero")
    fit_nw = pf.feols(
        "Y ~ treat", data=data_panel, vcov="NW", vcov_kwargs={"time_id": "year", "lag": 7}
    )

    fit_hetero.vcov(vcov="NW", vcov_kwargs={"lag": 7, "time_id": "year"})

    assert fit_hetero._vcov_type == "HAC"
    assert fit_hetero._vcov_type_detail == "NW"
    check_absolute_diff(fit_hetero._vcov, fit_nw._vcov, 1e-08, "py_vcov != r_vcov")
