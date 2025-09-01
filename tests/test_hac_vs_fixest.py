import numpy as np
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# rpy2 imports
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.utils.utils import ssc
from tests.test_vs_fixest import _c_to_as_factor, get_data_r

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
    ("Y~X1"),
    ("Y~X1+X2"),
    ("Y~X1|f2"),
    ("Y~X1|f2+f3"),
]


@pytest.fixture(scope="module")
def data_feols(N=1000, seed=76540251, beta_type="2", error_type="2"):
    data = pf.get_data(
        N=N, seed=seed, beta_type=beta_type, error_type=error_type, model="Feols"
    )

    data["time"] = np.arange(data.shape[0])
    return data


rng = np.random.default_rng(875)


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


test_counter_feols = 0
test_counter_fepois = 0
test_counter_feiv = 0

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
        # {},   # default lag, assume sorting
        # {"lag":7},
        # {"time_id": "time"},
        {"lag": 2, "time_id": "time"},
        {"lag": 5, "time_id": "time"},
    ],
)
@pytest.mark.parametrize("weights", [None, "weights"])
@pytest.mark.parametrize("fml", ols_fmls)
def test_single_fit_feols_hac(
    data_feols,
    dropna,
    inference,
    vcov_kwargs,
    weights,
    fml,
):
    global test_counter_feols
    test_counter_feols += 1

    adj = True
    cluster_adj = True
    ssc_ = ssc(adj=adj, cluster_adj=cluster_adj)

    lag = vcov_kwargs.get("lag", None)
    time_id = vcov_kwargs.get("time_id", None)
    # panel_id = vcov_kwargs.get("panel_id", None)

    data = data_feols.copy()

    if dropna:
        data = data.dropna()

    # long story, but categories need to be strings to be converted to R factors,
    # this then produces 'nan' values in the pd.DataFrame ...
    data[data == "nan"] = np.nan

    data_r = get_data_r(fml, data)
    r_fml = _c_to_as_factor(fml)

    mod = pf.feols(
        fml=fml,
        data=data,
        vcov=inference,
        vcov_kwargs=vcov_kwargs,
        weights=weights,
        ssc=ssc_,
    )
    if weights is not None:
        r_fixest = fixest.feols(
            ro.Formula(r_fml),
            vcov=fixest.vcov_NW(
                **{
                    **({} if lag is None else {"lag": lag}),
                    **({} if time_id is None else {"time": time_id}),
                }
            ),
            data=data_r,
            ssc=fixest.ssc(adj, "nested", cluster_adj, "min", "min", False),
            weights=ro.Formula("~" + weights),
        )
    else:
        r_fixest = fixest.feols(
            ro.Formula(r_fml),
            vcov=fixest.vcov_NW(
                **{
                    **({} if lag is None else {"lag": lag}),
                    **({} if time_id is None else {"time": time_id}),
                }
            ),
            data=data_r,
            ssc=fixest.ssc(adj, "nested", cluster_adj, "min", "min", False),
        )

    # r_fixest to global r env, needed for
    # operations as in dof.K
    ro.globalenv["r_fixest"] = r_fixest

    py_vcov = mod._vcov[0, 0]
    r_vcov = stats.vcov(r_fixest)[0, 0]

    check_absolute_diff(py_vcov, r_vcov, 1e-08, "py_vcov != r_vcov")


def test_vcov_updating(data_feols):
    fit_hetero = pf.feols("Y ~ X1", data=data_feols, vcov="hetero")
    fit_nw = pf.feols(
        "Y ~ X1", data=data_feols, vcov="NW", vcov_kwargs={"time_id": "time", "lag": 7}
    )

    fit_hetero.vcov(vcov="NW", vcov_kwargs={"lag": 7, "time_id": "time"})

    assert fit_hetero._vcov_type == "HAC"
    assert fit_hetero._vcov_type_detail == "NW"
    check_absolute_diff(fit_hetero._vcov, fit_nw._vcov, 1e-08, "py_vcov != r_vcov")
