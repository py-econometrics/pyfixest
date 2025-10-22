import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# rpy2 imports
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.utils.utils import ssc

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
poisson_fmls = [
    ("Y_count~treat"),
    ("Y_count~treat + unit"),
    ("Y_count~treat | unit"),
    ("Y_count~treat | unit + year"),
]
glm_fmls = [
    ("Y_binary~treat"),
    ("Y_binary~treat + unit"),
]


@pytest.fixture(scope="module")
def data_panel(N=1000, T=30, seed=421):
    rng = np.random.default_rng(seed)
    units = np.repeat(np.arange(N), T)
    time = np.tile(np.arange(T), N)
    treated_units = rng.choice(N, size=N // 2, replace=False)
    treat = np.zeros(N * T, dtype=int)
    midpoint = T // 2
    treat[(np.isin(units, treated_units)) & (time >= midpoint)] = 1
    ever_treated = np.isin(units, treated_units).astype(int)
    alpha = rng.normal(0, 1, N)
    gamma = np.random.normal(0, 0.5, T)

    # Generate AR(1) errors within each unit (rho=0.8 for strong autocorrelation)
    epsilon = np.empty(N * T)
    rho = 0.5
    for i in range(N):
        # Get indices for this unit
        start_idx = i * T
        end_idx = (i + 1) * T
        # Generate innovations
        innovations = rng.normal(0, 5, T)
        # Build AR(1) process within unit
        errors = np.empty(T)
        errors[0] = innovations[0]
        for t in range(1, T):
            errors[t] = rho * errors[t - 1] + innovations[t]
        epsilon[start_idx:end_idx] = errors

    Y = alpha[units] + gamma[time] + treat + epsilon

    # Add count variable for Poisson models
    # Ensure it's positive by exponentiating a scaled version
    Y_count = np.exp(
        0.5
        + 0.1 * alpha[units]
        + 0.05 * gamma[time]
        + 0.1 * treat
        + rng.normal(0, 0.3, N * T)
    )
    Y_count = np.maximum(Y_count, 0.1).astype(int)  # Ensure positive integers
    Y_binary = (Y_count > 0).astype(int)
    weights = rng.uniform(0, 1, N * T)
    return pd.DataFrame(
        {
            "unit": units,
            "year": time,
            "treat": treat,
            "ever_treated": ever_treated,
            "Y": Y,
            "Y_count": Y_count,
            "Y_binary": Y_binary,
            "weights": weights,
        }
    )


@pytest.fixture(scope="module")
def data_time():
    N = 200
    rng = np.random.default_rng(9291)
    data = pd.DataFrame(
        {
            "unit": rng.normal(0, 1, N),
            "year": np.arange(N),
            "treat": rng.choice([0, 1], N),
            "weights": rng.uniform(0, 1, N),
        }
    )

    # Generate AR(1) errors with rho=0.8 for strong autocorrelation
    rho = 0.8
    innovations = rng.normal(0, 1, N)
    epsilon = np.empty(N)
    epsilon[0] = innovations[0]
    for t in range(1, N):
        epsilon[t] = rho * epsilon[t - 1] + innovations[t]

    data["Y"] = data["unit"] - data["year"] + 0.5 * data["treat"] + epsilon
    # Add count variable for Poisson models
    data["Y_count"] = np.maximum(
        np.exp(
            0.5
            + 0.05 * data["unit"]
            - 0.01 * data["year"]
            + 0.1 * data["treat"]
            + rng.normal(0, 0.3, N)
        ),
        0.1,
    ).astype(int)
    data["Y_binary"] = (data["Y_count"] > 0).astype(int)
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


def _prepare_balanced_data(data, panel_id, balanced):
    """Prepare data by applying balancing/non-balancing modifications.

    Parameters
    ----------
    data : pd.DataFrame
        The input data frame.
    panel_id : str or None
        The panel ID column name, or None if no panel structure.
    balanced : str
        Type of balancing: "balanced-consecutive", "balanced-non-consecutive",
        "non-balanced-consecutive", or "non-balanced-non-consecutive".

    Returns
    -------
    pd.DataFrame
        Modified data frame with appropriate rows dropped.
    """
    if panel_id is None:
        return data

    first_25 = np.unique(data["unit"])[:25]
    cc = data.groupby("unit").cumcount()

    if balanced == "balanced-non-consecutive":
        # drop an interior row (e.g., the 2nd observation, cc==1) for EVERY unit
        # => all units lose exactly one row (balanced), and there is a gap (non-consecutive)
        data = data[cc != 1].reset_index(drop=True)
    elif balanced == "non-balanced-consecutive":
        # drop the *first* row (cc==0) but only for the first 25 units
        # => those units have T-1 rows (non-balanced), still consecutive (start at the 2nd period)
        mask = ~(data["unit"].isin(first_25) & (cc == 0))
        data = data[mask].reset_index(drop=True)
    elif balanced == "non-balanced-non-consecutive":
        # drop an interior row (e.g., the 3rd observation, cc==2) but only for the first 25 units
        # => those units have a gap (non-consecutive) and fewer rows (non-balanced)
        mask = ~(data["unit"].isin(first_25) & (cc == 2))
        data = data[mask].reset_index(drop=True)

    return data


def _get_r_panel_kwargs(time_id, panel_id, lag, inference):
    """Construct R panel kwargs for vcov_NW or vcov_DK.

    Parameters
    ----------
    time_id : str or None
        The time ID column name.
    panel_id : str or None
        The panel ID column name.
    lag : int or None
        The lag value for HAC.
    inference : str
        Type of inference: "NW" or "DK".

    Returns
    -------
    dict
        Dictionary with kwargs for R's vcov_NW or vcov_DK functions.
    """
    r_panel_kwars = {}
    if time_id is not None:
        r_panel_kwars["time"] = time_id
    if lag is not None:
        r_panel_kwars["lag"] = lag
    if inference == "NW" and panel_id is not None:
        r_panel_kwars["unit"] = panel_id
    return r_panel_kwars


ALL_F3 = ["str", "object", "int", "categorical", "float"]
SINGLE_F3 = ALL_F3[0]
BACKEND_F3 = [
    *[("numba", t) for t in ALL_F3],
    # *[(b, SINGLE_F3) for b in ("jax", "rust")],
]


@pytest.mark.against_r_core
@pytest.mark.parametrize("inference", ["NW", "DK"])
@pytest.mark.parametrize(
    "vcov_kwargs",
    [
        {"lag": 2, "time_id": "year"},
        {"lag": 4, "time_id": "year"},
        # now add panel id
        {"lag": 2, "time_id": "year", "panel_id": "unit"},
        {"lag": 4, "time_id": "year", "panel_id": "unit"},
        # lag not required when panel_id is provided
        {"time_id": "year", "panel_id": "unit"},
    ],
)
@pytest.mark.parametrize(
    "balanced",
    [
        "balanced-consecutive",
        "balanced-non-consecutive",
        "non-balanced-consecutive",
        "non-balanced-non-consecutive",
    ],
)
@pytest.mark.parametrize(
    "weights",
    [None, "weights"],
)
@pytest.mark.parametrize("fml", ols_fmls)
@pytest.mark.parametrize("k_adj", [True, False])
@pytest.mark.parametrize("G_adj", [True, False])
@pytest.mark.parametrize("k_fixef", ["none", "nonnested", "full"])
def test_single_fit_feols_hac_panel(
    data_panel,
    data_time,
    inference,
    vcov_kwargs,
    weights,
    fml,
    balanced,
    k_adj,
    G_adj,
    k_fixef,
):
    lag = vcov_kwargs.get("lag", None)
    time_id = vcov_kwargs.get("time_id", None)
    panel_id = vcov_kwargs.get("panel_id", None)
    data = data_panel if panel_id is not None else data_time

    if panel_id is None and balanced != "balanced-consecutive":
        pytest.skip("Don't test for non-balancedness when no panel data.")

    if panel_id is None and inference == "DK":
        pytest.skip(
            "Don't test for DK when no panel data, as ill-defined / collapes back to TS HAC."
        )

    data = _prepare_balanced_data(data, panel_id, balanced)

    if "|" in fml and panel_id is None:
        pytest.skip("Don't run fixed effect test when data is not a panel.")

    r_panel_kwars = _get_r_panel_kwargs(time_id, panel_id, lag, inference)

    r_fixest = fixest.feols(
        ro.Formula(fml),
        vcov=fixest.vcov_NW(**r_panel_kwars)
        if inference == "NW"
        else fixest.vcov_DK(**r_panel_kwars),
        data=data,
        **({"weights": ro.Formula(f"~{weights}")} if weights is not None else {}),
        panel_time_step=1,
        ssc=fixest.ssc(k_adj, k_fixef, False, G_adj, "min", "min"),
    )

    mod = pf.feols(
        fml=fml,
        data=data,
        vcov=inference,
        vcov_kwargs=vcov_kwargs,
        weights=weights,
        ssc=pf.ssc(k_adj=k_adj, k_fixef=k_fixef, G_adj=G_adj),
    )

    # r_fixest to global r env, needed for
    # operations as in dof.K
    ro.globalenv["r_fixest"] = r_fixest

    py_vcov = mod._vcov[0, 0]
    r_vcov = stats.vcov(r_fixest)[0, 0]

    check_absolute_diff(py_vcov, r_vcov, 1e-05, "py_vcov != r_vcov")


@pytest.mark.against_r_core
@pytest.mark.parametrize("inference", ["NW", "DK"])
@pytest.mark.parametrize(
    "vcov_kwargs",
    [
        {"lag": 2, "time_id": "year"},
        {"lag": 4, "time_id": "year"},
        # now add panel id
        {"lag": 2, "time_id": "year", "panel_id": "unit"},
        {"lag": 4, "time_id": "year", "panel_id": "unit"},
        # lag not required when panel_id is provided
        {"time_id": "year", "panel_id": "unit"},
    ],
)
@pytest.mark.parametrize(
    "balanced",
    [
        "balanced-consecutive",
        # "balanced-non-consecutive",
        # "non-balanced-consecutive",
        # "non-balanced-non-consecutive",
    ],
)
@pytest.mark.parametrize("fml", poisson_fmls)
def test_single_fit_fepois_hac_panel(
    data_panel, data_time, inference, vcov_kwargs, fml, balanced
):
    k_adj = False
    G_adj = False
    ssc_ = ssc(k_adj=k_adj, G_adj=G_adj)

    lag = vcov_kwargs.get("lag", None)
    time_id = vcov_kwargs.get("time_id", None)
    panel_id = vcov_kwargs.get("panel_id", None)
    data = data_panel if panel_id is not None else data_time

    if panel_id is None and balanced != "balanced-consecutive":
        pytest.skip("Don't test for non-balancedness when no panel data.")

    if panel_id is None and inference == "DK":
        pytest.skip(
            "Don't test for DK when no panel data, as ill-defined / collapes back to TS HAC."
        )

    data = _prepare_balanced_data(data, panel_id, balanced)

    if "|" in fml and panel_id is None:
        pytest.skip("Don't run fixed effect test when data is not a panel.")

    r_panel_kwars = _get_r_panel_kwargs(time_id, panel_id, lag, inference)

    r_fixest = fixest.fepois(
        ro.Formula(fml),
        vcov=fixest.vcov_NW(**r_panel_kwars)
        if inference == "NW"
        else fixest.vcov_DK(**r_panel_kwars),
        data=data,
        ssc=fixest.ssc(k_adj, "nested", False, G_adj, "min", "min"),
        panel_time_step=1,
    )

    mod = pf.fepois(
        fml=fml,
        data=data,
        vcov=inference,
        vcov_kwargs=vcov_kwargs,
        ssc=ssc_,
    )

    # r_fixest to global r env, needed for
    # operations as in dof.K
    ro.globalenv["r_fixest"] = r_fixest

    py_vcov = mod._vcov[0, 0]
    r_vcov = stats.vcov(r_fixest)[0, 0]

    check_absolute_diff(py_vcov, r_vcov, 1e-04, "py_vcov != r_vcov")


@pytest.mark.against_r_core
@pytest.mark.parametrize("inference", ["NW", "DK"])
@pytest.mark.parametrize(
    "vcov_kwargs",
    [
        {"lag": 2, "time_id": "year"},
        {"lag": 4, "time_id": "year"},
        # now add panel id
        {"lag": 2, "time_id": "year", "panel_id": "unit"},
        {"lag": 4, "time_id": "year", "panel_id": "unit"},
        # lag not required when panel_id is provided
        {"time_id": "year", "panel_id": "unit"},
    ],
)
@pytest.mark.parametrize(
    "balanced",
    [
        "balanced-consecutive",
        # "balanced-non-consecutive",
        # "non-balanced-consecutive",
        # "non-balanced-non-consecutive",
    ],
)
@pytest.mark.parametrize("fml", glm_fmls)
def test_single_fit_feglm_hac_panel(
    data_panel, data_time, inference, vcov_kwargs, fml, balanced
):
    k_adj = False
    G_adj = False
    ssc_ = ssc(k_adj=k_adj, G_adj=G_adj)

    lag = vcov_kwargs.get("lag", None)
    time_id = vcov_kwargs.get("time_id", None)
    panel_id = vcov_kwargs.get("panel_id", None)
    data = data_panel if panel_id is not None else data_time
    # panel_time_step = None

    if panel_id is None and balanced != "balanced-consecutive":
        pytest.skip("Don't test for non-balancedness when no panel data.")

    if panel_id is None and inference == "DK":
        pytest.skip(
            "Don't test for DK when no panel data, as ill-defined / collapes back to TS HAC."
        )

    if panel_id is not None:
        # pick the subset of units to alter for the non-balanced cases
        first_25 = np.unique(data["unit"])[:25]
        cc = data.groupby("unit").cumcount()

        if balanced == "balanced-non-consecutive":
            # drop an interior row (e.g., the 2nd observation, cc==1) for EVERY unit
            # => all units lose exactly one row (balanced), and there is a gap (non-consecutive)
            data = data[cc != 1].reset_index(drop=True)
            # panel_time_step = "unitary"
        elif balanced == "non-balanced-consecutive":
            # drop the *first* row (cc==0) but only for the first 25 units
            # => those units have T-1 rows (non-balanced), still consecutive (start at the 2nd period)
            mask = ~(data["unit"].isin(first_25) & (cc == 0))
            data = data[mask].reset_index(drop=True)
        elif balanced == "non-balanced-non-consecutive":
            # drop an interior row (e.g., the 3rd observation, cc==2) but only for the first 25 units
            # => those units have a gap (non-consecutive) and fewer rows (non-balanced)
            mask = ~(data["unit"].isin(first_25) & (cc == 2))
            data = data[mask].reset_index(drop=True)
            # panel_time_step = "unitary"
        else:
            pass

    if "|" in fml and panel_id is None:
        pytest.skip("Don't run fixed effect test when data is not a panel.")

    r_panel_kwars = ({"time": time_id} if time_id is not None else {}) | (
        {"lag": lag} if lag is not None else {}
    )

    if inference == "NW":
        r_panel_kwars |= {"unit": panel_id} if panel_id is not None else {}

    r_fixest = fixest.feglm(
        ro.Formula(fml),
        vcov=fixest.vcov_NW(**r_panel_kwars)
        if inference == "NW"
        else fixest.vcov_DK(**r_panel_kwars),
        data=data,
        ssc=fixest.ssc(k_adj, "nested", False, G_adj, "min", "min"),
        family=stats.binomial(link="logit"),
        panel_time_step=1,
    )

    mod = pf.feglm(
        fml=fml,
        data=data,
        vcov=inference,
        vcov_kwargs=vcov_kwargs,
        ssc=ssc_,
        family="logit",
    )
    # r_fixest to global r env, needed for
    # operations as in dof.K
    ro.globalenv["r_fixest"] = r_fixest

    py_coef = mod.coef().xs("treat")
    r_coef = stats.coef(r_fixest)[1]
    check_absolute_diff(py_coef, r_coef, 1e-06, "py_coef != r_coef")

    py_vcov = mod._vcov[1, 1]
    r_vcov = stats.vcov(r_fixest)[1, 1]

    check_absolute_diff(py_vcov, r_vcov, 1e-04, "py_vcov != r_vcov")


@pytest.mark.against_r_core
def test_vcov_updating(data_panel):
    fit_hetero = pf.feols("Y ~ treat", data=data_panel, vcov="hetero")
    fit_nw = pf.feols(
        "Y ~ treat",
        data=data_panel,
        vcov="NW",
        vcov_kwargs={"time_id": "year", "panel_id": "unit", "lag": 7},
    )

    fit_hetero.vcov(
        vcov="NW", vcov_kwargs={"lag": 7, "time_id": "year", "panel_id": "unit"}
    )

    assert fit_hetero._vcov_type == "HAC"
    assert fit_hetero._vcov_type_detail == "NW"
    check_absolute_diff(fit_hetero._vcov, fit_nw._vcov, 1e-08, "py_vcov != r_vcov")
