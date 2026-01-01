import numpy as np
import pandas as pd
import pytest

from pyfixest.estimation.ccv import _compute_CCV
from pyfixest.estimation import feols


@pytest.fixture
def data(local=False):
    """Load the census data used in the tests."""
    if local:
        return pd.read_stata("C:/Users/alexa/Downloads/census2000_5pc.dta")
    else:
        return pd.read_stata("http://www.damianclarke.net/stata/census2000_5pc.dta")


# function retrieved from Harvard Dataverse
@pytest.mark.extended
def compute_CCV_AAIW(data, depvar, cluster, seed, nmx, pk):
    """
    Compute the CCV variance using a slight variation of AAIW's code.

    The code is based on a Python implementation of the authours
    published under CC0 1.0 Deed and available at
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/27VMOT.
    This function has also benefitted from Daniel Pailanir
    and Damian Clarke's implementation in Stata available at
    https://github.com/Daniel-Pailanir/TSCB-CCV and published under GPL3 License.

    """
    rng = np.random.default_rng(seed)
    data["u"] = rng.choice([False, True], size=data.shape[0])

    u1 = data["u"] == True  # noqa: E712
    u0 = data["u"] == False  # noqa: E712
    w1 = data[nmx] == 1
    w0 = data[nmx] == 0

    # compute alpha, tau using first split
    alpha = data[u1 & w0][depvar].mean()
    tau = data[u1 & w1][depvar].mean() - data[u1 & w0][depvar].mean()

    # compute for each m
    tau_ms = {}
    nm = "tau_"
    pk_term = 0
    for m in data[cluster].unique():
        ind_m = data[cluster] == m
        aux1 = data[u1 & ind_m & w1][depvar].mean()
        aux0 = data[u1 & ind_m & w0][depvar].mean()

        aux1_full = data[ind_m & w1][depvar].mean()
        aux0_full = data[ind_m & w0][depvar].mean()
        aux_tau = aux1 - aux0
        aux_tau_full = aux1_full - aux0_full

        if (np.isnan(aux1)) or (np.isnan(aux0)):
            aux_tau = tau

        aux_nm = nm + str(m)
        tau_ms[aux_nm] = aux_tau

        # compute the pk term in u0
        Nm = data[ind_m].shape[0]
        aux_pk = Nm * ((aux_tau_full - tau) ** 2)
        pk_term = pk_term + aux_pk

    # compute the residuals
    data["resU"] = data[depvar] - alpha - data[nmx] * tau

    # Wbar
    Wbar = data[u1][nmx].mean()
    # Wbar = data[nmx].mean() # to match Guido

    # pk term
    pk_term = pk_term * (1.0 - pk) / data.shape[0]

    # compute avg Z
    Zavg = (np.sum(data["u"] == False)) / data.shape[0]  # noqa: E712

    # compute the normalized CCV using second split
    n = data.shape[0] * (Wbar**2) * ((1.0 - Wbar) ** 2)

    sum_CCV = 0
    for m in data[cluster].unique():
        ind_m = data[cluster] == m
        data_m = data[u0 & ind_m]
        aux_nm = nm + str(m)

        # tau term
        tau_term = (tau_ms[aux_nm] - tau) * Wbar * (1.0 - Wbar)

        # Residual term
        res_term = (data_m[nmx] - Wbar) * data_m["resU"]

        # square of sums
        sq_sum = np.sum(res_term - tau_term) ** 2

        # sum of squares
        sum_sq = np.sum((res_term - tau_term) ** 2)

        # compute CCV
        sum_CCV += (
            (1.0 / (Zavg**2)) * sq_sum
            - ((1.0 - Zavg) / (Zavg**2)) * sum_sq
            + n * pk_term
        )

    # normalize
    V_CCV = sum_CCV / n
    print("V_CCV: ", V_CCV)

    return V_CCV


@pytest.mark.parametrize("pk", [0.05, 0.5, 0.95])
@pytest.mark.extended
def test_ccv_against_AAIW(data, pk):
    N = data.shape[0]
    Y = data["ln_earnings"].values
    W = data["college"].values
    X = np.concatenate([np.ones((N, 1)), W.reshape(-1, 1)], axis=1)
    cluster_vec = data["state"].values
    seed = 2002

    rng = np.random.default_rng(seed)

    fml = "ln_earnings ~ college"
    tau_full = feols(fml, data=data).coef().xs("college")

    vcov_AAIW = compute_CCV_AAIW(
        data, depvar="ln_earnings", cluster="state", seed=seed, nmx="college", pk=pk
    )
    vcov = _compute_CCV(
        fml=fml,
        X=X,
        Y=Y,
        W=W,
        treatment="college",
        cluster_vec=cluster_vec,
        pk=pk,
        rng=rng,
        data=data,
        tau_full=tau_full,
    )

    assert np.abs(vcov - vcov_AAIW) < 1e-6


@pytest.mark.extended
def test_ccv_internally(data):
    """Test the ccv function internally."""
    # it does not matter where CRV inference is specified
    fit1 = feols("ln_earnings ~ college", data=data)
    fit2 = feols("ln_earnings ~ college", data=data, vcov={"CRV1": "state"})

    res1 = fit1.ccv(
        treatment="college", pk=0.05, qk=1, n_splits=2, seed=929, cluster="state"
    )
    res2 = fit2.ccv(treatment="college", pk=0.05, qk=1, n_splits=2, seed=929)

    assert np.all(res1 == res2)


@pytest.mark.extended
def test_against_stata(data):
    """
    Test the ccv function against the stata implementation of the CCV variance.

    The test values are taken from the readme example of the stata
    implementation of the CCV variance
    and can be found here: https://github.com/Daniel-Pailanir/TSCB-CCV.
    """
    # this can take a while

    fit = feols("ln_earnings ~ college", data=data, vcov={"CRV1": "state"})

    res_ccv1 = fit.ccv(treatment="college", pk=0.05, qk=1, n_splits=4, seed=929).xs(
        "CCV"
    )
    res_ccv2 = fit.ccv(treatment="college", pk=0.5, qk=1, n_splits=4, seed=929).xs(
        "CCV"
    )
    res_ccv3 = fit.ccv(treatment="college", pk=1, qk=0.05, n_splits=4, seed=929).xs(
        "CCV"
    )
    res_ccv4 = fit.ccv(treatment="college", pk=1, qk=0.5, n_splits=4, seed=929).xs(
        "CCV"
    )

    assert np.abs(res_ccv1["2.5%"] - 0.458) < 1e-02
    assert np.abs(res_ccv1["97.5%"] - 0.473) < 1e-02

    assert np.abs(res_ccv2["2.5%"] - 0.458) < 1e-02
    assert np.abs(res_ccv2["97.5%"] - 0.473) < 1e-02

    assert np.abs(res_ccv3["2.5%"] - 0.414) < 1e-02
    assert np.abs(res_ccv3["97.5%"] - 0.517) < 1e-02

    assert np.abs(res_ccv4["2.5%"] - 0.428) < 1e-02
    assert np.abs(res_ccv4["97.5%"] - 0.503) < 1e-02
