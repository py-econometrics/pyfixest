import numpy as np
import pandas as pd
from pyfixest.estimation import feols
import pytest

from pyfixest.ccv import _compute_CCV, ccv


# function retrieved from Harvard Dataverse
def compute_CCV_AAIW(df, depvar, cluster, seed, nmx, pk):
    """
    Compute the CCV variance using a slight variation of AAIW's code.

    The code is based on a Python implementation of the authours published under CC0 1.0 Deed and available at
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/27VMOT. This function has also
    benefitted from Daniel Pailanir and Damian Clarke's implementation in Stata available at
    https://github.com/Daniel-Pailanir/TSCB-CCV and published under GPL3 License.

    """

    rng = np.random.default_rng(seed)
    df["u"] = rng.choice([False, True], size=df.shape[0])

    u1 = df["u"] == True
    u0 = df["u"] == False
    w1 = df[nmx] == 1
    w0 = df[nmx] == 0

    # compute alpha, tau using first split
    alpha = df[u1 & w0][depvar].mean()
    tau = df[u1 & w1][depvar].mean() - df[u1 & w0][depvar].mean()
    tau_full = df[w1][depvar].mean() - df[w0][depvar].mean()

    # compute for each m
    tau_ms = {}
    nm = "tau_"
    pk_term = 0
    for m in df[cluster].unique():
        ind_m = df[cluster] == m
        aux1 = df[u1 & ind_m & w1][depvar].mean()
        aux0 = df[u1 & ind_m & w0][depvar].mean()

        aux1_full = df[ind_m & w1][depvar].mean()
        aux0_full = df[ind_m & w0][depvar].mean()
        aux_tau = aux1 - aux0
        aux_tau_full = aux1_full - aux0_full

        if (np.isnan(aux1)) or (np.isnan(aux0)):
            aux_tau = tau

        aux_nm = nm + str(m)
        tau_ms[aux_nm] = aux_tau

        # compute the pk term in u0
        Nm = df[ind_m].shape[0]
        aux_pk = Nm * ((aux_tau_full - tau) ** 2)
        pk_term = pk_term + aux_pk

    # compute the residuals
    df["resU"] = df[depvar] - alpha - df[nmx] * tau

    # Wbar
    Wbar = df[u1][nmx].mean()
    # Wbar = df[nmx].mean() # to match Guido

    # pk term
    pk_term = pk_term * (1.0 - pk) / df.shape[0]

    # compute avg Z
    Zavg = (np.sum(df["u"] == False)) / df.shape[0]

    # compute the normalized CCV using second split
    n = df.shape[0] * (Wbar**2) * ((1.0 - Wbar) ** 2)

    sum_CCV = 0
    for m in df[cluster].unique():
        ind_m = df[cluster] == m
        df_m = df[u0 & ind_m]
        aux_nm = nm + str(m)

        # tau term
        tau_term = (tau_ms[aux_nm] - tau) * Wbar * (1.0 - Wbar)

        # Residual term
        res_term = (df_m[nmx] - Wbar) * df_m["resU"]

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


@pytest.mark.skip(reason="This test is not yet implemented")
def test_ccv_against_AAIW():

    df = pd.read_stata("C:/Users/alexa/Downloads/census2000_5pc.dta")
    N = df.shape[0]
    Y = df["ln_earnings"].values
    W = df["college"].values.reshape(-1, 1)
    X = np.concatenate([np.ones((N, 1)), W], axis=1)
    cluster_vec = df["state"].values
    rng = np.random.default_rng(2002)

    fml = "ln_earnings ~ college"

    vcov_AAIW = compute_CCV_AAIW(
        df, depvar="ln_earnings", cluster="state", seed=2002, nmx="college", pk=0.05
    )
    vcov = _compute_CCV(
        fml=fml,
        X=X,
        Y=Y,
        W=W,
        treatment="college",
        cluster_vec=cluster_vec,
        pk=0.05,
        rng=rng,
        data=df,
    )

    assert vcov_AAIW == vcov


def test_against_stata():
    """
    Test the ccv function against the stata implementation of the CCV variance.

    The test values are taken from the readme example of the stata implementation of the CCV variance
    and can be found here: https://github.com/Daniel-Pailanir/TSCB-CCV.
    """

    data = pd.read_stata("C:/Users/alexa/Downloads/census2000_5pc.dta")

    fit = feols("ln_earnings ~ college", data=data, vcov = {"CRV1": "state"})

    res_ccv1 = fit.ccv(treatment = "college", pk = 0.05, qk = 1, n_splits = 8, seed = 929).xs("CCV")
    res_ccv2 = fit.ccv(treatment = "college", pk = 0.5, qk = 1, n_splits = 8, seed = 929).xs("CCV")
    res_ccv3 = fit.ccv(treatment = "college", pk = 1, qk = 0.05, n_splits = 8, seed = 929).xs("CCV")
    res_ccv4 = fit.ccv(treatment = "college", pk = 1, qk = 0.5, n_splits = 8, seed = 929).xs("CCV")

    assert np.abs(res_ccv1["2.5 %"] - 0.458) < 1e-02
    assert np.abs(res_ccv1["97.5 %"] - 0.473) < 1e-02

    assert np.abs(res_ccv2["2.5 %"] - 0.458) < 1e-02
    assert np.abs(res_ccv2["97.5 %"] - 0.473) < 1e-02

    assert np.abs(res_ccv3["2.5 %"] - 0.414) < 1e-02
    assert np.abs(res_ccv3["97.5 %"] - 0.517) < 1e-02

    assert np.abs(res_ccv4["2.5 %"] - 0.428) < 1e-02
    assert np.abs(res_ccv4["97.5 %"] - 0.503) < 1e-02
