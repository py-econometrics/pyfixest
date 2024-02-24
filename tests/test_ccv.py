import numpy as np
import pandas as pd


# function retrieved from Harvard Dataverse
def compute_CCV(df, depvar, cluster, seed, nmx, pk):
    """ @return: CCV variance for the case qk==1
        @param df dataframe with columns
            - 'u' indicator for random split (Z in the paper)
            - nmx name of the binary indicator W
            - cluster name of the cluster indicator
            - depvar outcome variable
        @param u binary variable indicating which split to use for computing the model and which for evaluation
        @param nmx string with the name of the treatment indicator W
        @param pk double in (0,1] as described in the paper
    """

    rng = np.random.default_rng(seed)
    df["u"] = rng.choice([False, True], size=df.shape[0])

    u1 = df['u']==True
    u0 = df['u']==False
    w1 = df[nmx]==1
    w0 = df[nmx]==0

    # compute alpha, tau using first split
    alpha = df[u1 & w0][depvar].mean()
    tau = df[u1 & w1][depvar].mean() - df[u1 & w0][depvar].mean()
    tau_full = df[w1][depvar].mean() - df[w0][depvar].mean()

    # compute for each m
    tau_ms = {}
    nm = 'tau_'
    pk_term = 0
    for m in df[cluster].unique():
        ind_m = df[cluster]==m
        aux1 = df[u1 & ind_m & w1][depvar].mean()
        aux0 =  df[u1 & ind_m & w0][depvar].mean()

        aux1_full = df[ind_m & w1][depvar].mean()
        aux0_full =  df[ind_m & w0][depvar].mean()
        aux_tau = aux1 - aux0
        aux_tau_full = aux1_full - aux0_full

        if (np.isnan(aux1)) or (np.isnan(aux0)):
            aux_tau = tau

        aux_nm = nm + str(m)
        tau_ms[aux_nm] = aux_tau

        # compute the pk term in u0
        Nm = df[ind_m].shape[0]
        aux_pk = Nm*((aux_tau_full-tau)**2)
        pk_term = pk_term + aux_pk

    # compute the residuals
    df['resU'] = df[depvar] - alpha - df[nmx]*tau

    # Wbar
    Wbar = df[u1][nmx].mean()
    #Wbar = df[nmx].mean() # to match Guido

    # pk term
    pk_term = pk_term*(1.-pk)/df.shape[0]


    # compute avg Z
    Zavg = (np.sum(df['u']==False))/df.shape[0]

    # compute the normalized CCV using second split
    n = (df.shape[0]*(Wbar**2)*((1.-Wbar)**2))

    sum_CCV = 0
    for m in df[cluster].unique():
        ind_m = df[cluster]==m
        df_m = df[u0 & ind_m]
        aux_nm = nm + str(m)

        # tau term
        tau_term = (tau_ms[aux_nm] - tau)*Wbar*(1.-Wbar)

        # Residual term
        res_term = (df_m[nmx] - Wbar)*df_m['resU']

        # square of sums
        sq_sum = np.sum(res_term - tau_term)**2

        # sum of squares
        sum_sq = np.sum((res_term - tau_term)**2)

        # compute CCV
        sum_CCV += (1./(Zavg**2))*sq_sum - ((1.-Zavg)/(Zavg**2))*sum_sq + n*pk_term

    # normalize
    V_CCV = sum_CCV / n
    print("V_CCV: ", V_CCV)

    return V_CCV

def test_ccv():

