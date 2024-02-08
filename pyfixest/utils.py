
import numpy as np
import pandas as pd
from formulaic import model_matrix


def ssc(adj=True, fixef_k="none", cluster_adj=True, cluster_df="min"):
    """
    Set the small sample correction factor applied in `get_ssc()`.

    Parameters
    ----------
        adj: bool, default True
            If True, applies a small sample correction of (N-1) / (N-k) where N is the number of observations
            and k is the number of estimated coefficients excluding any fixed effects projected out in either fixest::feols() or lfe::felm().
        fixef_k: str, default "none"
            Equal to 'none': the fixed effects parameters are discarded when calculating k in (N-1) / (N-k).
        cluster_adj: bool, default True
            If True, a cluster correction G/(G-1) is performed, with G the number of clusters.
        cluster_df: str, default "conventional"
            Controls how "G" is computed for multiway clustering if cluster_adj = True.
            Note that the covariance matrix in the multiway clustering case is of
            the form V = V_1 + V_2 - V_12. If "conventional", then each summand G_i
            is multiplied with a small sample adjustment G_i / (G_i - 1). If "min",
            all summands are multiplied with the same value, min(G) / (min(G) - 1)

    Returns
    -------
        A dictionary with encoded info on how to form small sample corrections
    """
    if adj not in [True, False]:
        raise ValueError("adj must be True or False.")
    if fixef_k not in ["none"]:
        raise ValueError("fixef_k must be 'none'.")
    if cluster_adj not in [True, False]:
        raise ValueError("cluster_adj must be True or False.")
    if cluster_df not in ["conventional", "min"]:
        raise ValueError("cluster_df must be 'conventional' or 'min'.")

    return {
        "adj": adj,
        "fixef_k": fixef_k,
        "cluster_adj": cluster_adj,
        "cluster_df": cluster_df,
    }


def get_ssc(ssc_dict, N, k, G, vcov_sign, vcov_type, is_twoway=False):
    """
    Compute small sample adjustment factors.

    Parameters
    ----------
    ssc_dict : dict
        A dictionary created via the ssc() function.
    N : int
        The number of observations.
    k : int
        The number of estimated parameters.
    G : int
        The number of clusters.
    vcov_sign : array-like
        A vector that helps create the covariance matrix.
    vcov_type : str
        The type of covariance matrix. Must be one of "iid", "hetero", or "CRV".
    is_twoway : bool, optional
        Whether the covariance matrix is of the form V = V_1 + V_2 - V_12. Default is False.

    Returns
    -------
    float
        A small sample adjustment factor.

    Raises
    ------
    ValueError
        If vcov_type is not "iid", "hetero", or "CRV", or if cluster_df is neither "conventional" nor "min".
    """
    adj = ssc_dict["adj"]
    fixef_k = ssc_dict["fixef_k"]  # noqa: F841 TODO: is this used?
    cluster_adj = ssc_dict["cluster_adj"]
    cluster_df = ssc_dict["cluster_df"]

    cluster_adj_value = 1
    adj_value = 1

    if vcov_type == "hetero":
        adj_value = N / (N - k) if adj else N / (N - 1)
    elif vcov_type in ["iid", "CRV"]:
        if adj:
            adj_value = (N - 1) / (N - k)
    else:
        raise ValueError("vcov_type must be either iid, hetero or CRV.")

    if vcov_type == "CRV" and cluster_adj:
        if cluster_df == "conventional":
            cluster_adj_value = G / (G - 1)
        elif cluster_df == "min":
            G = np.min(G)
            cluster_adj_value = G / (G - 1)
        else:
            raise ValueError("cluster_df is neither conventional nor min.")

    return adj_value * cluster_adj_value * vcov_sign


def get_data(N=1000, seed=1234, beta_type="1", error_type="1", model="Feols"):
    """
    Create a random example data set.

    Parameters
    ----------
    N : int, optional
        Number of observations. Default is 1000.
    seed : int, optional
        Seed for the random number generator. Default is 1234.
    beta_type : str, optional
        Type of beta coefficients. Must be one of '1', '2', or '3'. Default is '1'.
    error_type : str, optional
        Type of error term. Must be one of '1', '2', or '3'. Default is '1'.
    model : str, optional
        Type of the DGP. Must be either 'Feols' or 'Fepois'. Default is 'Feols'.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame with simulated data.

    Raises
    ------
    ValueError
        If beta_type is not '1', '2', or '3', or if error_type is not '1', '2', or '3',
        or if model is not 'Feols' or 'Fepois'.
    """
    rng = np.random.default_rng(seed)
    G = rng.choice(list(range(10, 20))).astype("int64")
    fe_dims = rng.choice(list(range(2, int(np.floor(np.sqrt(N))))), 3, True).astype(
        "int64"
    )

    # create the covariates
    X = rng.normal(0, 3, N * 5).reshape((N, 5))
    X[:, 0] = rng.choice(range(3), N, True)
    # X = pd.DataFrame(X)
    X[:, 2] = rng.choice(list(range(fe_dims[0])), N, True)
    X[:, 3] = rng.choice(list(range(fe_dims[1])), N, True)
    X[:, 4] = rng.choice(list(range(fe_dims[2])), N, True)

    X = pd.DataFrame(X)
    X.columns = ["X1", "X2", "f1", "f2", "f3"]
    # X1, X2, X3 as pd.Categorical
    X["f1"] = X["f1"].astype("category")
    X["f2"] = X["f2"].astype("category")
    X["f3"] = X["f3"].astype("category")

    mm = model_matrix("~ X1 + X2 + f1 + f2 + f3", data=X)

    k = mm.shape[1]

    # create the coefficients
    if beta_type == "1":
        beta = rng.normal(0, 1, k).reshape(k, 1)
    elif beta_type == "2":
        beta = rng.normal(0, 5, k).reshape(k, 1)
    elif beta_type == "3":
        beta = np.exp(rng.normal(0, 1, k)).reshape(k, 1)
    else:
        raise ValueError("beta_type needs to be '1', '2' or '3'.")

    # create the error term
    if error_type == "1":
        u = rng.normal(0, 1, N).reshape(N, 1)
    elif error_type == "2":
        u = rng.normal(0, 5, N).reshape(N, 1)
    elif error_type == "3":
        u = np.exp(rng.normal(0, 1, N)).reshape(N, 1)
    else:
        raise ValueError("error_type needs to be '1', '2' or '3'.")

    # create the depvar and cluster variable
    if model == "Feols":
        Y = (1 + mm.to_numpy() @ beta + u).flatten()
        Y2 = Y + rng.normal(0, 5, N)
    elif model == "Fepois":
        mu = np.exp(mm.to_numpy() @ beta).flatten()
        mu = 1 + mu / np.sum(mu)
        Y = rng.poisson(mu, N)
        Y2 = Y + rng.choice(range(10), N, True)
    else:
        raise ValueError("model needs to be 'Feols' or 'Fepois'.")

    Y, Y2 = (pd.Series(x.flatten()) for x in [Y, Y2])
    Y.name, Y2.name = "Y", "Y2"

    cluster = rng.choice(list(range(G)), N)
    cluster = pd.Series(cluster)
    cluster.name = "group_id"

    df = pd.concat([Y, Y2, X, cluster], axis=1)

    # add some NaN values
    df.loc[0, "Y"] = np.nan
    df.loc[1, "X1"] = np.nan
    df.loc[2, "f1"] = np.nan

    # compute some instruments
    df["Z1"] = df["X1"] + rng.normal(0, 1, N)
    df["Z2"] = df["X2"] + rng.normal(0, 1, N)

    # change all variables in the data frame to float
    for col in df.columns:
        df[col] = df[col].astype("float64")

    df[df == "nan"] = np.nan

    df["weights"] = rng.uniform(0, 1, N)
    # df["weights"].iloc[]

    return df


def get_poisson_data(N=1000, seed=4320):
    """
    Generate data following a Poisson regression DGP.

    Parameters
    ----------
    N : int, optional
        Number of observations. Default is 1000.
    seed : int, optional
        Seed for the random number generator. Default is 4320.

    Returns
    -------
    pandas.DataFrame
        Generated data with columns 'Y', 'X1', 'X2', 'X3', and 'X4'.
    """
    # create data
    np.random.seed(seed)
    X1 = np.random.normal(0, 1, N)
    X2 = np.random.choice([0, 1], N, True)
    X3 = np.random.choice([0, 1, 2, 3, 4, 5, 6], N, True)
    X4 = np.random.choice([0, 1], N, True)
    beta = np.array([1, 0, 1, 0])
    u = np.random.normal(0, 1, N)
    mu = np.exp(1 + X1 * beta[0] + X2 * beta[1] + X3 * beta[2] + X4 * beta[3] + u)

    Y = np.random.poisson(mu, N)

    return pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3, "X4": X4})


def absolute_diff(x, y, tol=1e-03):
    """

    Calculate the absolute difference between two values.

    Parameters
    ----------
    x : numpy.ndarray
        Numeric array representing the reference value.
    y : numpy.ndarray
        Numeric array representing the value to compare against.
    tol : float, optional
        Tolerance value used to determine if two values are different. Default is 1e-03.

    Returns
    -------
    bool
        True if the absolute difference is greater than the tolerance and there is a non-zero value in y, False otherwise.
    """
    absolute_diff = (np.abs(x - y) > tol).any()
    if not any(y == 0):
        relative_diff = (np.abs(x - y) / np.abs(y) > tol).any()
        return absolute_diff and relative_diff
    else:
        return absolute_diff
