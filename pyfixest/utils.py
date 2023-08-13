import numpy as np
import pandas as pd
from formulaic import model_matrix


def get_data(N=1000, seed=1234, beta_type="1", error_type="1", model = "Feols"):

    """
    create a random example data set
    Args:
        N: number of observations
        seed: seed for the random number generator
        beta_type: type of beta coefficients
        error_type: type of error term
        model: type of the dgp. Either "Feols" or "Fepois"
    Returns:
        df: a pandas data frame with simulated data
    """

    rng = np.random.default_rng(seed)
    G = rng.choice(list(range(10, 100))).astype("int64")
    fe_dims = rng.choice(list(range(2, int(np.floor(np.sqrt(N))))), 3, True).astype(
        "int64"
    )

    # create the covariates
    X = rng.normal(0, 1, N * 5).reshape((N, 5))
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

    Y, Y2 = [pd.Series(x.flatten()) for x in [Y, Y2]]
    Y.name, Y2.name = "Y", "Y2"

    cluster = rng.choice(list(range(0, G)), N)
    cluster = pd.Series(cluster)
    cluster.name = "group_id"

    df = pd.concat([Y, Y2, X, cluster], axis=1)

    # add some NaN values
    df.loc[0, "Y"] = np.nan
    df.loc[1, "X1"] = np.nan
    df.loc[2, "f1"] = np.nan

    # compute some instruments
    df["Z1"] = df["X1"] + np.random.normal(0, 1, N)
    df["Z2"] = df["X2"] + np.random.normal(0, 1, N)

    # change all variables in the data frame to float
    for col in df.columns:
        df[col] = df[col].astype("float64")

    df[df == "nan"] = np.nan

    # add separation (simple to detect)
    if model == "Fepois":
        where_zeros = np.where(df["Y"] == 0)[0]     # because np.where evaluates to a tuple
        # draw three random indices
        idx = rng.choice(where_zeros, 3, True)
        df.loc[idx[0], "f1"] = np.max(df["f1"]) + 1
        df.loc[idx[1], "f2"] = np.max(df["f2"]) + 1
        df.loc[idx[2], "f3"] = np.max(df["f3"]) + 1

    return df


def get_poisson_data(N=1000, seed=4320):
    """
    Generate data following a Poisson regression dgp.
    Args:
        N: number of observations
        seed: seed for the random number generator
    Returns:
        data: a pandas data frame
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

    data = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3, "X4": X4})

    return data


def absolute_diff(x, y, tol=1e-03):
    absolute_diff = (np.abs(x - y) > tol).any()
    if not any(y == 0):
        relative_diff = (np.abs(x - y) / np.abs(y) > tol).any()
        res = absolute_diff and relative_diff
    else:
        res = absolute_diff

    return res
