import numpy as np

def ssc(adj=True, fixef_k="none", cluster_adj=True, cluster_df="conventional"):
    '''
    Set the small sample correction factor applied in `get_ssc()`
    Parameters:
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
    Returns:
        A dictionary with encoded info on how to form small sample corrections
    '''

    if adj not in [True, False]:
        raise ValueError("adj must be True or False.")
    if fixef_k not in ["none"]:
        raise ValueError("fixef_k must be 'none'.")
    if cluster_adj not in [True, False]:
        raise ValueError("cluster_adj must be True or False.")
    if cluster_df not in ["conventional", "min"]:
        raise ValueError("cluster_df must be 'conventional' or 'min'.")

    res = {'adj': adj, 'fixef_k': fixef_k,
           'cluster_adj': cluster_adj, 'cluster_df': cluster_df}

    return res


def get_ssc(ssc_dict, N, k, G, vcov_sign, vcov_type):
    """
    Compute small sample adjustment factors

    Args:
    - ssc_dict: An dictionariy created via the ssc() function
    - N: The number of observations
    - k: The number of estimated parameters
    - G: The number of clusters
    - vcov_sign: A vector that helps create the covariance matrix
    - vcov_type: Either "iid", "hetero" or "CRV"

    Returns:
    - A small sample adjustment factor
    """

    adj = ssc_dict['adj']
    fixef_k = ssc_dict['fixef_k']
    cluster_adj = ssc_dict['cluster_adj']
    cluster_df = ssc_dict['cluster_df']

    print("adj", adj)

    cluster_adj_value = 1
    adj_value = 1

    if vcov_type == "hetero":
        if adj:
            adj_value = N / (N-k)
        else:
            adj_value = N / (N-1)
    elif vcov_type in ["iid", "CRV"]:
        if adj:
            adj_value = (N - 1) / (N - k)
        else:
            adj_value = 1

    if vcov_type == "CRV":

        if cluster_adj:
            if cluster_df == 'conventional':
                cluster_adj_value = G / (G - 1)
            elif cluster_df == "min":
                G = np.min(G)
                cluster_adj_value = G / (G - 1)
            else:
                raise ValueError("cluster_df is neither conventional nor min.")

    print("ssc", adj_value * cluster_adj_value * vcov_sign)

    return adj_value * cluster_adj_value * vcov_sign
