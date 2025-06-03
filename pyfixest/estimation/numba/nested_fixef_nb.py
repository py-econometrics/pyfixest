import numba as nb
import numpy as np


@nb.njit(parallel=True)
def _count_fixef_fully_nested_all(
    all_fixef_array: np.ndarray,
    cluster_colnames: np.ndarray,
    cluster_data: np.ndarray,
    fe_data: np.ndarray,
) -> tuple[np.ndarray, int]:
    """

    Compute the number of nested fixed effects over all fixed effects.

    Parameters
    ----------
    all_fixef_array : np.ndarray
        A 1D array with the names of all fixed effects in the model.
    cluster_colnames : np.ndarray
        A 1D array with the names of all cluster variables in the model.
    cluster_data : np.ndarray
        A 2D array with the cluster data.
    fe_data : np.ndarray
        A 2D array with the fixed effects.

    Returns
    -------
    k_fe_nested : np.ndarray
        A numpy array with shape (all_fixef_array.size, ) containing boolean values that
        indicate whether a given fixed effect is fully nested within a cluster or not.
    n_fe_fully_nested : int
        The number of fixed effects that are fully nested within a clusters.
    """
    k_fe_nested_flag = np.zeros(all_fixef_array.size, dtype=np.bool_)
    n_fe_fully_nested = 0

    for fi in nb.prange(all_fixef_array.size):
        this_fe_name = all_fixef_array[fi]

        found_in_cluster = False
        for col_i in range(cluster_colnames.size):
            if this_fe_name == cluster_colnames[col_i]:
                found_in_cluster = True
                k_fe_nested_flag[fi] = True
                n_fe_fully_nested += 1
                break

        if not found_in_cluster:
            for col_j in range(cluster_colnames.size):
                clusters_col = cluster_data[:, col_j]
                fe_col = fe_data[:, fi]
                is_fully_nested = _count_fixef_fully_nested(clusters_col, fe_col)
                if is_fully_nested:
                    k_fe_nested_flag[fi] = True
                    n_fe_fully_nested += 1
                    break

    return k_fe_nested_flag, n_fe_fully_nested


@nb.njit
def _count_fixef_fully_nested(clusters: np.ndarray, f: np.ndarray) -> bool:
    """
    Check if a given fixed effect is fully nested within a given cluster.

    Parameters
    ----------
    clusters : np.ndarray
        A vector of cluster assignments.
    f : np.ndarray
        A matrix of fixed effects.

    Returns
    -------
    np.array(np.bool_)
        An array of booleans indicating whether each fixed effect is fully nested within clusters.
        True if the fixed effect is fully nested within clusters, False otherwise.
    """
    unique_vals = np.unique(f)
    n_unique_vals = len(unique_vals)
    counts = 0
    for val in unique_vals:
        mask = f == val
        distinct_clusters = np.unique(clusters[mask])
        if len(distinct_clusters) == 1:
            counts += 1
    is_fe_nested = counts == n_unique_vals

    return is_fe_nested
