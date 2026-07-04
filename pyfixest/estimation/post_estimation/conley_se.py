"""
Conley (1999) Spatial HAC Standard Errors.

Computes heteroskedasticity and spatial autocorrelation consistent (spatial HAC)
variance-covariance matrices using a distance-based kernel.

Reference:
    Conley, T.G. (1999): "GMM Estimation with Cross Sectional Dependence,"
    Journal of Econometrics, 92, 1-45.
"""

from __future__ import annotations

import numpy as np

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# --- Haversine distance (vectorized for numpy, jitted for numba) ---

def _haversine_scalar(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km between two points given in degrees."""
    R = 6371.0  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _conley_meat_numpy(
    scores: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    cutoff: float,
    kernel: str = "bartlett",
) -> np.ndarray:
    """
    Compute the Conley spatial HAC meat matrix using pure numpy.

    This is the fallback when numba is not installed. It's slower but correct.

    Parameters
    ----------
    scores : np.ndarray
        n x k matrix of scores (X_i * e_i for each observation).
    lat : np.ndarray
        n-vector of latitudes in degrees.
    lon : np.ndarray
        n-vector of longitudes in degrees.
    cutoff : float
        Distance cutoff in km. Pairs beyond this are given zero weight.
    kernel : str
        "bartlett" (linearly decreasing) or "uniform" (0/1).

    Returns
    -------
    meat : np.ndarray
        k x k meat matrix for the sandwich estimator.
    """
    n, k = scores.shape
    meat = np.zeros((k, k))

    # Precompute radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    R = 6371.0

    for i in range(n):
        # Vectorized distance from i to all j
        dlat = lat_rad - lat_rad[i]
        dlon = lon_rad - lon_rad[i]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad[i]) * np.cos(lat_rad) * np.sin(dlon / 2) ** 2
        dist = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        # Apply kernel
        mask = dist <= cutoff
        if kernel == "bartlett":
            weights = np.where(mask, 1.0 - dist / cutoff, 0.0)
        else:
            weights = mask.astype(float)

        # meat += sum_j w_ij * s_i' * s_j
        # = s_i' @ (W_i * S)  where W_i is the weight vector for row i
        weighted_scores = scores * weights[:, None]  # n x k, each row j scaled by w_ij
        meat += np.outer(scores[i], weighted_scores.sum(axis=0))

    return meat


if HAS_NUMBA:
    @numba.njit(cache=True)
    def _haversine_numba(lat1, lon1, lat2, lon2):
        """Haversine distance in km, numba-compiled."""
        R = 6371.0
        dlat = (lat2 - lat1) * 0.017453292519943295  # pi/180
        dlon = (lon2 - lon1) * 0.017453292519943295
        lat1_r = lat1 * 0.017453292519943295
        lat2_r = lat2 * 0.017453292519943295
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    @numba.njit(parallel=True, cache=True)
    def _conley_meat_numba(scores, lat, lon, cutoff, use_bartlett):
        """
        Compute Conley meat matrix with numba parallelization.

        Parameters
        ----------
        scores : np.ndarray (n, k)
        lat, lon : np.ndarray (n,) in degrees
        cutoff : float, km
        use_bartlett : bool, True for bartlett kernel, False for uniform

        Returns
        -------
        meat : np.ndarray (k, k)
        """
        n, k = scores.shape
        # Each thread accumulates into its own local meat to avoid race conditions
        meat = np.zeros((k, k))

        for i in numba.prange(n):
            local_meat = np.zeros((k, k))
            for j in range(n):
                d = _haversine_numba(lat[i], lon[i], lat[j], lon[j])
                if d <= cutoff:
                    w = (1.0 - d / cutoff) if use_bartlett else 1.0
                    for p in range(k):
                        for q in range(k):
                            local_meat[p, q] += w * scores[i, p] * scores[j, q]
            # Accumulate (numba handles prange reduction)
            for p in range(k):
                for q in range(k):
                    meat[p, q] += local_meat[p, q]

        return meat


def conley_meat(
    scores: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    cutoff: float,
    kernel: str = "bartlett",
) -> np.ndarray:
    """
    Compute the Conley spatial HAC meat matrix.

    Uses numba if available for speed, otherwise falls back to numpy.

    Parameters
    ----------
    scores : np.ndarray
        n x k score matrix (X_i * e_i).
    lat : np.ndarray
        Latitude in degrees for each observation.
    lon : np.ndarray
        Longitude in degrees for each observation.
    cutoff : float
        Distance cutoff in kilometers.
    kernel : str
        "bartlett" or "uniform". Default "bartlett".

    Returns
    -------
    meat : np.ndarray
        k x k matrix.
    """
    if kernel not in ("bartlett", "uniform"):
        raise ValueError(f"kernel must be 'bartlett' or 'uniform', got '{kernel}'.")

    scores = np.ascontiguousarray(scores, dtype=np.float64)
    lat = np.ascontiguousarray(lat, dtype=np.float64)
    lon = np.ascontiguousarray(lon, dtype=np.float64)

    if HAS_NUMBA:
        return _conley_meat_numba(scores, lat, lon, cutoff, kernel == "bartlett")
    else:
        return _conley_meat_numpy(scores, lat, lon, cutoff, kernel)


def vcov_conley(
    scores: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    cutoff: float,
    kernel: str,
    bread: np.ndarray,
    is_iv: bool,
    tXZ: np.ndarray,
    tZZinv: np.ndarray,
    tZX: np.ndarray,
) -> np.ndarray:
    """
    Compute the Conley spatial HAC variance-covariance matrix.

    Parameters
    ----------
    scores : np.ndarray
        n x k score matrix.
    lat, lon : np.ndarray
        Coordinates in degrees.
    cutoff : float
        Distance cutoff in km.
    kernel : str
        "bartlett" or "uniform".
    bread : np.ndarray
        (X'X)^{-1} matrix.
    is_iv : bool
        Whether this is an IV regression.
    tXZ, tZZinv, tZX : np.ndarray
        IV projection matrices (unused if is_iv=False).

    Returns
    -------
    vcov : np.ndarray
        k x k variance-covariance matrix.
    """
    meat = conley_meat(scores, lat, lon, cutoff, kernel)

    # IV sandwich projection
    projected_meat = tXZ @ tZZinv @ meat @ tZZinv @ tZX if is_iv else meat

    return bread @ projected_meat @ bread
