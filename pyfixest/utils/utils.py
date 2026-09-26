from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_args

import numpy as np
import pandas as pd
from formulaic import Formula
from formulaic.utils.context import capture_context as _capture_context

from pyfixest.utils.dev_utils import _create_rng

if TYPE_CHECKING:
    # Annotation-only: pyfixest.estimation imports this module at package
    # import time, so a module-level runtime import here would be circular.
    from pyfixest.estimation.internals.literals import GDfOptions, KFixefOptions


@dataclass(frozen=True, slots=True, kw_only=True)
class Ssc:
    """Small-sample correction options.

    Parameters
    ----------
    k_adj : bool, default True
        Apply the ``(N - 1) / (N - k)`` adjustment (``N / (N - k)`` for
        heteroskedasticity-robust estimators).
    k_fixef : {"none", "full", "nonnested"}, default "nonnested"
        Which fixed-effect parameters count toward ``k``.
    G_adj : bool, default True
        Apply the ``G / (G - 1)`` cluster adjustment.
    G_df : {"min", "conventional"}, default "min"
        Whether multiway clustering adjusts every dimension with the smallest
        cluster count or with its own.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    pf.ssc(k_adj=False)
    ```
    """

    k_adj: bool = True
    k_fixef: KFixefOptions = "nonnested"
    G_adj: bool = True
    G_df: GDfOptions = "min"

    def __post_init__(self) -> None:
        from pyfixest.estimation.internals.literals import GDfOptions, KFixefOptions

        if not isinstance(self.k_adj, bool):
            raise TypeError("k_adj must be True or False.")
        if not isinstance(self.G_adj, bool):
            raise TypeError("G_adj must be True or False.")
        k_fixef_values = get_args(KFixefOptions)
        if self.k_fixef not in k_fixef_values:
            raise ValueError(
                f"k_fixef must be one of {k_fixef_values}; got {self.k_fixef!r}."
            )
        g_df_values = get_args(GDfOptions)
        if self.G_df not in g_df_values:
            raise ValueError(f"G_df must be one of {g_df_values}; got {self.G_df!r}.")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Ssc:
        """Build from a legacy ``{"k_adj": ..., "k_fixef": ..., ...}`` dict."""
        unknown = set(mapping) - {"k_adj", "k_fixef", "G_adj", "G_df"}
        if unknown:
            raise ValueError(
                f"ssc accepts the keys k_adj, k_fixef, G_adj, and G_df; got {sorted(unknown)}."
            )
        return cls(**mapping)


@dataclass(frozen=True, slots=True, kw_only=True)
class DegreesOfFreedomCounts:
    """Model counts that enter the small-sample correction.

    Parameters
    ----------
    N : int or float
        Number of observations (the frequency-weight sum under fweights).
    k : int
        Number of estimated coefficients, excluding fixed effects.
    k_fe : int
        Number of fixed-effect levels across all fixed effects.
    n_fe : int
        Number of fixed effects; ``Y ~ X | f1 + f2`` has two.
    k_fe_nested : int
        Fixed-effect levels nested within the cluster variables.
    n_fe_fully_nested : int
        Fixed effects fully nested within the cluster variables.
    G : int or float
        Number of clusters; the number of time periods for HAC and ``N``
        for heteroskedasticity-robust inference.
    """

    N: int | float
    k: int
    k_fe: int
    n_fe: int
    k_fe_nested: int = 0
    n_fe_fully_nested: int = 0
    G: int | float


@dataclass(frozen=True, slots=True, kw_only=True)
class SmallSampleCorrection:
    """Result of ``get_ssc()``.

    Parameters
    ----------
    adj : float
        Factor that multiplies the covariance matrix.
    df_k : int
        Parameters counted by the ``k_adj`` adjustment.
    df_t : int or float
        Degrees of freedom of the t reference distribution.
    """

    adj: float
    df_k: int
    df_t: int | float


def ssc(
    k_adj: bool = True,
    k_fixef: KFixefOptions = "nonnested",
    G_adj: bool = True,
    G_df: GDfOptions = "min",
    **kwargs: Any,
) -> Ssc:
    """
    Set the small sample correction factor applied in `get_ssc()`.

    Parameters
    ----------
    k_adj : bool, default True
        If True, applies a small sample correction of (N-1) / (N-k) where N
        is the number of observations and k is the number of estimated
        coefficients excluding any fixed effects projected out by either
        `feols()` or `fepois()`.
    k_fixef : str, default "none"
        Equal to 'none': the fixed effects parameters are discarded when
        calculating k in (N-1) / (N-k).
    G_adj : bool, default True
        If True, a cluster correction G/(G-1) is performed, with G the number
        of clusters. This argument is only relevant for clustered errors.
    G_df : str, default "conventional"
        Controls how "G" is computed for multiway clustering if G_adj = True.
        Note that the covariance matrix in the multiway clustering case is of
        the form V = V_1 + V_2 - V_12. If "conventional", then each summand G_i
        is multiplied with a small sample adjustment G_i / (G_i - 1). If "min",
        all summands are multiplied with the same value, min(G) / (min(G) - 1).
        This argument is only relevant for clustered errors.

    Details
    -------
    The small sample correction choices mimic fixest's behavior. For details, see
    https://cran.r-project.org/web/packages/fixest/vignettes/standard_errors.html.

    In general, if k_adj = True, we multiply the variance covariance matrix V with a
    small sample correction factor of (N-1) / (N-k), where N is the number of
    observations and k is the number of estimated coefficients.

    If k_fixef = "none", the fixed effects parameters are discarded when
    calculating k. This is the default behavior and currently the only
    option. Note that it is not r-fixest's default behavior.

    Hence if k_adj = True, the covariance matrix is computed as
    V = V x (N-1) / (N-k) for iid and heteroskedastic errors.

    If k_adj = False, no small sample correction is applied of the type
    above is applied.

    If G_adj = True, a cluster correction of G/(G-1) is performed,
    with G the number of clusters.

    If k_adj = True and G_adj = True, V = V x (N - 1) / N - k) x G/(G-1)
    for cluster robust errors where G is the number of clusters.

    If k_adj = False and G_adj = True, V = V x G/(G-1) for cluster robust
    errors, i.e. we drop the (N-1) / (N-k) factor. And if G_adj = False,
    no cluster correction is applied.

    Things are slightly more complicated for multiway clustering. In this
    case, we compute the variance covariance matrix as V = V1 + V2 - V_12.

    If G_adj = True and G_df = "conventional", then
    V += [V x G_i / (G_i - 1) for i in [1, 2, 12]], i.e. each separate
    covariance matrix G_i is multiplied with a small sample adjustment
    G_i / (G_i - 1) corresponding to the number of clusters in the
    respective covariance matrix. This is the default behavior
    for clustered errors.

    If G_df = "min", then
    V += [V x min(G) / (min(G) - 1) for i in [1, 2, 12]].

    Returns
    -------
    Ssc
        The validated options; see [Ssc](/reference/utils.utils.Ssc.qmd).

    Examples
    --------
    ```{python}
    import pyfixest as pf

    data = pf.get_data()

    # turn off both the k and the G adjustment
    fit = pf.feols("Y ~ X1 | f1", data, vcov={"CRV1": "f1"})
    fit_no_adj = pf.feols(
        "Y ~ X1 | f1", data, vcov={"CRV1": "f1"}, ssc=pf.ssc(k_adj=False, G_adj=False)
    )

    pf.etable([fit, fit_no_adj])
    ```

    Defaults follow `fixest`. See
    [On Small Sample Corrections](/explanation/ssc.qmd) for details.
    """
    deprecated_mapping = {
        "adj": "k_adj",
        "fixef_k": "k_fixef",
        "cluster_df": "G_df",
        "cluster_adj": "G_adj",
    }

    for old_name, new_name in deprecated_mapping.items():
        if old_name in kwargs:
            warnings.warn(
                f"The '{old_name}' argument is deprecated. Use '{new_name}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Update parameter values if new name not already provided
            if new_name == "k_adj" and "k_adj" not in kwargs:
                k_adj = kwargs[old_name]
            elif new_name == "k_fixef" and "k_fixef" not in kwargs:
                k_fixef = kwargs[old_name]
            elif new_name == "G_df" and "G_df" not in kwargs:
                G_df = kwargs[old_name]
            elif new_name == "G_adj" and "G_adj" not in kwargs:
                G_adj = kwargs[old_name]

    return Ssc(k_adj=k_adj, k_fixef=k_fixef, G_adj=G_adj, G_df=G_df)


def get_ssc(
    ssc_options: Ssc,
    counts: DegreesOfFreedomCounts,
    *,
    vcov_type: str,
) -> SmallSampleCorrection:
    """
    Compute the small sample adjustment factor and the degrees of freedom.

    Parameters
    ----------
    ssc_options : Ssc
        The options created via the ssc() function.
    counts : DegreesOfFreedomCounts
        Observation, coefficient, fixed-effect, and cluster counts.
    vcov_type : str
        The type of covariance matrix: "iid", "hetero", "HAC", or "CRV".

    Returns
    -------
    SmallSampleCorrection
        The factor `adj` that multiplies the covariance matrix, the parameter
        count `df_k` it used, and the degrees of freedom `df_t` of the t
        distribution.

    Examples
    --------
    Called internally by the estimation functions. Use it directly to reproduce
    an adjustment factor by hand.

    ```{python}
    import pyfixest as pf
    from pyfixest.utils.utils import DegreesOfFreedomCounts, get_ssc

    # cluster-robust adjustment: 1000 observations, 3 coefficients, 20 clusters
    counts = DegreesOfFreedomCounts(N=1000, k=3, k_fe=0, n_fe=0, G=20)
    get_ssc(pf.ssc(), counts, vcov_type="CRV")
    ```

    Configure the behaviour with [ssc()](/reference/utils.utils.ssc.qmd). See
    [On Small Sample Corrections](/explanation/ssc.qmd) for the formulas.
    """
    N, k, k_fe, n_fe = counts.N, counts.k, counts.k_fe, counts.n_fe
    k_fe_nested, n_fe_fully_nested = counts.k_fe_nested, counts.n_fe_fully_nested
    G: int | float = counts.G

    G_adj_value = 1.0
    adj_value = 1.0

    # see here for why:
    # https://github.com/lrberge/fixest/issues/554

    # subtract one for each fixed effect, except for the first
    k_fe_adj = k_fe - (n_fe - 1) if n_fe > 1 else k_fe

    if ssc_options.k_fixef == "none":
        df_k = k
    elif ssc_options.k_fixef == "nonnested":
        if n_fe == 0:
            df_k = k
        elif k_fe_nested == 0:
            # no nested fe, so just add all fixed effects
            df_k = k + k_fe_adj
        else:
            # subtract nested fixed effects and add one for each fully nested
            # subtracted fixed effect back
            df_k = k + k_fe_adj - k_fe_nested + n_fe_fully_nested
    else:
        # "full": add all fixed effects
        df_k = k + k_fe_adj if n_fe > 0 else k

    if ssc_options.k_adj:
        adj_value = (N - 1) / (N - df_k) if vcov_type != "hetero" else N / (N - df_k)

    # G_adj applied with G = N for hetero but not for iid
    if vcov_type in ["CRV", "HAC"] and ssc_options.G_adj:
        G_adj_value = G / (G - 1)

    df_t = N - df_k if vcov_type in ["iid", "hetero", "HAC-TS"] else G - 1
    return SmallSampleCorrection(adj=adj_value * G_adj_value, df_k=df_k, df_t=df_t)


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

    Examples
    --------
    ```{python}
    import pyfixest as pf

    data = pf.get_data()
    data.head()
    ```

    The data set contains a continuous outcome `Y`, covariates `X1` and `X2`,
    fixed effects `f1`, `f2` and `f3`, an instrument `Z1`, and some missing
    values. Set `model="Fepois"` for a count outcome.

    ```{python}
    pf.get_data(model="Fepois")["Y"].head()
    ```
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

    formula = Formula("~ X1 + X2 + f1 + f2 + f3")
    mm = formula.get_model_matrix(data=X, output="pandas")

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

    if model == "Fepois":
        # add separation
        idx = np.array([10, 11, 12])
        df.loc[idx[0], "f1"] = np.max(df["f1"]) + 1
        df.loc[idx[1], "f2"] = np.max(df["f2"]) + 1
        df.loc[idx[2], "f3"] = np.max(df["f3"]) + 1

    return df


def simultaneous_crit_val(
    C: np.ndarray, S: int, alpha: float = 0.05, seed: int | None = None
) -> float:
    """
    Simultaneous Critical Values.

    Obtain critical values for simultaneous inference on linear model parameters
    using the Multiplier bootstrap.

    Parameters
    ----------
    C: numpy.ndarray
        Positive semidefinite covariance matrix. Symmetric, with as many
        rows/columns as parameters of interest.
    S: int
        Number of replications
    alpha: float
        Significance level. Defaults to 0.05
    seed: int, optional
        Seed for the random number generator. Default is None.

    Returns
    -------
    float
        Estimated (1 - alpha) quantile of the largest absolute Gaussian coordinate.

    Raises
    ------
    ValueError
        If the covariance matrix has negative eigenvalues exceeding
        floating-point roundoff.
    """

    def msqrt(C: np.ndarray) -> np.ndarray:
        eig_vals, eig_vecs = np.linalg.eigh(C)
        # Rank-deficient cluster covariance matrices can have tiny negative
        # eigenvalues from roundoff. The tolerance scales with matrix size and
        # spectral norm; genuinely indefinite matrices are not valid covariances.
        tolerance = np.finfo(eig_vals.dtype).eps * C.shape[0] * np.max(np.abs(eig_vals))
        if np.min(eig_vals) < -tolerance:
            raise ValueError("Covariance matrix must be positive semidefinite.")
        return (
            eig_vecs
            @ np.diag(np.sqrt(np.maximum(eig_vals, 0)))
            @ np.linalg.inv(eig_vecs)
        )

    rng = _create_rng(seed)
    p = C.shape[0]
    tmaxs = np.max(np.abs(msqrt(C) @ rng.normal(size=(p, S))), axis=0)
    return np.quantile(tmaxs, 1 - alpha)


def capture_context(context: int | Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Explicitly capture the context to be used by subsequent formula
    materialisations.

    Parameters
    ----------
    context: Union[int, Mapping[str, Any]]
        The context from which variables (and custom transforms/etc)
        should be inherited.

        When specified as an integer, it is interpreted as a frame offset
        from the caller's frame. Since we use this function in the context
        of a library, we need to account for the extra frames, hence
        we add 2 to the context (one for this and one for the frame the
        function is being called in).

        Otherwise, a mapping from variable name to value is expected.

        When nesting in a library, and attempting to capture user-context,
        make sure you account for the extra frames introduced by your wrappers.

    Returns
    -------
    Mapping[str, Any]
        The context that should be later passed to the Formulaic materialization
        procedure like: `.get_model_matrix(..., context=<this object>)`.
    """
    # formulaic's `_capture_context` returns `None` when frame introspection
    # fails; callers rely on an empty mapping (not None) for "no context".
    return (
        (_capture_context(context + 2) or {}) if isinstance(context, int) else context
    )


def _check_balanced(panel_arr: np.ndarray, time_arr: np.ndarray) -> bool:
    """
    Check if the panel data is balanced.

    Parameters
    ----------
    panel_arr: np.ndarray
        The panel variable for clustering.
    time_arr: np.ndarray
        The time variable for clustering.

    Returns
    -------
    bool
        True if the panel data is balanced, False otherwise.
    """
    unique_panels = np.unique(panel_arr)
    unique_times = np.unique(time_arr)
    expected_time_count = len(unique_times)

    for panel_id in unique_panels:
        mask = panel_arr == panel_id
        panel_times = np.unique(time_arr[mask])

        if len(panel_times) != expected_time_count:
            return False

    return True
