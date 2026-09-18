from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyfixest.estimation.internals.literals import WeightsTypeOptions


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationWeights:
    """Canonical observation weights retained by a fitted model.

    ``values`` are always user-scale weights; ``None`` means no weights.
    Observation counts, including the frequency-weight sum, live in
    ``EstimationSample``.

    Parameters
    ----------
    values : NDArray[np.float64] or None
        Flat, user-scale observation weights. ``None`` for an unweighted fit.
    weights_type : {"aweights", "fweights"} or None
        Weight type. ``None`` for an unweighted fit.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X1", pf.get_data(), weights="weights")
    fit.observation_weights.values[:3]
    ```
    """

    values: NDArray[np.float64] | None
    weights_type: WeightsTypeOptions | None

    def __post_init__(self) -> None:
        # `unweighted()` and `from_values()` are the only constructors used by
        # the estimators; this guard catches direct misconstruction.
        if self.values is not None and self.weights_type is None:
            raise ValueError("Weighted observations must declare a `weights_type`.")

    @classmethod
    def unweighted(cls) -> ObservationWeights:
        """Construct the representation of an unweighted fit."""
        return cls(values=None, weights_type=None)

    @classmethod
    def from_values(
        cls,
        weights: NDArray[np.float64],
        *,
        weights_type: WeightsTypeOptions,
    ) -> ObservationWeights:
        """Construct canonical weighted state from user-scale weights."""
        observation_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        return cls(values=observation_weights, weights_type=weights_type)

    @property
    def is_weighted(self) -> bool:
        """Whether this state contains user-supplied observation weights."""
        return self.values is not None


@dataclass(frozen=True, slots=True, kw_only=True)
class DroppedRowCounts:
    """Counts of rows removed at each sample-filtering stage.

    Counts are mutually exclusive and stages run in field order, matching
    ``DropStageOptions``. Sample splitting is not a filtering stage.

    Parameters
    ----------
    missing : int
        Rows with a missing value in any formula variable.
    infinite : int
        Rows with an infinite value in a materialized column.
    singleton : int
        Rows removed as singleton fixed-effect levels (``fixef_rm="singleton"``).
    separation : int
        Rows removed by the GLM separation check.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X1 | f1", pf.get_data())
    fit.sample_info.dropped_by_stage
    ```
    """

    missing: int = 0
    infinite: int = 0
    singleton: int = 0
    separation: int = 0

    @property
    def total(self) -> int:
        """Number of dropped rows over all stages."""
        return self.missing + self.infinite + self.singleton + self.separation


@dataclass(frozen=True, slots=True, kw_only=True)
class EstimationSample:
    """Summary of the sample used to fit a model.

    ``dropped_row_index`` holds zero-based row positions within the estimator
    input after sample splitting.

    Parameters
    ----------
    dropped_row_index : frozenset[int]
        Positions of the rows dropped by any filtering stage.
    n_rows : int
        Number of physical fitted rows.
    n_obs : int or float
        ``n_rows``, or the weight sum for frequency weights.
    dropped_by_stage : DroppedRowCounts
        Dropped rows by filtering stage; their total equals
        ``len(dropped_row_index)``.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X1 | f1", pf.get_data())
    fit.sample_info.n_rows, fit.sample_info.n_obs, fit.sample_info.dropped_by_stage.missing
    ```
    """

    dropped_row_index: frozenset[int]
    n_rows: int
    n_obs: int | float
    dropped_by_stage: DroppedRowCounts

    def __post_init__(self) -> None:
        if self.dropped_by_stage.total != len(self.dropped_row_index):
            raise ValueError(
                "Dropped-row counts must sum to the size of the dropped row index."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class WithinLinearData:
    """Response and regressors after demeaning by the fixed effects.

    Without fixed effects, the arrays retain their original values. When
    observation weights are supplied, demeaning uses those weights.
    These arrays have not been multiplied by square-root observation weights.

    Parameters
    ----------
    response : NDArray[np.float64]
        Demeaned response, shape (n_rows, 1).
    design : NDArray[np.float64]
        Demeaned regressors after removing collinear columns,
        shape (n_rows, n_coefficients).

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X1 | f1", pf.get_data())
    fit.within_data.design.shape
    ```
    """

    response: NDArray[np.float64]
    design: NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class WithinIvData(WithinLinearData):
    """Response, regressors, and instruments after demeaning by the fixed effects.

    ``design`` is the full structural regressor matrix, including the
    endogenous regressors. ``instruments`` is the full instrument matrix,
    including exogenous regressors that instrument themselves.
    Every array is demeaned using the observation weights when supplied, but
    is not multiplied by their square roots. Without fixed effects, the
    arrays retain their original values.

    Parameters
    ----------
    response : NDArray[np.float64]
        Demeaned response, shape (n_rows, 1).
    design : NDArray[np.float64]
        Demeaned regressors, including endogenous regressors, after removing
        collinear columns, shape (n_rows, n_coefficients).
    instruments : NDArray[np.float64]
        Demeaned instruments, including exogenous regressors, after removing
        collinear columns, shape (n_rows, n_instruments).
    endogenous : NDArray[np.float64]
        Demeaned endogenous regressor, shape (n_rows, 1).

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X2 + [X1 ~ Z1] | f1", pf.get_data())
    fit.within_data.instruments.shape
    ```
    """

    instruments: NDArray[np.float64]
    endogenous: NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class GlmWorkingState:
    """Final GLM IRLS state in within scale.

    ``working_weights`` are the IRLS weights of the last iteration,
    ``W_i = w_i / (g'(mu_i)^2 V(mu_i))``, where ``w_i`` is the user-supplied
    observation weight (one when unweighted). They therefore already contain
    the observation weights, which live separately and unchanged in
    ``ObservationWeights``; nothing downstream multiplies by ``w`` again. For
    the Gaussian family ``W`` equals ``w``. Square-root weighted arrays are
    solver-local temporaries and deliberately absent.

    Parameters
    ----------
    working_response_within : NDArray[np.float64]
        Final within-scale working response, shape (n_rows,).
    design_within : NDArray[np.float64]
        Demeaned working regressors after removing collinear columns,
        shape (n_rows, n_coefficients).
    working_weights : NDArray[np.float64]
        Final IRLS weights including observation weights, shape (n_rows,).
    eta : NDArray[np.float64]
        Linear predictor including any offset, shape (n_rows,).
    mu : NDArray[np.float64]
        Response mean, shape (n_rows,).
    response_residuals : NDArray[np.float64]
        Observed response minus its mean, shape (n_rows,).
    working_residuals : NDArray[np.float64]
        Within working response minus design times coefficients, shape (n_rows,).

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feglm("Y ~ X1", pf.get_data(), family="gaussian")
    fit.working_state.mu[:3]
    ```
    """

    working_response_within: NDArray[np.float64]
    design_within: NDArray[np.float64]
    working_weights: NDArray[np.float64]
    eta: NDArray[np.float64]
    mu: NDArray[np.float64]
    response_residuals: NDArray[np.float64]
    working_residuals: NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class FittedValues:
    """In-sample predictions on the link and the response scale.

    fixest draws the same distinction through the ``type`` argument of
    ``fitted.values()``. Every fitted model publishes this value as
    ``fit.fitted_values``. For linear, IV, and quantile models the two scales
    coincide and share one array. GLMs publish their linear predictor as
    ``link`` and its inverse-link mean as ``response``, the same arrays their
    ``GlmWorkingState`` carries as IRLS output.

    Parameters
    ----------
    link : NDArray[np.float64]
        Prediction on the scale of the linear predictor, including the
        fixed-effect contribution and any offset, shape (n_rows,).
    response : NDArray[np.float64]
        Prediction on the scale of the dependent variable, ``E(Y|X)``,
        shape (n_rows,).

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X1 | f1", pf.get_data())
    fit.fitted_values.response[:3]
    ```
    """

    link: NDArray[np.float64]
    response: NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class SandwichComponents:
    """Scores, Hessian, and bread of a fitted model's sandwich covariance.


    Parameters
    ----------
    scores : NDArray[np.float64]
        Weighted scores ``W X * u``, shape (n_rows, n_coefficients).
    hessian : NDArray[np.float64]
        Weighted cross-product ``X' W X``, shape (n_coefficients,
        n_coefficients).
    bread : NDArray[np.float64]
        Inverse of ``hessian``, shape (n_coefficients, n_coefficients).

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X1 | f1", pf.get_data())
    fit.sandwich.bread
    ```
    """

    scores: NDArray[np.float64]
    hessian: NDArray[np.float64]
    bread: NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class VarianceCovariance:
    """Variance-covariance estimate of the coefficients and its building blocks.

    Parameters
    ----------
    vcov : NDArray[np.float64]
        Small-sample-adjusted covariance matrix, shape (n_coefficients,
        n_coefficients).
    meat : NDArray[np.float64] or None
        Adjusted meat of the sandwich, shape (n_coefficients,
        n_coefficients), so that ``vcov == bread @ meat @ bread`` with the
        bread of ``fit.sandwich``. For multiway clustering the per-dimension
        meats enter with their signs and adjustment factors. ``None`` where
        no sandwich exists: ``"iid"``, ``"CRV3"``, and quantile regression.
    ssc : NDArray[np.float64]
        Small-sample adjustment factors. Length one, or one entry per cluster
        dimension for CRV inference: three for two-way clustering.
    df_k : int
        Number of parameters counted by the ``k_adj`` adjustment.
    df_t : int or float
        Degrees of freedom of the t reference distribution.
    vcov_type : str
        Estimator family: ``"iid"``, ``"hetero"``, ``"HAC"``, ``"CRV"``, or
        ``"nid"``.
    vcov_type_detail : str
        Requested estimator, for example ``"HC1"``, ``"NW"``, or ``"CRV1"``.
    clustervar : tuple[str, ...]
        Cluster variables; empty unless clustered.
    G : tuple[int, ...]
        Cluster counts per dimension after the ``G_df`` rule; empty unless
        clustered.

    Examples
    --------
    ```{python}
    import numpy as np
    import pyfixest as pf

    fit = pf.feols("Y ~ X1 | f1", pf.get_data(), vcov={"CRV1": "f1"})
    cov = fit.variance_covariance
    cov.vcov_type_detail, cov.G, cov.df_t, cov.ssc
    ```

    ```{python}
    bread = fit.sandwich.bread
    np.allclose(cov.vcov, bread @ cov.meat @ bread)
    ```
    """

    vcov: NDArray[np.float64]
    meat: NDArray[np.float64] | None
    ssc: NDArray[np.float64]
    df_k: int
    df_t: int | float
    vcov_type: str
    vcov_type_detail: str
    clustervar: tuple[str, ...]
    G: tuple[int, ...]

    @property
    def is_clustered(self) -> bool:
        """Whether the estimator clusters on at least one variable."""
        return bool(self.clustervar)


@dataclass(frozen=True, slots=True, kw_only=True)
class Capabilities:
    """Inference and post-estimation features a fitted model supports.

    Each model class publishes one value in its constructor, so a method can
    reject an unsupported estimator before it reads any estimation state
    rather than reinterpreting another estimator's arrays.

    Parameters
    ----------
    crv3_inference : bool
        Whether ``vcov()`` accepts ``"CRV3"``. The jackknife refits the model
        on leave-one-cluster-out samples, so it is restricted to estimators
        whose refit replays the original estimation contract.
    hac_inference : bool
        Whether ``vcov()`` accepts the Newey-West and Driscoll-Kraay
        estimators.
    multiway_clustering : bool
        Whether ``vcov()`` accepts more than one cluster variable.
    wildboottest : bool
        Whether ``wildboottest()`` can resample the fit. Unweighted OLS only.
    cluster_causal_variance : bool
        Whether ``ccv()``, the causal cluster variance estimator, is available.
    decomposition : bool
        Whether ``decompose()``, the Gelbach decomposition, is available.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    pf.feols("Y ~ X1 | f1", pf.get_data()).capabilities
    ```

    ```{python}
    pf.fepois("Y ~ X1 | f1", pf.get_data(model="Fepois")).capabilities
    ```
    """

    crv3_inference: bool
    hac_inference: bool
    multiway_clustering: bool
    wildboottest: bool
    cluster_causal_variance: bool
    decomposition: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class RitestStatistics:
    """Randomization-inference draws, the sample statistic, and the p-value.

    ``ritest(store_ritest_statistics=True)`` publishes one value as
    ``fit.ritest_statistics``.

    Parameters
    ----------
    statistics : NDArray[np.float64]
        Resampled test statistics, shape (reps + 1,). The first entry is the
        observed statistic under the original assignment; the remaining
        ``reps`` entries are the randomization draws.
    sample_stat : float
        The observed statistic, centered at the null hypothesis value.
    pvalue : float
        Randomization-inference p-value of the test.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X1 + X2", pf.get_data())
    fit.ritest("X1", reps=100, store_ritest_statistics=True)
    fit.ritest_statistics.pvalue
    ```
    """

    statistics: NDArray[np.float64]
    sample_stat: float
    pvalue: float


@dataclass(frozen=True, slots=True, kw_only=True)
class CoefficientTable:
    """Coefficient estimates with their standard errors, statistics, and bounds.

    ``get_inference()`` publishes one value as ``fit.coeftable`` from the
    current ``fit.variance_covariance``; it is the table that fixest's
    ``coeftable()`` prints, plus the confidence bounds of ``confint()``.

    Parameters
    ----------
    estimate : NDArray[np.float64]
        Coefficient estimates, shape (n_coefficients,).
    se : NDArray[np.float64]
        Standard errors, the square root of the covariance diagonal.
    tstat : NDArray[np.float64]
        ``estimate / se``.
    pvalue : NDArray[np.float64]
        Two-sided p-values from the model's reference distribution with
        ``variance_covariance.df_t`` degrees of freedom.
    conf_int : NDArray[np.float64]
        Lower and upper confidence bounds, shape (2, n_coefficients), at
        level ``1 - alpha``.
    alpha : float
        Significance level of the bounds.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X1 | f1", pf.get_data(), vcov="HC1")
    table = fit.coeftable
    table.estimate, table.se, table.conf_int
    ```
    """

    estimate: NDArray[np.float64]
    se: NDArray[np.float64]
    tstat: NDArray[np.float64]
    pvalue: NDArray[np.float64]
    conf_int: NDArray[np.float64]
    alpha: float
