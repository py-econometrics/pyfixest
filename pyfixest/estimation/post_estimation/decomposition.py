from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Any, get_args

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.linalg import pinv
from scipy.sparse import csc_matrix, diags, hstack, spmatrix, vstack
from scipy.sparse.linalg import lsqr
from scipy.stats import t
from tqdm import tqdm

from pyfixest.errors import VcovTypeNotSupportedError
from pyfixest.estimation.internals.literals import (
    DecompositionInference,
    DecompositionVcovDetail,
    DecompositionVcovFamily,
    DecompositionVcovTypeOptions,
    _validate_literal_argument,
)
from pyfixest.estimation.internals.vcov_ import crv1_meat
from pyfixest.estimation.internals.vcov_utils import (
    ClusterPrep,
    SscContext,
    assemble_crv_vcov,
    prepare_cluster_state,
)
from pyfixest.utils.dev_utils import DataFrameType

# Panel name mappings for consistent API
PANEL_ALIASES = {
    "levels": "Levels (units)",
    "share_full": "Share of Full Effect",
    "share_explained": "Share of Explained Effect",
}
_DECOMPOSITION_VCOV_FAMILIES: dict[DecompositionVcovDetail, DecompositionVcovFamily] = {
    "iid": "iid",
    "HC1": "hetero",
    "CRV1": "CRV",
}


@dataclass(frozen=True, slots=True)
class GelbachVcovConfig:
    """Canonical covariance configuration for analytical Gelbach inference.

    Attributes
    ----------
    detail : DecompositionVcovDetail
        User-facing estimator within the family.
    ssc : float | None
        Small-sample correction for non-clustered inference.
    df : int | None
        Degrees of freedom for non-clustered inference.
    cluster_prep : ClusterPrep | None
        Prepared cluster identifiers and fixed-effect nesting state.
    ssc_context : SscContext | None
        Model quantities used for clustered small-sample corrections.
    """

    detail: DecompositionVcovDetail = "HC1"
    ssc: float | None = None
    df: int | None = None
    cluster_prep: ClusterPrep | None = None
    ssc_context: SscContext | None = None

    @property
    def family(self) -> DecompositionVcovFamily:
        "Return the computational covariance family."
        return _DECOMPOSITION_VCOV_FAMILIES[self.detail]


@dataclass(frozen=True, slots=True)
class GelbachCoreState:
    """Influence-function state for analytical Gelbach covariance.

    Attributes
    ----------
    scores : np.ndarray
        Full-effect and mediator-group influence scores, shape `(N, 1 + J)`.
    full_resid : np.ndarray
        Full-regression residuals, shape `(N,)`.
    auxiliary_resid : np.ndarray
        Grouped auxiliary-regression residuals, shape `(N, J)`.
    full_weight : np.ndarray
        Full-regression loading for the focal coefficient, shape `(N,)`.
    beta2_loading : np.ndarray
        Full-regression loadings for grouped mediator effects, shape `(N, J)`.
    short_weight : np.ndarray
        Short-regression loading for the focal coefficient, shape `(N,)`.
    group_names : tuple[str, ...]
        Names of the reported mediator groups.
    remainder_col : int | None
        Core-score column holding unreported mediators, if any.
    """

    scores: np.ndarray
    full_resid: np.ndarray
    auxiliary_resid: np.ndarray
    full_weight: np.ndarray
    beta2_loading: np.ndarray
    short_weight: np.ndarray
    group_names: tuple[str, ...]
    remainder_col: int | None


def _parse_cluster_spec(cluster_spec: str) -> tuple[str, ...]:
    "Parse and validate a one- or two-way cluster specification."
    clustervar = tuple(name.strip() for name in cluster_spec.split("+"))
    if not 1 <= len(clustervar) <= 2 or any(not name for name in clustervar):
        raise ValueError("CRV1 supports one- or two-way clustering.")
    return clustervar


def _unsupported_decomposition_vcov(detail: str) -> VcovTypeNotSupportedError:
    "Build the error for a recognized but unsupported covariance estimator."
    return VcovTypeNotSupportedError(
        f"Analytical decomposition inference does not support {detail}. "
        "Supported estimators are IID, HC1, and CRV1."
    )


def _parse_decomposition_vcov(
    *,
    vcov: DecompositionVcovTypeOptions | dict[str, str] | None,
    cluster: str | None,
    parent_is_clustered: bool,
    parent_vcov_detail: str,
    parent_clustervar: list[str] | None,
    only_coef: bool,
) -> tuple[DecompositionVcovDetail, tuple[str, ...]]:
    "Normalize decomposition covariance input into detail and cluster variables."
    if vcov is not None and cluster is not None:
        raise ValueError(
            "Specify CRV1 inference with either 'cluster' or 'vcov', not both."
        )

    if isinstance(vcov, dict):
        if len(vcov) != 1:
            raise ValueError("The decomposition 'vcov' dictionary must have one entry.")
        detail, cluster_spec = next(iter(vcov.items()))
        if detail in {"CRV2", "CRV3"}:
            raise _unsupported_decomposition_vcov(detail)
        if detail != "CRV1":
            raise ValueError(
                "Clustered decomposition inference requires {'CRV1': 'cluster'}."
            )
        if not isinstance(cluster_spec, str):
            raise TypeError("The CRV1 cluster specification must be a string.")
        return "CRV1", _parse_cluster_spec(cluster_spec)

    if isinstance(vcov, str):
        if vcov in {"HC2", "HC3"}:
            raise _unsupported_decomposition_vcov(vcov)
        try:
            _validate_literal_argument(vcov, DecompositionVcovTypeOptions)
        except ValueError as exc:
            raise ValueError(
                "Analytical decomposition inference supports 'iid', 'hetero', "
                "'HC1', or {'CRV1': 'cluster'}."
            ) from exc
        return ("iid", ()) if vcov == "iid" else ("HC1", ())

    if vcov is not None:
        raise TypeError("'vcov' must be a string, dictionary, or None.")
    if cluster is not None:
        if not isinstance(cluster, str):
            raise TypeError("The CRV1 cluster specification must be a string.")
        return "CRV1", _parse_cluster_spec(cluster)
    if parent_is_clustered and not only_coef:
        if parent_vcov_detail != "CRV1":
            raise _unsupported_decomposition_vcov(parent_vcov_detail)
        if parent_clustervar is None:
            raise RuntimeError("The clustered parent model has no cluster variables.")
        return "CRV1", tuple(parent_clustervar)
    return "HC1", ()


def prepare_decomposition_vcov(
    *,
    vcov: DecompositionVcovTypeOptions | dict[str, str] | None,
    cluster: str | None,
    parent_is_clustered: bool,
    parent_vcov_detail: str,
    parent_clustervar: list[str] | None,
    only_coef: bool,
    data: DataFrameType | None,
    ssc_context: SscContext,
    fixef: str | None,
    fe: pd.DataFrame | np.ndarray | None,
    k_fe: np.ndarray | pd.Series | None,
) -> GelbachVcovConfig:
    "Parse and prepare covariance state for analytical Gelbach inference."
    detail, clustervar = _parse_decomposition_vcov(
        vcov=vcov,
        cluster=cluster,
        parent_is_clustered=parent_is_clustered,
        parent_vcov_detail=parent_vcov_detail,
        parent_clustervar=parent_clustervar,
        only_coef=only_coef,
    )

    if only_coef:
        return GelbachVcovConfig(detail=detail)
    if detail == "CRV1":
        prep = prepare_cluster_state(
            data=data,
            clustervar=list(clustervar),
            ssc_dict=ssc_context.ssc_dict,
            fixef=fixef,
            fe=fe,
            k_fe=k_fe,
        )
        if min(prep.G) < 2:
            raise ValueError("CRV1 inference requires at least two clusters.")
        return GelbachVcovConfig(
            detail=detail,
            cluster_prep=prep,
            ssc_context=ssc_context,
        )

    family = _DECOMPOSITION_VCOV_FAMILIES[detail]
    ssc, _, df = ssc_context.get_ssc(
        vcov_type=family,
        G=1 if family == "iid" else ssc_context.N,
    )
    return GelbachVcovConfig(
        detail=detail,
        ssc=float(ssc[0]),
        df=int(df),
    )


def _sparse_grouped_values(
    values: np.ndarray, group_indices: list[list[int]]
) -> csc_matrix:
    "Store grouped parameter values with one nonzero per assigned mediator."
    if not group_indices:
        return csc_matrix((len(values), 0), dtype=float)

    rows = np.concatenate([np.asarray(idx, dtype=int) for idx in group_indices])
    columns = np.concatenate(
        [np.full(len(idx), group_idx) for group_idx, idx in enumerate(group_indices)]
    )
    return csc_matrix(
        (values[rows], (rows, columns)),
        shape=(len(values), len(group_indices)),
    )


def _inference_group_indices(
    *,
    reported_groups: dict[str, list[int]],
    mediator_names: list[str],
) -> tuple[list[list[int]], int | None]:
    "Append an internal group for mediators omitted from the reported groups."
    group_indices = [list(idx) for idx in reported_groups.values()]
    assigned = np.zeros(len(mediator_names), dtype=bool)
    for idx in group_indices:
        assigned[idx] = True

    eligible = np.array([name != "Intercept" for name in mediator_names])
    remainder = np.flatnonzero(eligible & ~assigned).tolist()
    if not remainder:
        return group_indices, None

    group_indices.append(remainder)
    # Column zero is the full-effect score, so the appended group's one-based
    # position is also its column in the compact covariance basis.
    return group_indices, len(group_indices)


def _normalize_mediator_groups(
    *,
    combine_covariates: dict[str, list[str] | re.Pattern[str]] | None,
    coefficient_names: list[str],
    mediator_names: list[str],
    x1_vars: list[str] | None,
) -> tuple[dict[str, list[str]], dict[str, list[int]]]:
    "Normalize mediator groups and build their column indices."
    mediator_indices = {name: idx for idx, name in enumerate(mediator_names)}
    if combine_covariates is None:
        groups = {name: [name] for name in mediator_names if name != "Intercept"}
    else:
        groups = {}
        for group_name, covariates in combine_covariates.items():
            if isinstance(covariates, re.Pattern):
                matched = [
                    name for name in coefficient_names if covariates.search(name)
                ]
                if not matched:
                    raise ValueError(f"No covariates match the regex {covariates}.")
                groups[group_name] = matched
            elif isinstance(covariates, list):
                groups[group_name] = list(covariates)
            else:
                raise TypeError("Values in combine_covariates_dict must be lists.")

    x1_set = set(x1_vars or [])
    owners: dict[str, str] = {}
    indices: dict[str, list[int]] = {}
    for group_name, covariates in groups.items():
        overlap = sorted(x1_set.intersection(covariates))
        if overlap:
            raise ValueError(
                f"Variables {overlap} cannot be in both x1_vars "
                "and combine_covariates values."
            )

        group_indices = []
        for covariate in covariates:
            if covariate not in mediator_indices:
                raise ValueError(
                    f"The variable '{covariate}' is not in the mediator names."
                )
            if covariate in owners:
                raise ValueError(
                    f"Variables {{{covariate!r}}} are in both "
                    f"'{owners[covariate]}' and '{group_name}' groups."
                )
            owners[covariate] = group_name
            group_indices.append(mediator_indices[covariate])
        indices[group_name] = group_indices

    return groups, indices


@dataclass
class GelbachResults:
    """Container for all Gelbach decomposition results."""

    direct_effect: float
    full_effect: float
    explained_effect: float
    unexplained_effect: float
    mediator_effects: dict[str, float]

    def __post_init__(self):
        """Validate that explained_effect equals sum of mediator effects."""
        computed_explained = sum(self.mediator_effects.values())
        if not np.isclose(self.explained_effect, computed_explained, atol=1e-10):
            raise ValueError(
                f"Explained effect {self.explained_effect} != sum of mediators {computed_explained}"
            )

    @property
    def absolute(self) -> dict[str, float]:
        """Absolute levels (backward compatibility with contribution_dict)."""
        return {
            "direct_effect": self.direct_effect,
            "full_effect": self.full_effect,
            "explained_effect": self.explained_effect,
            "unexplained_effect": self.unexplained_effect,
            **self.mediator_effects,
        }

    @property
    def relative_to_explained(self) -> dict[str, float]:
        """Relative to explained effect (backward compatibility)."""
        if self.explained_effect == 0:
            return {name: np.nan for name in self.absolute}
        return {
            name: value / self.explained_effect for name, value in self.absolute.items()
        }

    @property
    def relative_to_direct(self) -> dict[str, float]:
        """Relative to direct effect (backward compatibility)."""
        if self.direct_effect == 0:
            return {name: np.nan for name in self.absolute}
        return {
            name: value / self.direct_effect for name, value in self.absolute.items()
        }

    @property
    def all_effect_names(self) -> list[str]:
        """All effect names (core + mediators)."""
        return list(self.absolute.keys())

    def to_dict(self, relative_to: str | None = None) -> dict[str, float]:
        """
        Convert to dictionary format.

        Parameters
        ----------
        relative_to : str, optional
            If None, returns absolute values.
            If "explained", returns values relative to explained effect.
            If "direct", returns values relative to direct effect.
        """
        if relative_to is None:
            return self.absolute
        elif relative_to == "explained":
            return self.relative_to_explained
        elif relative_to == "direct":
            return self.relative_to_direct
        else:
            raise ValueError(
                f"relative_to must be None, 'explained', or 'direct'. Got {relative_to}"
            )


@dataclass(frozen=True, slots=True)
class GelbachComputation:
    """Intermediate Gelbach quantities shared by point estimates and inference."""

    results: GelbachResults
    core_state: GelbachCoreState | None


def _build_gelbach_core_state(
    *,
    X: spmatrix,
    X1: spmatrix,
    X2: spmatrix,
    Y: np.ndarray,
    beta_full: np.ndarray,
    beta2: np.ndarray,
    x1_inv: np.ndarray,
    x_inv: np.ndarray,
    gamma_matrix: np.ndarray,
    mask: np.ndarray,
    decomp_var_in_X1_idx: int,
    decomp_var_in_X_idx: int,
    mediator_names: list[str],
    reported_groups: dict[str, list[int]],
) -> GelbachCoreState:
    "Build the compact influence basis for analytical Gelbach inference."
    Y = np.asarray(Y, dtype=float).reshape(-1)
    beta_full = np.asarray(beta_full, dtype=float)
    beta2 = np.asarray(beta2, dtype=float)
    gamma = gamma_matrix[decomp_var_in_X1_idx, :]

    full_resid = Y - np.asarray(X @ beta_full).reshape(-1)
    short_weight = np.asarray(X1 @ x1_inv[:, decomp_var_in_X1_idx]).reshape(-1)
    full_weight = np.asarray(X @ x_inv[:, decomp_var_in_X_idx]).reshape(-1)
    beta2_indices = np.flatnonzero(mask)
    group_indices, remainder_col = _inference_group_indices(
        reported_groups=reported_groups,
        mediator_names=mediator_names,
    )

    if group_indices:
        group_gamma = _sparse_grouped_values(gamma, group_indices)
        group_beta2 = _sparse_grouped_values(beta2, group_indices)
        beta2_weight = (group_gamma.T @ x_inv[:, beta2_indices].T).T
        beta2_loading = np.asarray(X @ beta2_weight)
        auxiliary_fit = (group_beta2.T @ gamma_matrix.T).T
        grouped_mediators = (X2 @ group_beta2).toarray()
        auxiliary_resid = grouped_mediators - np.asarray(X1 @ auxiliary_fit)
    else:
        nobs = X.shape[0]
        beta2_loading = np.empty((nobs, 0))
        auxiliary_resid = np.empty((nobs, 0))

    mediator_scores = (
        beta2_loading * full_resid[:, None] + short_weight[:, None] * auxiliary_resid
    )
    scores = np.column_stack((full_weight * full_resid, mediator_scores))
    return GelbachCoreState(
        scores=scores,
        full_resid=full_resid,
        auxiliary_resid=auxiliary_resid,
        full_weight=full_weight,
        beta2_loading=beta2_loading,
        short_weight=short_weight,
        group_names=tuple(reported_groups),
        remainder_col=remainder_col,
    )


def _iid_gelbach_vcov(*, state: GelbachCoreState, nobs: int) -> np.ndarray:
    "Compute Gelbach's homoskedastic joint-system covariance."
    residuals = np.column_stack((state.full_resid, state.auxiliary_resid))
    residual_cov = residuals.T @ residuals / (nobs - 1)

    full_equation_loadings = np.column_stack((state.full_weight, state.beta2_loading))
    loading_crossprod = full_equation_loadings.T @ full_equation_loadings
    auxiliary_crossprod = full_equation_loadings.T @ state.short_weight
    short_crossprod = state.short_weight @ state.short_weight

    cross_cov = np.concatenate(([0.0], residual_cov[0, 1:]))
    vcov = residual_cov[0, 0] * loading_crossprod
    vcov += np.outer(auxiliary_crossprod, cross_cov)
    vcov += np.outer(cross_cov, auxiliary_crossprod)
    vcov[1:, 1:] += short_crossprod * residual_cov[1:, 1:]
    return vcov


def _crv1_gelbach_vcov(
    *, scores: np.ndarray, config: GelbachVcovConfig
) -> tuple[np.ndarray, int]:
    "Compute CRV1 covariance through PyFixest's shared cluster loop."
    if config.cluster_prep is None or config.ssc_context is None:
        raise RuntimeError("CRV1 inference requires prepared cluster state.")

    vcov, _, _, df = assemble_crv_vcov(
        prep=config.cluster_prep,
        k=scores.shape[1],
        ssc_context=config.ssc_context,
        cluster_vcov=partial(crv1_meat, scores),
    )
    return vcov, df


def _absolute_transform(
    *,
    absolute_names: list[str],
    group_names: tuple[str, ...],
    remainder_col: int | None,
) -> np.ndarray:
    "Map the compact covariance basis to every reported Gelbach effect."
    n_core = 1 + len(group_names) + int(remainder_col is not None)
    transform = np.zeros((len(absolute_names), n_core))
    row_idx = {name: idx for idx, name in enumerate(absolute_names)}

    transform[row_idx["direct_effect"], :] = 1.0
    transform[row_idx["full_effect"], 0] = 1.0
    transform[row_idx["explained_effect"], 1 : len(group_names) + 1] = 1.0
    transform[row_idx["unexplained_effect"], 0] = 1.0
    if remainder_col is not None:
        transform[row_idx["unexplained_effect"], remainder_col] = 1.0
    for group_idx, group_name in enumerate(group_names, start=1):
        transform[row_idx[group_name], group_idx] = 1.0
    return transform


@dataclass
class GelbachDecomposition:
    """
    Gelbach Decomposition (equivalent to a Linear Mediation Model).

    Implements the Gelbach (2016) decomposition method to decompose the effect of a
    focal variable into explained and unexplained components. The method
    compares coefficients from a "short" regression (outcome on treatment) with a
    "long" regression (outcome on focal variable plus covariates).

    Initial implementation by Apoorva Lal at
    https://gist.github.com/apoorvalal/e7dc9f3e52dcd9d51854b28b3e8a7ba4.


    This class performs the statistical decomposition and provides methods for
    summarizing and displaying results via `tidy()`, `summary()`, and `etable()`.

    Parameters
    ----------
    decomp_var : str
        The focal variable whose effect is to be decomposed.
    coefnames : list[str]
        Names of all coefficients in the regression model.
    depvarname : str
        Name of the dependent variable.
    nthreads : int, optional
        Number of threads for bootstrap inference, by default -1 (use all available).
    x1_vars : list[str], optional
        Additional variables to include in both short and long regressions, by default None.
    cluster_df : pd.Series, optional
        Cluster variable for bootstrap inference, by default None.
    combine_covariates : dict[str, list[str] | re.Pattern[str]], optional
        Dictionary grouping mediator variables for analysis, by default None.
    agg_first : bool, optional
        Whether to use aggregate-first algorithm for high-dimensional mediators, by default False.
    only_coef : bool, optional
        If True, skip inference and only compute point estimates, by default False.
    inference : str, optional
        Inference method. One of "analytic" or "bootstrap", by default "analytic".
    vcov_config : GelbachVcovConfig, optional
        Canonical variance-estimator configuration for analytical inference.
    weights_type : str, optional
        Type of weights from the parent model, by default None.
    atol : float, optional
        Absolute tolerance for linear solver, by default None.
    btol : float, optional
        Relative tolerance for linear solver, by default None.

    Attributes
    ----------
    results : GelbachResults
        Container with all decomposition results including direct, indirect, and total effects.
        Provides access to absolute effects and relative effects via properties.

    References
    ----------
    Gelbach, J. B. (2016). When do covariates matter? And which ones, and how much?
    Journal of Labor Economics, 34(2), 509-543.

    """

    # Core parameters
    decomp_var: str
    coefnames: list[str]
    depvarname: str
    nthreads: int = -1
    x1_vars: list[str] | None = None
    cluster_df: pd.Series | None = None
    combine_covariates: dict[str, list[str] | re.Pattern[str]] | None = None
    agg_first: bool | None = False
    only_coef: bool = False
    inference: DecompositionInference = "analytic"
    vcov_config: GelbachVcovConfig = field(default_factory=GelbachVcovConfig)
    weights_type: str | None = None
    atol: float | None = None
    btol: float | None = None

    # Define attributes initialized post-creation
    unique_clusters: np.ndarray | None = field(init=False, default=None)
    mask: np.ndarray = field(init=False)
    mediator_names: list[str] = field(init=False)
    _combine_covariate_indices: dict[str, list[int]] = field(
        init=False, default_factory=dict
    )
    X_dict: dict[Any, Any] = field(init=False, default_factory=dict)
    Y_dict: dict[Any, Any] = field(init=False, default_factory=dict)

    def __post_init__(self):
        if (
            self.inference == "analytic"
            and not self.only_coef
            and self.vcov_config.family == "CRV"
            and self.vcov_config.cluster_prep is None
        ):
            raise ValueError("CRV1 inference requires prepared cluster state.")

        self._check_covariates()

        x1_variables = (
            [self.decomp_var]
            if self.x1_vars is None
            else [self.decomp_var, *self.x1_vars]
        )

        x1_set = set(x1_variables)
        self.mask = np.array([name not in x1_set for name in self.coefnames])
        self.mediator_names = [
            name
            for name, is_mediator in zip(self.coefnames, self.mask, strict=True)
            if is_mediator
        ]

        # Handle clustering setup if cluster bootstrap is requested
        if (
            self.cluster_df is not None
            and not self.only_coef
            and self.inference == "bootstrap"
        ):
            self.unique_clusters = self.cluster_df.unique()
        else:
            self.unique_clusters = None

        if self.combine_covariates is not None and not self.agg_first:
            warnings.warn(
                "You have provided combine_covariates, but agg_first is False. "
                "We recommend setting agg_first=True as this might massively "
                "decrease the computation time (in particular when "
                "bootstrapping CIs)."
            )

        (
            self.combine_covariates_dict,
            self._combine_covariate_indices,
        ) = _normalize_mediator_groups(
            combine_covariates=self.combine_covariates,
            coefficient_names=self.coefnames,
            mediator_names=self.mediator_names,
            x1_vars=self.x1_vars,
        )

    @property
    def vcov(self) -> str:
        "Return the canonical analytical covariance estimator."
        return self.vcov_config.detail

    def _check_covariates(self):
        if self.decomp_var not in self.coefnames:
            raise ValueError(
                f"The decomposition variable '{self.decomp_var}' is not in the coefficient names."
            )
        if self.x1_vars is not None:
            for var in self.x1_vars:
                if var not in self.coefnames:
                    raise ValueError(
                        f"The variable '{var}' is not in the coefficient names."
                    )
        if self.x1_vars is not None and self.decomp_var in self.x1_vars:
            raise ValueError(
                f"The decomposition variable '{self.decomp_var}' cannot be included in the x1_vars argument."
            )

    def fit(
        self,
        X: spmatrix,
        Y: np.ndarray,
        weights: np.ndarray | None = None,
        store: bool = True,
    ):
        "Fit Linear Mediation Model."
        if store:
            if weights is not None:
                self._weights_sqrt = np.sqrt(weights.flatten())
                self.X = diags(self._weights_sqrt, 0) @ X
                self.Y = Y * self._weights_sqrt
                if self.weights_type == "fweights":
                    N = float(np.sum(weights))
                    if N.is_integer():
                        self.N = int(N)
                    else:
                        raise ValueError(
                            "The sum of weights is not an integer, which is not "
                            "supported with frequency weights."
                        )
                else:
                    self.N = X.shape[0]
            else:
                self._weights_sqrt = np.ones(X.shape[0])
                self.N = X.shape[0]
                self.X = X
                self.Y = Y

            self.X1 = hstack([self._weights_sqrt[:, None], self.X[:, ~self.mask]])
            self.X2 = self.X[:, self.mask]

            self.names_X1 = ["Intercept", self.decomp_var]
            if self.x1_vars is not None:
                self.names_X1 += self.x1_vars
            self.names_X = list(self.coefnames)
            self.decomp_var_in_X1_idx = self.names_X1.index(self.decomp_var)
            self.decomp_var_in_X_idx = self.names_X.index(self.decomp_var)

            compute_analytic_state = not self.only_coef and self.inference == "analytic"
            computation = self.compute_gelbach(
                X1=self.X1,
                X2=self.X2,
                Y=self.Y,
                X=self.X,
                agg_first=self.agg_first,
                compute_analytic_state=compute_analytic_state,
            )
            self.results = computation.results

            if not self.only_coef and self.inference == "analytic":
                if computation.core_state is None:
                    raise RuntimeError("Analytic inference state was not computed.")
                self._compute_analytic_inference(state=computation.core_state)

            # Prepare cluster bootstrap if relevant
            self.X_dict = {}
            self.Y_dict = {}

            if (
                self.unique_clusters is not None
                and not self.only_coef
                and self.inference == "bootstrap"
            ):
                for g in self.unique_clusters:
                    cluster_idx = np.where(self.cluster_df == g)[0]
                    self.X_dict[g] = self.X[cluster_idx]
                    self.Y_dict[g] = self.Y[cluster_idx]

            return self.results

        else:
            # need to compute X1, X2 in bootstrap sample

            X1 = hstack([np.ones((X.shape[0], 1)), X[:, ~self.mask]])
            X2 = X[:, self.mask]

            computation = self.compute_gelbach(
                X1=X1,
                X2=X2,
                Y=Y,
                X=X,
                agg_first=self.agg_first,
                compute_analytic_state=False,
            )

            return computation.results

    def bootstrap(self, rng: np.random.Generator, B: int = 1_000, alpha: float = 0.05):
        "Bootstrap Confidence Intervals for Total, Mediated and Direct Effects."
        self.alpha = alpha
        self.B = B

        # convert to csr for easier vstacking
        if self.unique_clusters is not None:
            self.X_dict = {g: self.X_dict[g].tocsr() for g in self.X_dict}

        _bootstrapped = Parallel(n_jobs=self.nthreads)(
            delayed(self._bootstrap)(rng=rng) for _ in tqdm(range(B))
        )

        # unpack
        (
            self._bootstrap_absolute_df,
            self._bootstrap_relative_explained_df,
            self._bootstrap_relative_direct_df,
        ) = self._unpack_bootstrap_results(_bootstrapped)

        # compute ci
        self._absolute_ci = self._compute_bootstrap_ci(
            self._bootstrap_absolute_df, alpha
        )
        self._relative_explained_ci = self._compute_bootstrap_ci(
            self._bootstrap_relative_explained_df, alpha
        )
        self._relative_direct_ci = self._compute_bootstrap_ci(
            self._bootstrap_relative_direct_df, alpha
        )

    def _compute_bootstrap_ci(
        self, bootstrap_df: pd.DataFrame, alpha: float
    ) -> pd.DataFrame:
        """Compute percentile confidence intervals from bootstrap replications.

        Parameters
        ----------
        bootstrap_df : pd.DataFrame
            DataFrame with bootstrap replications (rows) and effects (columns).
        alpha : float
            Significance level for confidence intervals.

        Returns
        -------
        pd.DataFrame
            DataFrame with ci_lower and ci_upper columns.
        """
        ci_df = pd.DataFrame(
            {
                "ci_lower": np.percentile(bootstrap_df, 100 * (alpha / 2), axis=0),
                "ci_upper": np.percentile(bootstrap_df, 100 * (1 - alpha / 2), axis=0),
            },
            index=bootstrap_df.columns,
        )
        return ci_df.astype(float)

    def _compute_analytic_inference(self, *, state: GelbachCoreState) -> None:
        """Compute analytical covariance matrices for Gelbach effects."""
        if self.vcov_config.family == "CRV":
            core_vcov, self._analytic_df = _crv1_gelbach_vcov(
                scores=state.scores,
                config=self.vcov_config,
            )
        else:
            if self.vcov_config.df is None or self.vcov_config.ssc is None:
                raise RuntimeError("Non-clustered inference requires SSC and df state.")
            self._analytic_df = self.vcov_config.df
            ssc = self.vcov_config.ssc
            if self.vcov_config.family == "iid":
                core_vcov = ssc * _iid_gelbach_vcov(state=state, nobs=self.N)
            else:
                scores = state.scores
                if self.weights_type == "fweights":
                    scores = scores / np.asarray(self._weights_sqrt)[:, None]
                core_vcov = ssc * (scores.T @ scores)

        absolute_names = list(self.results.absolute)
        transform = _absolute_transform(
            absolute_names=absolute_names,
            group_names=state.group_names,
            remainder_col=state.remainder_col,
        )
        absolute_vcov = transform @ core_vcov @ transform.T
        absolute_vcov = (absolute_vcov + absolute_vcov.T) / 2
        self._absolute_vcov = pd.DataFrame(
            absolute_vcov, index=absolute_names, columns=absolute_names
        )

        self._relative_explained_vcov = self._relative_vcov(
            self.results.absolute,
            self._absolute_vcov,
            denominator_name="explained_effect",
        )
        self._relative_direct_vcov = self._relative_vcov(
            self.results.absolute,
            self._absolute_vcov,
            denominator_name="direct_effect",
        )

    def _relative_vcov(
        self,
        estimates: dict[str, float],
        absolute_vcov: pd.DataFrame,
        denominator_name: str,
    ) -> pd.DataFrame:
        """Apply the multivariate delta method to relative Gelbach effects."""
        estimates_series = pd.Series(estimates, dtype=float)
        names = estimates_series.index
        denominator = estimates_series[denominator_name]
        if denominator == 0:
            return pd.DataFrame(np.nan, index=names, columns=names)

        denominator_idx = names.get_loc(denominator_name)
        jacobian = np.eye(len(names)) / denominator
        jacobian[:, denominator_idx] -= estimates_series.to_numpy() / denominator**2
        vcov = absolute_vcov.loc[names, names].to_numpy()
        relative_vcov = jacobian @ vcov @ jacobian.T
        relative_vcov = (relative_vcov + relative_vcov.T) / 2

        return pd.DataFrame(relative_vcov, index=names, columns=names)

    def _analytic_inference_df(
        self,
        estimates: dict[str, float],
        vcov: pd.DataFrame,
        alpha: float,
    ) -> pd.DataFrame:
        """Create analytical SE and CI columns from a covariance matrix."""
        estimates_series = pd.Series(estimates, dtype=float)
        vcov = vcov.loc[estimates_series.index, estimates_series.index]
        variance = np.diag(vcov).copy()
        scale = max(1.0, float(np.max(np.abs(variance), initial=0.0)))
        tolerance = 100 * np.finfo(float).eps * scale
        roundoff_negative = (variance < 0) & (variance >= -tolerance)
        variance[roundoff_negative] = 0.0
        with np.errstate(invalid="ignore"):
            standard_errors = np.sqrt(variance)
        std_error = pd.Series(
            standard_errors, index=estimates_series.index, dtype=float
        )
        crit = t.ppf(1 - alpha / 2, self._analytic_df)

        return pd.DataFrame(
            {
                "std_error": std_error,
                "ci_lower": estimates_series - crit * std_error,
                "ci_upper": estimates_series + crit * std_error,
            },
            index=estimates_series.index,
        )

    def _unpack_bootstrap_results(
        self, bootstrapped: list
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Unpack bootstrap results into DataFrames for different effect types.

        Parameters
        ----------
        bootstrapped : list
            List of GelbachResults from bootstrap iterations.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            DataFrames for absolute, relative_to_explained, and relative_to_direct effects.
        """
        absolute_df = pd.DataFrame([res.absolute for res in bootstrapped])
        relative_explained_df = pd.DataFrame(
            [res.relative_to_explained for res in bootstrapped]
        )
        relative_direct_df = pd.DataFrame(
            [res.relative_to_direct for res in bootstrapped]
        )
        return absolute_df, relative_explained_df, relative_direct_df

    def _bootstrap(self, rng: np.random.Generator):
        "Run a single bootstrap iteration."
        if self.unique_clusters is not None:
            idx_clusters = rng.choice(
                self.unique_clusters, len(self.unique_clusters), replace=True
            ).tolist()

            X = vstack([self.X_dict[g].tocsr() for g in idx_clusters])
            Y = np.concatenate([self.Y_dict[g] for g in idx_clusters])

        else:
            idx_rows: NDArray[np.int_] = rng.choice(self.N, self.N)
            X = self.X.tocsr()[idx_rows, :]
            Y = self.Y[idx_rows]

        return self.fit(X=X, Y=Y, store=False)

    def compute_gelbach(
        self,
        X1: spmatrix,
        X2: spmatrix,
        Y: np.ndarray,
        X: spmatrix,
        agg_first: bool | None,
        compute_analytic_state: bool = False,
    ) -> GelbachComputation:
        "Run the Gelbach decomposition."
        # Compute direct effect
        beta_short = lsqr(X1, Y, atol=self.atol, btol=self.btol)[0]

        # Compute beta_full and beta2
        beta_full = lsqr(X, Y, atol=self.atol, btol=self.btol)[0]
        beta2 = beta_full[self.mask]

        core_state = None
        delta = None
        group_delta = None

        if compute_analytic_state:
            x1_inv = pinv(
                np.asarray((X1.T @ X1).toarray(), dtype=float), check_finite=False
            )
            x_inv = pinv(
                np.asarray((X.T @ X).toarray(), dtype=float), check_finite=False
            )
            gamma_matrix = x1_inv @ np.asarray((X1.T @ X2).toarray(), dtype=float)
            gamma = gamma_matrix[self.decomp_var_in_X1_idx, :]

            delta = gamma * beta2
            core_state = _build_gelbach_core_state(
                X=X,
                X1=X1,
                X2=X2,
                Y=Y,
                beta_full=beta_full,
                beta2=beta2,
                x1_inv=x1_inv,
                x_inv=x_inv,
                gamma_matrix=gamma_matrix,
                mask=self.mask,
                decomp_var_in_X1_idx=self.decomp_var_in_X1_idx,
                decomp_var_in_X_idx=self.decomp_var_in_X_idx,
                mediator_names=self.mediator_names,
                reported_groups=self._combine_covariate_indices,
            )
        elif agg_first:
            H = X2.multiply(beta2).tocsc()
            membership = _sparse_grouped_values(
                np.ones(len(beta2)),
                list(self._combine_covariate_indices.values()),
            )
            Hg = H @ membership

            group_delta = np.array(
                [
                    lsqr(X1, Hg.getcol(j).toarray().ravel())[0][
                        self.decomp_var_in_X1_idx
                    ]
                    for j in range(Hg.shape[1])
                ]
            )
        else:
            gamma = np.array(
                [
                    lsqr(X1, X2[:, j].toarray().flatten())[0][self.decomp_var_in_X1_idx]
                    for j in range(X2.shape[1])
                ]
            )

            delta = gamma * beta2

        if group_delta is not None:
            mediator_effects = {
                name: float(group_delta[i])
                for i, name in enumerate(self._combine_covariate_indices)
            }
        else:
            if delta is None:
                raise RuntimeError("Gelbach mediator deltas were not computed.")
            mediator_effects = {
                name: float(np.sum(delta[variable_idx]))
                for name, variable_idx in self._combine_covariate_indices.items()
            }

        direct_effect = float(beta_short[self.decomp_var_in_X1_idx])
        full_effect = float(beta_full[self.decomp_var_in_X_idx])
        explained_effect = sum(mediator_effects.values())
        unexplained_effect = direct_effect - explained_effect

        gelbach_results = GelbachResults(
            direct_effect=direct_effect,
            full_effect=full_effect,
            explained_effect=explained_effect,
            unexplained_effect=unexplained_effect,
            mediator_effects=mediator_effects,
        )

        return GelbachComputation(
            results=gelbach_results,
            core_state=core_state,
        )

    def tidy(self, alpha: float = 0.05, panels: str = "all") -> pd.DataFrame:
        """
        Tidy the Gelbach decomposition output into a DataFrame.

        Return a tidy pd.DataFrame with the decomposition results, including
        point estimates and confidence intervals.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence intervals, by default 0.05.
            Computes a 95% confidence interval when alpha = 0.05.
        panels : str, optional
            Which panels to include. One of 'all', 'levels', 'share_explained',
            'share_full', by default "all". Also accepts full names for backward compatibility.

        Returns
        -------
        pd.DataFrame
            A tidy DataFrame with the decomposition results.
        """
        panel_specs = (
            (
                "Levels (units)",
                self.results.absolute,
                "_absolute_vcov",
                "_absolute_ci",
            ),
            (
                "Share of Full Effect",
                self.results.relative_to_direct,
                "_relative_direct_vcov",
                "_relative_direct_ci",
            ),
            (
                "Share of Explained Effect",
                self.results.relative_to_explained,
                "_relative_explained_vcov",
                "_relative_explained_ci",
            ),
        )
        panel_frames = {}
        for panel_name, estimates, vcov_attr, ci_attr in panel_specs:
            frame = (
                pd.Series(estimates, name="coefficients", dtype=float)
                .rename_axis("effect")
                .to_frame()
            )
            if not self.only_coef:
                inference_frame = (
                    self._analytic_inference_df(
                        estimates,
                        getattr(self, vcov_attr),
                        alpha,
                    )
                    if self.inference == "analytic"
                    else getattr(self, ci_attr)
                )
                frame = pd.concat([frame, inference_frame], axis=1)
            frame["panels"] = panel_name
            panel_frames[panel_name] = frame

        normalized_panels = PANEL_ALIASES.get(panels, panels)

        if panels == "all":
            return pd.concat(panel_frames.values(), axis=0)
        if normalized_panels in panel_frames:
            return panel_frames[normalized_panels]

        valid_options = ["all", *PANEL_ALIASES.keys(), *PANEL_ALIASES.values()]
        raise ValueError(
            f"The 'panels' parameter must be one of {valid_options}. Got '{panels}'."
        )

    def _build_panel_summary(
        self, panel_df: pd.DataFrame, panel_name: str, digits: int
    ) -> pd.DataFrame:
        """Build summary DataFrame for a single panel."""
        summary_data = {}

        summary_data[self.decomp_var] = self._format_main_effects_row(panel_df, digits)

        has_se = "std_error" in panel_df.columns
        if has_se:
            summary_data[f"{self.decomp_var}_se"] = self._format_main_effects_se_row(
                panel_df, digits
            )

        if not self.only_coef:
            summary_data[f"{self.decomp_var}_ci"] = self._format_main_effects_ci_row(
                panel_df, digits
            )

        for mediator in self.combine_covariates_dict:
            if mediator in panel_df.index:
                summary_data[mediator] = self._format_mediator_row(
                    panel_df, mediator, digits
                )
                if has_se:
                    summary_data[f"{mediator}_se"] = self._format_mediator_se_row(
                        panel_df, mediator, digits
                    )
                if not self.only_coef and "ci_lower" in panel_df.columns:
                    summary_data[f"{mediator}_ci"] = self._format_mediator_ci_row(
                        panel_df, mediator, digits
                    )

        summary_data = self._apply_panel_specific_rules(summary_data, panel_name)

        return self._convert_to_dataframe(summary_data)

    def _format_main_effects_row(
        self, panel_df: pd.DataFrame, digits: int
    ) -> dict[str, str]:
        """Format the main decomp_var effects row."""
        return {
            effect: self._format_effect_value(panel_df, effect, digits)
            for effect in ["direct_effect", "full_effect", "explained_effect"]
        }

    def _format_main_effects_se_row(
        self, panel_df: pd.DataFrame, digits: int
    ) -> dict[str, str]:
        """Format the standard error row for main effects."""
        return {
            effect: self._format_se_value(panel_df, effect, digits)
            for effect in ["direct_effect", "full_effect", "explained_effect"]
        }

    def _format_main_effects_ci_row(
        self, panel_df: pd.DataFrame, digits: int
    ) -> dict[str, str]:
        """Format the CI row for main effects."""
        return {
            effect: self._format_ci_value(panel_df, effect, digits)
            for effect in ["direct_effect", "full_effect", "explained_effect"]
        }

    def _format_mediator_row(
        self, panel_df: pd.DataFrame, mediator: str, digits: int
    ) -> dict[str, str]:
        """Format a mediator effects row."""
        coef = panel_df.loc[mediator, "coefficients"]
        return {
            "direct_effect": "-",
            "full_effect": "-",
            "explained_effect": f"{coef:.{digits}f}",
        }

    def _format_mediator_se_row(
        self, panel_df: pd.DataFrame, mediator: str, digits: int
    ) -> dict[str, str]:
        """Format a mediator standard error row."""
        se_str = self._format_se_value(panel_df, mediator, digits)
        return {
            "direct_effect": "-",
            "full_effect": "-",
            "explained_effect": se_str,
        }

    def _format_mediator_ci_row(
        self, panel_df: pd.DataFrame, mediator: str, digits: int
    ) -> dict[str, str]:
        """Format a mediator CI row."""
        ci_str = self._format_ci_value(panel_df, mediator, digits)
        return {
            "direct_effect": "-",
            "full_effect": "-",
            "explained_effect": ci_str,
        }

    def _format_effect_value(
        self, panel_df: pd.DataFrame, effect: str, digits: int
    ) -> str:
        """Format a single effect value."""
        if effect in panel_df.index:
            coef = panel_df.loc[effect, "coefficients"]
            return f"{coef:.{digits}f}"
        return "-"

    def _format_se_value(self, panel_df: pd.DataFrame, effect: str, digits: int) -> str:
        """Format a standard error value."""
        if effect in panel_df.index and "std_error" in panel_df.columns:
            se = panel_df.loc[effect, "std_error"]
            return f"({se:.{digits}f})"
        return "-"

    def _format_ci_value(self, panel_df: pd.DataFrame, effect: str, digits: int) -> str:
        """Format a confidence interval value."""
        if (
            effect in panel_df.index
            and not self.only_coef
            and "ci_lower" in panel_df.columns
        ):
            ci_lower = panel_df.loc[effect, "ci_lower"]
            ci_upper = panel_df.loc[effect, "ci_upper"]
            return f"[{ci_lower:.{digits}f}, {ci_upper:.{digits}f}]"
        return "-"

    def _apply_panel_specific_rules(self, summary_data: dict, panel_name: str) -> dict:
        """Apply panel-specific formatting rules."""
        if panel_name == "Share of Full Effect" and not self.only_coef:
            # Don't print CIs as they are [1,1]
            if f"{self.decomp_var}_se" in summary_data:
                summary_data[f"{self.decomp_var}_se"]["direct_effect"] = "-"
            if f"{self.decomp_var}_ci" in summary_data:
                summary_data[f"{self.decomp_var}_ci"]["direct_effect"] = "-"
        elif panel_name == "Share of Explained Effect":
            summary_data[self.decomp_var]["direct_effect"] = "-"
            summary_data[self.decomp_var]["full_effect"] = "-"
            # Remove inference rows entirely
            if not self.only_coef:
                summary_data.pop(f"{self.decomp_var}_se", None)
                summary_data.pop(f"{self.decomp_var}_ci", None)

        return summary_data

    def _convert_to_dataframe(self, summary_data: dict) -> pd.DataFrame:
        """Convert summary data dict to DataFrame with proper formatting."""
        df = pd.DataFrame(summary_data).T
        df.columns = ["direct_effect", "full_effect", "explained_effect"]

        # Clean up inference row names
        df.index = pd.Index(
            [
                "" if name.endswith("_ci") or name.endswith("_se") else name
                for name in df.index
            ]
        )

        return df

    def etable(
        self,
        panels: str = "all",
        caption: str | None = None,
        column_heads: list[str] | None = None,
        panel_heads: list[str] | None = None,
        rgroup_sep: str | None = None,
        add_notes: str | None = None,
        **kwargs,
    ) -> pd.DataFrame | str | None:
        """
        Generate a table summarizing the Gelbach decomposition results.

        Supports various output formats including html (via great tables), markdown, and LaTeX.

        Parameters
        ----------
        panels : str, optional
            Which panels to include. One of 'all', 'levels', 'share_full', 'share_explained'.
        caption : str, optional
            Caption for the table, by default None.
        column_heads : list[str], optional
            Column names for the table. Must be length 3 if provided, by default None.
        panel_heads : list[str], optional
            Custom names for the panel sections. Length must match number of panels shown, by default None.
        rgroup_sep : str, optional
            Row group separator style. Options: 'tb', 't', 'b', '', by default "t".
        add_notes : str, optional
            Additional notes to append to the table, by default None.
        **kwargs : dict, optional
            Additional arguments passed to maketables.MTable (type, digits, etc.).

        Returns
        -------
        Union[pd.DataFrame, str, None]
            Formatted table. Type depends on output format specified in kwargs.

        Examples
        --------
        ```{python}
        import pyfixest as pf
        data = pf.gelbach_data(nobs=500)
        fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)
        gb = fit.decompose(decomp_var = "x1", x1_vars = ["x21"],reps = 10, nthreads = 1)
        gb.etable()
        ```

        We can change the column headers:

        ```{python}
        gb.etable(column_heads = ["Full Difference", "Unexplained Difference", "Explained Difference"])
        """
        from maketables import MTable

        if column_heads is not None and len(column_heads) != 3:
            raise ValueError("The 'column_heads' parameter must be a list of length 3.")

        panels_arg_to_label = PANEL_ALIASES

        if panels == "all":
            panel_list = [
                "Levels (units)",
                "Share of Full Effect",
                "Share of Explained Effect",
            ]
        else:
            panel_list = (
                [panels_arg_to_label[panels]]
                if isinstance(panels, str)
                else [panels_arg_to_label[panel] for panel in panels]
            )

        for panel in panel_list:
            if panel not in [
                "Levels (units)",
                "Share of Full Effect",
                "Share of Explained Effect",
            ]:
                raise ValueError(
                    f"The 'panels' parameter must be one of 'Levels (units)', 'Share of Full Effect', 'Share of Explained Effect'. Got '{panel}'."
                )

        if panel_heads is not None and len(panel_heads) != len(panel_list):
            raise ValueError(
                f"The 'panel_heads' parameter must have length {len(panel_list)} to match the number of panels panels. Got {len(panel_heads)}."
            )

        # Build formatted DataFrame directly
        digits = kwargs.get("digits", 3)  # Default to 3 if not specified
        df = self.tidy(panels="all").round(digits)
        panels_to_include = df["panels"].unique()

        results = {}
        for panel_name in panels_to_include:
            panel_df = df[df["panels"] == panel_name].copy()
            results[panel_name] = self._build_panel_summary(
                panel_df, panel_name, digits
            )

        res = pd.concat(results, axis=0)

        if isinstance(res.index, pd.MultiIndex):
            mask = res.index.get_level_values(0).isin(panel_list)
            res_sub = res.loc[mask, :]
        else:
            res_sub = res

        if self.x1_vars is not None:
            default_model_notes = [
                f"Col 1: Adjusted Difference (by {'+'.join(self.x1_vars)}) - Coefficient on {self.decomp_var} in short regression.",
                f"Col 2: Adjusted Difference - Coefficient on {self.decomp_var} in long regression.",
                f"Col 3: Explained Difference - Difference in coefficients of {self.decomp_var} in short and long regression.",
            ]

        else:
            default_model_notes = [
                f"Col 1: Raw Difference - Coefficient on {self.decomp_var} in short regression .",
                f"Col 2: Adjusted Difference - Coefficient on {self.decomp_var} in long regression.",
                f"Col 3: Explained Difference - Difference in coefficients of {self.decomp_var} in short and long regression.",
            ]

        panel_num = 0
        if "Levels (units)" in panel_list:
            panel_num += 1
            default_model_notes.append(f"Panel {panel_num}: Levels (units).")
        if "Share of Full Effect" in panel_list:
            panel_num += 1
            default_model_notes.append(
                f"Panel {panel_num}: Share of Full Effect: Levels normalized by coefficient of the short regression."
            )
        if "Share of Explained Effect" in panel_list:
            panel_num += 1
            default_model_notes.append(
                f"Panel {panel_num}: Share of Explained Effect: Levels normalized by coefficient of the long regression."
            )

        default_model_heads = [
            "Initial Difference",
            "Adjusted Difference",
            "Explained Difference",
        ]

        res_sub.columns = (
            column_heads if column_heads is not None else default_model_heads
        )

        if panel_heads is not None and isinstance(res_sub.index, pd.MultiIndex):
            panel_mapping = {
                panel_list[i]: panel_heads[i] for i in range(len(panel_list))
            }

            new_index_level_0 = [
                panel_mapping.get(x, x) for x in res_sub.index.get_level_values(0)
            ]
            new_index = pd.MultiIndex.from_arrays(
                [new_index_level_0, res_sub.index.get_level_values(1)],
                names=res_sub.index.names,
            )
            res_sub.index = new_index

        notes = f"""
            Decomposition variable: {self.decomp_var}.
        """

        if self.x1_vars is not None:
            notes += f"""
            Control Variables: {", ".join(self.x1_vars)}.
            """

        if not self.only_coef:
            if self.inference == "analytic":
                vcov_label = {
                    "iid": "IID",
                    "HC1": "HC1",
                    "CRV1": "CRV1",
                }[self.vcov]
                notes += f"""
                Standard errors and CIs use {vcov_label} delta-method analytical inference.
                """
            else:
                notes += f"""
                    CIs are computed using B = {self.B} bootstrap replications
                """
                if self.cluster_df is None:
                    notes += " using iid sampling."
                else:
                    notes += f" using clustered sampling by {self.cluster_df.name}."

        notes += "\n".join(default_model_notes)

        if add_notes is not None:
            notes += f"""
            {add_notes}
            """

        rgroup_sep_val = "t" if rgroup_sep is None else rgroup_sep
        output_type = kwargs.pop("type", "gt")

        table = MTable(
            res_sub,
            notes=notes,
            caption=caption,
            rgroup_sep=rgroup_sep_val,
            **kwargs,
        )

        # Return based on type parameter
        if output_type == "gt":
            return table.make(type="gt")
        elif output_type == "tex":
            return table.make(type="tex")
        elif output_type == "df":
            return table.df
        elif output_type == "md":
            result = table.df.to_markdown()
            print(result)
            return None
        elif output_type == "html":
            return table.make(type="html")
        else:
            return table.make(type="gt")

    def coefplot(
        self,
        annotate_shares: bool = True,
        title: str | None = None,
        figsize: tuple[int, int] | None = None,
        keep: list | str | None = None,
        drop: list | str | None = None,
        exact_match: bool = False,
        labels: dict | None = None,
        notes: str | None = None,
    ):
        """
        Create a waterfall chart showing Gelbach decomposition results.
        The chart shows the transition from the initial difference (direct effect)
        through individual mediator contributions to the full effect, with a spanner
        showing the total explained effect above the mediator bars.

        Parameters
        ----------
        annotate_shares : bool, optional
            Whether to show percentage shares in parentheses. Default True.
        title : Optional[str], optional
            Chart title. If None, uses default title with decomposition variable.
        figsize : Optional[tuple[int, int]], optional
            Figure size (width, height) in inches. Default (12, 8).
        keep : Optional[Union[list, str]], optional
            The pattern for retaining mediator names. You can pass a string (one
            pattern) or a list (multiple patterns). Default is keeping all mediators.
            Uses regular expressions to select mediators. Note: is applied before the
            labels argument.
        drop : Optional[Union[list, str]], optional
            The pattern for excluding mediator names. You can pass a string (one
            pattern) or a list (multiple patterns). Syntax is the same as for `keep`.
            Default is keeping all mediators. Can be used simultaneously with `keep`.
            Note: is applied after the labels argument.
        exact_match : bool, optional
            Whether to use exact match for `keep` and `drop`. Default is False.
            If True, patterns will be matched exactly instead of using regex.
        labels : Optional[dict], optional
            Dictionary to relabel mediator variables. Keys are original names,
            values are new display names. Applied after `keep` and `drop`.
        notes : Optional[str], optional
            Custom notes to display below the chart. If None, shows default
            decomposition information.

        Examples
        --------
        ```python
        import pyfixest as pf

        data = pf.gelbach_data(nobs=500)
        fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)
        gb = fit.decompose(decomp_var="x1", only_coef=True)
        # Basic waterfall chart
        gb.coefplot()
        # Custom labels and styling
        gb.coefplot(
            labels={"x21": "Education", "x22": "Experience", "x23": "Age"},
            figsize=(14, 8),
            notes="Custom decomposition analysis",
        )
        # With filtering
        gb.coefplot(
            keep=["x2.*"],  # Keep only variables starting with x2
            drop=["x23"],  # But exclude x23
            exact_match=False,
        )
        ```
        """
        from pyfixest.report.visualize_decomposition import create_decomposition_plot

        # Get the decomposition data
        df = self.tidy()

        # Call the standalone plotting function
        create_decomposition_plot(
            decomposition_data=df,
            depvarname=self.depvarname,
            decomp_var=self.decomp_var,
            annotate_shares=annotate_shares,
            title=title,
            figsize=figsize,
            keep=keep,
            drop=drop,
            exact_match=exact_match,
            labels=labels,
            notes=notes,
        )


def _decompose_arg_check(
    type: str,
    has_weights: bool,
    weights_type: str | None,
    is_iv: bool,
    method: str,
    only_coef: bool,
    inference: str,
) -> None:
    "Check arguments for decomposition."
    supported_decomposition_types = ["gelbach"]

    if type not in supported_decomposition_types:
        raise ValueError(
            f"'type' {type} is not in supported types {supported_decomposition_types}."
        )

    try:
        _validate_literal_argument(inference, DecompositionInference)
    except ValueError as exc:
        supported = get_args(DecompositionInference)
        raise ValueError(
            f"'inference' must be one of {supported}. Got '{inference}'."
        ) from exc

    if has_weights and weights_type not in {"aweights", "fweights"}:
        raise NotImplementedError(
            "Decomposition is currently only supported for models with analytical "
            "or frequency weights."
        )
    if has_weights and inference == "bootstrap" and not only_coef:
        raise NotImplementedError(
            "Bootstrap decomposition inference is currently not supported for weighted models."
        )

    if is_iv:
        raise NotImplementedError(
            "Decomposition is currently not supported for IV models."
        )

    if method == "fepois":
        raise NotImplementedError(
            "Decomposition is currently not supported for Poisson regression."
        )

    return None
