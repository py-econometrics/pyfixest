import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, diags, hstack, vstack
from scipy.sparse.linalg import lsqr
from tqdm import tqdm

from pyfixest.utils.dev_utils import _find_stack_level

# Panel name mappings for consistent API
PANEL_ALIASES = {
    "levels": "Levels (units)",
    "share_full": "Share of Full Effect",
    "share_explained": "Share of Explained Effect",
}


@dataclass
class GelbachResults:
    """Coefficient decomposition and contributions returned by Gelbach estimation.

    Access as ``gb.results`` after ``gb = fit.decompose(...)``. Field names
    are historical: ``direct_effect`` is the focal coefficient in the short
    regression, while ``full_effect`` is its coefficient in the long regression.

    Attributes
    ----------
    direct_effect : float
        Focal coefficient in the short model, including ``x1_vars`` if supplied.
    full_effect : float
        Focal coefficient in the long model with all covariates.
    explained_effect : float
        Sum of the selected mediator/group contributions. Equals the
        short-minus-long coefficient change when all mediators are included.
    unexplained_effect : float
        ``direct_effect - explained_effect``. Equals the long-model coefficient
        up to numerical tolerance only when all mediators are included.
    mediator_effects : dict[str, float]
        Contributions keyed by covariate or user-defined group name.
    absolute : dict[str, float]
        The four named effects followed by mediator contributions in their units.
    relative_to_explained : dict[str, float]
        All effects divided by ``explained_effect``; all ``NaN`` when it is zero.
    relative_to_direct : dict[str, float]
        All effects divided by the short-model coefficient ``direct_effect``;
        all ``NaN`` when it is zero.
    all_effect_names : list[str]
        Core effect names followed by mediator names, in dictionary order.

    See Also
    --------
    [GelbachDecomposition](/reference/estimation.post_estimation.decomposition.GelbachDecomposition.qmd),
    [decompose()](/reference/estimation.models.feols_.Feols.decompose.qmd).
    """

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
        """Return core effects and mediator contributions in coefficient units.

        Keys are ``direct_effect``, ``full_effect``, ``explained_effect``,
        ``unexplained_effect``, followed by mediator names. A new dictionary is
        returned; the stored result is unchanged.
        """
        return {
            "direct_effect": self.direct_effect,
            "full_effect": self.full_effect,
            "explained_effect": self.explained_effect,
            "unexplained_effect": self.unexplained_effect,
            **self.mediator_effects,
        }

    @property
    def relative_to_explained(self) -> dict[str, float]:
        """Return all effects divided by the sum of included contributions.

        The denominator is ``explained_effect``, the sum of selected mediator/group
        contributions. It equals the full coefficient change only when all
        mediators are included.
        Returns ``NaN`` for every entry when that denominator is zero.
        """
        if self.explained_effect == 0:
            return {name: np.nan for name in self.absolute}
        return {
            name: value / self.explained_effect for name, value in self.absolute.items()
        }

    @property
    def relative_to_direct(self) -> dict[str, float]:
        """Return all effects divided by the short-model focal coefficient.

        The denominator is the historically named ``direct_effect`` field.
        Returns ``NaN`` for every entry when that denominator is zero.
        """
        if self.direct_effect == 0:
            return {name: np.nan for name in self.absolute}
        return {
            name: value / self.direct_effect for name, value in self.absolute.items()
        }

    @property
    def all_effect_names(self) -> list[str]:
        """Return core effect names followed by mediator names in storage order."""
        return list(self.absolute.keys())

    def to_dict(self, relative_to: str | None = None) -> dict[str, float]:
        """Return absolute or normalized decomposition effects as a dictionary.

        Parameters
        ----------
        relative_to : str, optional
            ``None`` returns absolute effects. ``"explained"`` divides by the
            sum of included mediator/group contributions; ``"direct"`` divides
            by the short-model coefficient. Other values raise ``ValueError``.

        Returns
        -------
        dict[str, float]
            Core effects and mediator contributions. Normalization by zero gives
            ``NaN`` for every entry. The stored results are unchanged.

        See Also
        --------
        [GelbachDecomposition.tidy()](/reference/estimation.post_estimation.decomposition.GelbachDecomposition.qmd#pyfixest.estimation.post_estimation.decomposition.GelbachDecomposition.tidy).
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


@dataclass
class GelbachDecomposition:
    """Fitted Gelbach decomposition returned by `fit.decompose()`.

    Access as ``gb = fit.decompose(decomp_var="X", ...)`` on a supported
    OLS result. The method compares the focal coefficient in a short model
    with its coefficient in the original long model. Contributions sum to
    the short-minus-long coefficient change when all mediators are included.
    A partial ``combine_covariates`` mapping reports only the selected groups.
    This algebraic decomposition does not by itself establish causal mediation.

    Use ``gb.tidy()`` for numerical results, ``gb.etable()`` for a formatted
    table, and ``gb.coefplot()`` for a waterfall plot. Construct this object
    through ``fit.decompose()`` rather than fitting its internal arrays directly.

    Parameters
    ----------
    decomp_var : str
        Focal coefficient to decompose.
    coefnames : list[str]
        Names of the long-model design columns, in their array order.
    depvarname : str
        Dependent-variable name for reporting.
    nthreads : int, optional
        Bootstrap workers. Default -1 uses all available cores.
    x1_vars : list[str], optional
        Other regressors retained in both short and long models; default `None`.
    cluster_df : pd.Series, optional
        Bootstrap cluster identifiers aligned with retained rows; default `None`
        uses independent-row resampling.
    combine_covariates : dict[str, list[str]], optional
        Mediator groups; default `None` treats covariates separately.
    agg_first : bool, optional
        Aggregate contributions before auxiliary regressions; default `False`.
    only_coef : bool, optional
        Whether to skip inference. The internal constructor defaults to `True`;
        ``fit.decompose()`` defaults to `False` and runs bootstrap inference.
    atol : float, optional
        Stopping tolerance passed to the short- and long-model least-squares
        solves; default `None`. Auxiliary contribution regressions use the
        solver defaults instead.
    btol : float, optional
        Stopping tolerance passed to the short- and long-model least-squares
        solves; default `None`. Auxiliary contribution regressions use the
        solver defaults instead.

    Attributes
    ----------
    results : GelbachResults
        Access as ``gb.results``. ``direct_effect`` is the short-model coefficient,
        ``full_effect`` is the long-model coefficient, and ``explained_effect``
        sums the selected mediator/group contributions. If all mediators are
        included, the explained effect equals short minus long and
        ``unexplained_effect`` equals the long coefficient up to numerical
        tolerance. For partial groups, unexplained is short minus the selected
        contributions. These names are historical.

    Examples
    --------
    ```{python}
    import pyfixest as pf
    from pyfixest.utils.dgps import gelbach_data

    fit = pf.feols("y ~ x1 + x21 + x22 + x23", gelbach_data(nobs=200))
    gb = fit.decompose(decomp_var="x1", only_coef=True)
    gb.tidy(panels="levels")
    ```

    Notes
    -----
    The ``share_full`` panel divides by the short-model coefficient, despite
    its label "Share of Full Effect". ``share_explained`` divides by the
    sum of included mediator/group contributions. A zero denominator
    produces ``NaN`` shares.
    Inference uses percentile bootstrap intervals when the decomposition was
    created with ``only_coef=False``. Small example replication counts only
    illustrate usage; use more replications for analysis.

    The method follows Gelbach (2016), "When do covariates matter? And which
    ones, and how much?", Journal of Labor Economics 34(2), 509-543,
    [doi:10.1086/683668](https://doi.org/10.1086/683668). Initial implementation
    by [Apoorva Lal](https://gist.github.com/apoorvalal/e7dc9f3e52dcd9d51854b28b3e8a7ba4).

    See Also
    --------
    [decompose()](/reference/estimation.models.feols_.Feols.decompose.qmd),
    [regression decomposition guide](/how-to/regression_decomposition.qmd),
    [fitted-result overview](/reference-guide/fitted-results.qmd).
    """

    # Core parameters
    decomp_var: str
    coefnames: list[str]
    depvarname: str
    nthreads: int = -1
    x1_vars: list[str] | None = None
    cluster_df: pd.Series | None = None
    combine_covariates: dict[str, list[str]] | None = None
    agg_first: bool | None = False
    only_coef: bool = True
    atol: float | None = None
    btol: float | None = None

    # Define attributes initialized post-creation
    cluster_dict: dict[Any, Any] | None = field(init=False, default=None)
    unique_clusters: np.ndarray | None = field(init=False, default=None)
    mask: np.ndarray = field(init=False)
    mediator_names: list[str] = field(init=False)
    X_dict: dict[Any, Any] = field(init=False, default_factory=dict)
    X1_dict: dict[Any, Any] = field(init=False, default_factory=dict)
    X2_dict: dict[Any, Any] = field(init=False, default_factory=dict)
    Y_dict: dict[Any, Any] = field(init=False, default_factory=dict)

    def __post_init__(self):
        x1_variables = (
            [self.decomp_var]
            if self.x1_vars is None
            else [self.decomp_var, *self.x1_vars]
        )

        # build index for all variables in X1: decomp_var, x1_vars
        x1_indices = [self.coefnames.index(var) for var in x1_variables]
        self.mask = np.ones(len(self.coefnames), dtype=bool)
        self.mask[x1_indices] = False

        self.mediator_names = [
            name for name in self.coefnames if self.mask[self.coefnames.index(name)]
        ]

        # Handle clustering setup if cluster_df is provided
        if self.cluster_df is not None and not self.only_coef:
            self.unique_clusters = cast(np.ndarray, self.cluster_df.unique())
            self.cluster_dict = {
                cluster: self.cluster_df[self.cluster_df == cluster].index
                for cluster in self.unique_clusters
            }
        else:
            self.unique_clusters = None
            self.cluster_dict = None

        if self.combine_covariates is None:
            self.combine_covariates_dict = {
                x: [x] for x in self.mediator_names if x != "Intercept"
            }
        else:
            self.combine_covariates_dict = self.combine_covariates

        if self.combine_covariates is not None and not self.agg_first:
            warnings.warn(
                "You have provided combine_covariates, but agg_first is False. We recommend setting agg_first=True as this might massively decrease the computation time (in particular when boostrapping CIs).",
                UserWarning,
                stacklevel=_find_stack_level(),
            )

        self._check_covariates()
        self._check_combine_covariates()

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

        # Check x1_vars don't overlap with combine_covariates keys
        if self.x1_vars is not None and self.combine_covariates is not None:
            combine_values = set(
                list(itertools.chain.from_iterable(self.combine_covariates.values()))
            )
            x1_set = set(self.x1_vars)
            overlap = x1_set & combine_values
            if overlap:
                raise ValueError(
                    f"Variables {sorted(overlap)} cannot be in both x1_vars and combine_covariates keys."
                )

    def _check_combine_covariates(self):
        # Check that each value in self.combine_covariates_dict is in self.mediator_names
        for _, values in self.combine_covariates_dict.items():
            if not isinstance(values, list):
                raise TypeError("Values in combine_covariates_dict must be lists.")
            for v in values:
                if v not in self.mediator_names:
                    raise ValueError(
                        f"The variable '{v}' is not in the mediator names."
                    )

        # Check for overlap in values between different keys
        all_values = {
            k: set([v] if isinstance(v, str) else v)
            for k, v in self.combine_covariates_dict.items()
        }
        for key1, values1 in all_values.items():
            for key2, values2 in all_values.items():
                if key1 != key2 and values1 & values2:
                    overlap = values1 & values2
                    raise ValueError(
                        f"Variables {overlap} are in both '{key1}' and '{key2}' groups."
                    )

    def fit(
        self,
        X: csc_matrix,
        Y: np.ndarray,
        weights: np.ndarray | None = None,
        store: bool = True,
    ):
        "Fit Linear Mediation Model."
        if store:
            self.X = X
            self.Y = Y

            self.X1 = self.X[:, ~self.mask]
            self.X1 = hstack([np.ones((self.X1.shape[0], 1)), self.X1])
            self.X2 = self.X[:, self.mask]

            if weights is not None:
                weights_sqrt = np.sqrt(weights.flatten())
                S = diags(weights_sqrt, 0)
                self.X = S @ self.X
                self.X1 = S @ self.X1
                self.X2 = S @ self.X2
                self.Y = self.Y * weights_sqrt
                # treat weights as frequency weights
                N = np.sum(weights)
                if N.is_integer():
                    self.N = int(N)
                else:
                    raise ValueError(
                        "The sum of weights is not an integer, which is not supported with frequency weights."
                    )
            else:
                self.N = X.shape[0]

            self.names_X1 = ["Intercept", self.decomp_var]
            if self.x1_vars is not None:
                self.names_X1 += self.x1_vars
            self.names_X = list(self.coefnames)
            self.decomp_var_in_X1_idx = self.names_X1.index(self.decomp_var)
            self.decomp_var_in_X_idx = self.names_X.index(self.decomp_var)

            results = self.compute_gelbach(
                X1=self.X1,
                X2=self.X2,
                Y=self.Y,
                X=self.X,
                agg_first=self.agg_first,
            )

            (
                self.direct_effect,
                self.beta_full,
                self.beta2,
                self.results,
            ) = results

            # Prepare cluster bootstrap if relevant
            self.X_dict = {}
            self.Y_dict = {}

            if self.unique_clusters is not None and not self.only_coef:
                for g in self.unique_clusters:
                    cluster_idx = np.where(self.cluster_df == g)[0]
                    self.X_dict[g] = self.X[cluster_idx]
                    self.Y_dict[g] = self.Y[cluster_idx]

            return self.results

        else:
            # need to compute X1, X2 in bootstrap sample

            X1 = hstack([np.ones((X.shape[0], 1)), X[:, ~self.mask]])
            X2 = X[:, self.mask]

            results = self.compute_gelbach(
                X1=X1,
                X2=X2,
                Y=Y,
                X=X,
                agg_first=self.agg_first,
            )

            (
                _,
                _,
                _,
                bootstrap_results,
            ) = results

            return bootstrap_results

    def bootstrap(self, rng: np.random.Generator, B: int = 1_000):
        """Draw bootstrap replications for Total, Mediated and Direct Effects."""
        self.B = B

        # convert to csr for easier vstacking
        if self.unique_clusters is not None:
            self.X_dict = {g: self.X_dict[g].tocsr() for g in self.X_dict}

        _bootstrapped = Parallel(n_jobs=self.nthreads)(
            delayed(self._bootstrap)(rng=child_rng) for child_rng in tqdm(rng.spawn(B))
        )

        # unpack
        (
            self._bootstrap_absolute_df,
            self._bootstrap_relative_explained_df,
            self._bootstrap_relative_direct_df,
        ) = self._unpack_bootstrap_results(_bootstrapped)

    def _compute_ci(self, bootstrap_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
        """Compute confidence intervals from bootstrap DataFrame.

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
        X1: csc_matrix,
        X2: csc_matrix,
        Y: np.ndarray,
        X: csc_matrix,
        agg_first: bool | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        GelbachResults,
    ]:
        "Run the Gelbach decomposition."
        N = X1.shape[0]

        # Compute direct effect
        direct_effect = lsqr(X1, Y, atol=self.atol, btol=self.btol)[0]
        direct_effect_array = np.array([direct_effect[self.decomp_var_in_X1_idx]])

        # Compute beta_full and beta2
        beta_full = lsqr(X, Y, atol=self.atol, btol=self.btol)[0]
        beta2 = beta_full[self.mask]

        mediator_effects = {}

        if agg_first:
            # Compute H and Hg
            H = X2.multiply(beta2).tocsc()  # csc better for slicing columns than csr
            Hg = np.zeros((N, len(self.combine_covariates_dict)))

            for i, (_, covariates) in enumerate(self.combine_covariates_dict.items()):
                variable_idx = [self.mediator_names.index(cov) for cov in covariates]
                Hg[:, i] = np.sum(H[:, variable_idx], axis=1).flatten()

            # Compute delta
            delta = np.array(
                [
                    lsqr(X1, Hg[:, j])[0][self.decomp_var_in_X1_idx]
                    for j in range(Hg.shape[1])
                ]
            )

            for i, (name, _) in enumerate(self.combine_covariates_dict.items()):
                mediator_effects[name] = float(delta[i])
        else:
            gamma = np.array(
                [
                    lsqr(X1, X2[:, j].toarray().flatten())[0][self.decomp_var_in_X1_idx]
                    for j in range(X2.shape[1])
                ]
            )

            delta = gamma * beta2

            for name, covariates in self.combine_covariates_dict.items():
                variable_idx = [self.mediator_names.index(cov) for cov in covariates]
                mediator_effects[name] = float(np.sum(delta[variable_idx]))

        direct_effect = float(direct_effect_array[0])
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

        results = (
            direct_effect_array,
            beta_full,
            beta2,
            gelbach_results,
        )

        return results

    def _dict_to_df(self, data: dict[str, float]) -> pd.DataFrame:
        """Convert a mapping of effects to a tidy 2-column DataFrame.

        Returns a DataFrame with index 'effect' and a single column 'coefficients'.
        """
        return pd.DataFrame(
            list(data.items()), columns=["effect", "coefficients"]
        ).set_index("effect")

    def tidy(self, alpha: float = 0.05, panels: str = "all") -> pd.DataFrame:
        """Return decomposition estimates and optional bootstrap intervals.

        Parameters
        ----------
        alpha : float, optional
            Significance level, default 0.05. With inference, must lie in (0, 1)
            and yields percentile bootstrap bounds at level ``1 - alpha``.
            Ignored for ``only_coef=True`` results.
        panels : str, optional
            ``"all"`` (default), ``"levels"``, ``"share_full"``, or
            ``"share_explained"``. Full panel labels are also accepted. The
            ``share_full`` denominator is the short-model coefficient;
            ``share_explained`` uses the sum of included mediator/group
            contributions.

        Returns
        -------
        pandas.DataFrame
            Index named ``effect`` contains four core effects and then mediators.
            Columns are ``coefficients`` and ``panels``, plus ``ci_lower`` and
            ``ci_upper`` when bootstrap inference was run. ``"all"`` stacks levels,
            shares of the short-model coefficient, then shares of the included
            contribution sum; the effect index repeats across panels. Shares are
            fractions, not percentages. The stored result is unchanged.

        Examples
        --------
        ```{python}
        import pyfixest as pf
        from pyfixest.utils.dgps import gelbach_data

        fit = pf.feols("y ~ x1 + x21 + x22 + x23", gelbach_data(nobs=200))
        gb = fit.decompose(decomp_var="x1", only_coef=True)
        gb.tidy(panels="levels")
        ```

        See Also
        --------
        [decompose()](/reference/estimation.models.feols_.Feols.decompose.qmd),
        [regression decomposition guide](/how-to/regression_decomposition.qmd).
        """
        absolute_df = self._dict_to_df(self.results.absolute)
        relative_explained_df = self._dict_to_df(self.results.relative_to_explained)
        relative_direct_df = self._dict_to_df(self.results.relative_to_direct)

        if not self.only_coef:
            if not 0 < alpha < 1:
                raise ValueError(f"alpha must be in (0, 1). Got {alpha}.")
            absolute_df = pd.concat(
                [absolute_df, self._compute_ci(self._bootstrap_absolute_df, alpha)],
                axis=1,
            )
            relative_explained_df = pd.concat(
                [
                    relative_explained_df,
                    self._compute_ci(self._bootstrap_relative_explained_df, alpha),
                ],
                axis=1,
            )
            relative_direct_df = pd.concat(
                [
                    relative_direct_df,
                    self._compute_ci(self._bootstrap_relative_direct_df, alpha),
                ],
                axis=1,
            )

        absolute_df["panels"] = np.repeat("Levels (units)", len(absolute_df))
        relative_explained_df["panels"] = np.repeat(
            "Share of Explained Effect", len(relative_explained_df)
        )
        relative_direct_df["panels"] = np.repeat(
            "Share of Full Effect", len(relative_direct_df)
        )

        normalized_panels = PANEL_ALIASES.get(panels, panels)

        if panels == "all":
            return pd.concat(
                [
                    absolute_df,
                    relative_direct_df,
                    relative_explained_df,
                ],
                axis=0,
            )
        elif normalized_panels == "Levels (units)":
            return absolute_df
        elif normalized_panels == "Share of Explained Effect":
            return relative_explained_df
        elif normalized_panels == "Share of Full Effect":
            return relative_direct_df
        else:
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

        if not self.only_coef:
            summary_data[f"{self.decomp_var}_ci"] = self._format_main_effects_ci_row(
                panel_df, digits
            )

        for mediator in self.combine_covariates_dict:
            if mediator in panel_df.index:
                summary_data[mediator] = self._format_mediator_row(
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
            summary_data[f"{self.decomp_var}_ci"]["direct_effect"] = "-"
        elif panel_name == "Share of Explained Effect":
            summary_data[self.decomp_var]["direct_effect"] = "-"
            summary_data[self.decomp_var]["full_effect"] = "-"
            # Remove CIs entirely
            if not self.only_coef:
                summary_data.pop(f"{self.decomp_var}_ci", None)

        return summary_data

    def _convert_to_dataframe(self, summary_data: dict) -> pd.DataFrame:
        """Convert summary data dict to DataFrame with proper formatting."""
        df = pd.DataFrame(summary_data).T
        df.columns = ["direct_effect", "full_effect", "explained_effect"]

        # Clean up CI row names
        df.index = pd.Index(["" if name.endswith("_ci") else name for name in df.index])

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
        """Create a formatted decomposition table with optional share panels.

        Parameters
        ----------
        panels : str, optional
            ``"all"`` (default), ``"levels"``, ``"share_full"``, or
            ``"share_explained"``. A list of the short names is also accepted.
            See ``tidy()`` for the share denominators.
        caption : str, optional
            Table caption; default `None`.
        column_heads : list[str], optional
            Three display headings for short-model coefficient, long-model
            coefficient, and included contribution sum, in that order. `None` uses
            "Initial Difference", "Adjusted Difference", "Explained Difference".
        panel_heads : list[str], optional
            Replacement panel labels, one per displayed panel; default `None`.
        rgroup_sep : str, optional
            Row-group separator passed to ``MTable``. `None` uses ``"t"``; choices
            include ``"tb"``, ``"t"``, ``"b"``, and ``""``.
        add_notes : str, optional
            Text appended to the generated table notes; default `None`.
        **kwargs
            Formatting arguments passed to ``maketables.MTable``, such as ``digits``.
            ``type`` selects output: ``"gt"`` (default), ``"df"``, ``"tex"``,
            ``"html"``, or ``"md"``. Unrecognized types fall back to ``"gt"``.

        Returns
        -------
        object
            ``"gt"`` returns a Great Tables object. ``"df"`` returns the formatted
            `DataFrame`; ``"tex"`` and ``"html"`` return the corresponding MTable
            output. ``"md"`` prints Markdown and returns `None`. Confidence intervals
            use the default 95% level when bootstrap inference was run.

        Examples
        --------
        ```{python}
        import pyfixest as pf
        from pyfixest.utils.dgps import gelbach_data

        fit = pf.feols("y ~ x1 + x21 + x22 + x23", gelbach_data(nobs=200))
        gb = fit.decompose(decomp_var="x1", only_coef=True)
        gb.etable(type="df", panels="levels")
        ```

        ```{python}
        gb.etable(type="df", column_heads=["Short model", "Long model", "Change"])
        ```

        Notes
        -----
        Share denominators follow ``tidy()``. The generated note for the
        "Share of Explained Effect" panel currently names the long regression
        although the values divide by the sum of included contributions; use the
        numerical definition above when interpreting that panel.

        See Also
        --------
        [decompose()](/reference/estimation.models.feols_.Feols.decompose.qmd),
        [regression decomposition guide](/how-to/regression_decomposition.qmd).
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
        """Display a waterfall plot of the coefficient decomposition.

        Displays the short- and long-model coefficients and selected mediator/group
        contributions; a spanner shows their sum. Contributions explain the full
        coefficient change only when all mediators are included.

        Parameters
        ----------
        annotate_shares : bool, optional
            Annotate percentage shares of the short-model coefficient and the
            sum of included contributions, where defined; default `True`.
        title : str, optional
            Plot title; `None` derives a title from the decomposition variable.
        figsize : tuple[int, int], optional
            Matplotlib figure width and height in inches; `None` uses (12, 8).
        keep : list[str] or str, optional
            Patterns selecting mediator names before relabeling; default `None` keeps
            all. Regular expressions are used unless ``exact_match=True``.
        drop : list[str] or str, optional
            Patterns excluding mediator names, after ``keep`` and before relabeling;
            default `None` excludes none. Filtering changes the displayed bars.
        exact_match : bool, optional
            Use exact names for ``keep`` and ``drop`` instead of regex; default `False`.
        labels : dict[str, str], optional
            Display names keyed by original mediator names, applied after filtering.
        notes : str, optional
            Text below the chart; `None` supplies decomposition information.

        Returns
        -------
        None
            Displays a Matplotlib plot via ``plt.show()``. The current figure can
            be obtained with ``matplotlib.pyplot.gcf()`` after the call.

        Examples
        --------
        ```{python}
        import pyfixest as pf
        from pyfixest.utils.dgps import gelbach_data

        fit = pf.feols("y ~ x1 + x21 + x22 + x23", gelbach_data(nobs=200))
        gb = fit.decompose(decomp_var="x1", only_coef=True)
        gb.coefplot(labels={"x21": "Education", "x22": "Experience", "x23": "Age"})
        ```

        See Also
        --------
        [decompose()](/reference/estimation.models.feols_.Feols.decompose.qmd),
        [regression decomposition guide](/how-to/regression_decomposition.qmd).
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
) -> None:
    "Check arguments for decomposition."
    supported_decomposition_types = ["gelbach"]

    if type not in supported_decomposition_types:
        raise ValueError(
            f"'type' {type} is not in supported types {supported_decomposition_types}."
        )

    if has_weights and weights_type != "fweights":
        raise NotImplementedError(
            "Decomposition is currently only supported for models with frequency weights."
        )
    if has_weights and not only_coef:
        raise NotImplementedError(
            "Decomposition is currently only supported for models with frequency weights when only_coef is False."
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
