import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.sparse import hstack, spmatrix, vstack
from scipy.sparse.linalg import lsqr
from tqdm import tqdm

# Panel name mappings for consistent API
PANEL_ALIASES = {
    "levels": "Levels (units)",
    "share_full": "Share of Full Effect",
    "share_explained": "Share of Explained Effect",
}


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

    def to_dict(self, relative_to: Optional[str] = None) -> dict[str, float]:
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
    nthreads : int, optional
        Number of threads for bootstrap inference, by default -1 (use all available).
    x1_vars : list[str], optional
        Additional variables to include in both short and long regressions, by default None.
    cluster_df : pd.Series, optional
        Cluster variable for bootstrap inference, by default None.
    combine_covariates : dict[str, list[str]], optional
        Dictionary grouping mediator variables for analysis, by default None.
    agg_first : bool, optional
        Whether to use aggregate-first algorithm for high-dimensional mediators, by default False.
    only_coef : bool, optional
        If True, skip bootstrap inference and only compute point estimates, by default True.
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
    nthreads: int = -1
    x1_vars: Optional[list[str]] = None
    cluster_df: Optional[pd.Series] = None
    combine_covariates: Optional[dict[str, list[str]]] = None
    agg_first: Optional[bool] = False
    only_coef: bool = True
    atol: Optional[float] = None
    btol: Optional[float] = None

    # Define attributes initialized post-creation
    cluster_dict: Optional[dict[Any, Any]] = field(init=False, default=None)
    unique_clusters: Optional[np.ndarray] = field(init=False, default=None)
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
            self.unique_clusters = self.cluster_df.unique()
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
                "You have provided combine_covariates, but agg_first is False. We recommend setting agg_first=True as this might massively decrease the computation time (in particular when boostrapping CIs)."
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

    def fit(self, X: spmatrix, Y: np.ndarray, store: bool = True):
        "Fit Linear Mediation Model."
        if store:
            self.X = X
            self.N = X.shape[0]

            self.X1 = self.X[:, ~self.mask]
            self.X1 = hstack([np.ones((self.N, 1)), self.X1])
            self.names_X1 = ["Intercept", self.decomp_var]
            if self.x1_vars is not None:
                self.names_X1 += self.x1_vars
            self.names_X = list(self.coefnames)
            self.decomp_var_in_X1_idx = self.names_X1.index(self.decomp_var)
            self.decomp_var_in_X_idx = self.names_X.index(self.decomp_var)

            self.X2 = self.X[:, self.mask]
            self.Y = Y

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
        self._absolute_ci = self._compute_ci(self._bootstrap_absolute_df, alpha)
        self._relative_explained_ci = self._compute_ci(
            self._bootstrap_relative_explained_df, alpha
        )
        self._relative_direct_ci = self._compute_ci(
            self._bootstrap_relative_direct_df, alpha
        )

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
        X1: spmatrix,
        X2: spmatrix,
        Y: np.ndarray,
        X: spmatrix,
        agg_first: Optional[bool],
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
        absolute_df = self._dict_to_df(self.results.absolute)
        relative_explained_df = self._dict_to_df(self.results.relative_to_explained)
        relative_direct_df = self._dict_to_df(self.results.relative_to_direct)

        if not self.only_coef:
            absolute_df = pd.concat([absolute_df, self._absolute_ci], axis=1)
            relative_explained_df = pd.concat(
                [relative_explained_df, self._relative_explained_ci],
                axis=1,
            )
            relative_direct_df = pd.concat(
                [relative_direct_df, self._relative_direct_ci], axis=1
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
        caption: Optional[str] = None,
        column_heads: Optional[list[str]] = None,
        panel_heads: Optional[list[str]] = None,
        rgroup_sep: Optional[str] = None,
        add_notes: Optional[str] = None,
        **kwargs,
    ) -> Union[pd.DataFrame, str, None]:
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
            Additional arguments passed to make_table function (type, digits, etc.).

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
        from pyfixest.report.make_table import make_table

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

        kwargs["rgroup_sep"] = "t" if rgroup_sep is None else rgroup_sep
        kwargs["caption"] = caption
        kwargs["notes"] = notes

        return make_table(res_sub, **kwargs)


    def coefplot(
        self,
        components_order: Optional[list[str]] = None,
        annotate_shares: bool = True,
        title: Optional[str] = None,
        figsize: Optional[tuple[int, int]] = None,
        keep: Optional[Union[list, str]] = None,
        drop: Optional[Union[list, str]] = None,
        exact_match: bool = False,
        labels: Optional[dict] = None,
        notes: Optional[str] = None,
    ):
        """
        Create a waterfall chart showing Gelbach decomposition results.
        The chart shows the transition from the initial difference (direct effect)
        through individual mediator contributions to the full effect, with a spanner
        showing the total explained effect above the mediator bars.
        Parameters
        ----------
        components_order : Optional[list[str]], optional
            Order of mediator components to display. If None, uses natural order from tidy().
        annotate_shares : bool, optional
            Whether to show percentage shares in parentheses. Default True.
        title : Optional[str], optional
            Chart title. If None, uses default title with decomposition variable.
        figsize : Optional[tuple[int, int]], optional
            Figure size (width, height) in inches. Default (12, 8).
        keep : Optional[Union[list, str]], optional
            The pattern for retaining mediator names. You can pass a string (one
            pattern) or a list (multiple patterns). Default is keeping all mediators.
            Uses regular expressions to select mediators.
        drop : Optional[Union[list, str]], optional
            The pattern for excluding mediator names. You can pass a string (one
            pattern) or a list (multiple patterns). Syntax is the same as for `keep`.
            Default is keeping all mediators. Can be used simultaneously with `keep`.
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
        # Custom ordering and labels
        gb.coefplot(
            components_order=["x22", "x21", "x23"],
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
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for coefplot. Install with: pip install matplotlib"
            )

        # Import the filtering function from pyfixest
        from pyfixest.utils.dev_utils import _select_order_coefs

        df = self.tidy()
        levels = df[df["panels"].str.lower().eq("levels (units)".lower())].copy()

        if levels.empty:
            raise ValueError("No rows found with panels == 'Levels (units)'.")
        if "direct_effect" not in levels.index or "full_effect" not in levels.index:
            raise ValueError(
                "Expected 'direct_effect' and 'full_effect' in 'Levels (units)'."
            )

        # Extract key values
        direct_effect = float(levels.loc["direct_effect", "coefficients"])
        full_effect = float(levels.loc["full_effect", "coefficients"])

        # Get mediator components (exclude the key summary effects)
        exclude = {
            "direct_effect",
            "full_effect",
            "unexplained_effect",
            "explained_effect",
        }
        mediators = [e for e in levels.index if e not in exclude]

        # Apply keep/drop filtering
        if keep is not None or drop is not None:
            keep_list = keep if isinstance(keep, list) else ([keep] if keep else [])
            drop_list = drop if isinstance(drop, list) else ([drop] if drop else [])
            mediators = _select_order_coefs(
                mediators, keep_list, drop_list, exact_match
            )

        # Apply user-specified order if provided
        if components_order:
            # Keep only valid mediators, maintain order, add any missing ones at end
            ordered_mediators = [c for c in components_order if c in mediators]
            ordered_mediators.extend(
                [c for c in mediators if c not in components_order]
            )
        else:
            ordered_mediators = mediators

        # Apply labels if provided
        display_labels = {}
        if labels:
            for med in ordered_mediators:
                display_labels[med] = labels.get(med, med)
        else:
            display_labels = {med: med for med in ordered_mediators}

        # Get mediator values
        mediator_data = [
            (med, float(levels.loc[med, "coefficients"])) for med in ordered_mediators
        ]

        # Separate red (move away from zero) and green (move toward zero) effects
        red_effects = []
        green_effects = []
        for name, val in mediator_data:
            # sign_product > 0 means signs are the same, which moves toward zero
            # We use a small epsilon to handle val == 0 correctly
            if (np.sign(direct_effect) * np.sign(val)) >= 0:
                green_effects.append((name, val))
            else:
                red_effects.append((name, val))

        # Sort by decreasing absolute value
        red_effects.sort(key=lambda x: abs(x[1]), reverse=True)
        green_effects.sort(key=lambda x: abs(x[1]), reverse=True)

        # Reorder mediators: red first (away from zero), then green (toward zero)
        reordered_mediators = red_effects + green_effects
        ordered_mediator_names = [name for name, _ in reordered_mediators]
        mediator_values = [val for _, val in reordered_mediators]

        # Calculate explained effect
        explained_effect = sum(val for _, val in mediator_data)

        # Create waterfall positions
        n_components = len(reordered_mediators)
        positions = list(range(n_components + 2))

        # Values for each bar
        bar_values = [direct_effect] + mediator_values + [full_effect]

        # Calculate cumulative positions for proper waterfall
        bar_bottoms = []
        bar_heights = []
        bar_types = []

        # Initial bar
        bar_bottoms.append(min(0, direct_effect))
        bar_heights.append(abs(direct_effect))
        bar_types.append("initial")

        # Mediator bars
        cumulative_position = direct_effect
        for name, val in reordered_mediators:
            old_pos = cumulative_position
            cumulative_position -= val
            bar_bottoms.append(min(old_pos, cumulative_position))
            bar_heights.append(abs(val))

            # Set bar type for coloring based on the sign product rule
            if (np.sign(direct_effect) * np.sign(val)) >= 0:
                bar_types.append("mediator_green") # Moves toward zero
            else:
                bar_types.append("mediator_red")   # Moves away from zero

        # Final bar (full effect)
        bar_bottoms.append(min(0, full_effect))
        bar_heights.append(abs(full_effect))
        bar_types.append("final")

        # Colors based on whether effects move toward or away from zero
        colors = []
        for bar_type in bar_types:
            if bar_type == "initial":
                colors.append("#1f77b4")  # Blue for initial
            elif bar_type == "mediator_green":
                colors.append("#2ca02c")  # Green (reducing effect)
            elif bar_type == "mediator_red":
                colors.append("#d62728")  # Red (increasing effect)
            elif bar_type == "final":
                colors.append("#1f77b4")  # Blue for final

        # Set default figsize
        if figsize is None:
            figsize = (12, 8)

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Draw bars
        bars = ax.bar(
            positions,
            bar_heights,
            bottom=bar_bottoms,
            width=0.6,
            color=colors,
            alpha=0.8,
        )

        # Set y-axis limits to handle both positive and negative values
        # Calculate the range of all bar positions
        all_bar_tops = [bottom + height for bottom, height in zip(bar_bottoms, bar_heights)]
        all_bar_bottoms = bar_bottoms

        y_min = min(min(all_bar_bottoms), 0) * 1.15
        y_max = max(max(all_bar_tops), 0) * 1.15

        # Ensure we have some padding even if all values are close to zero
        if abs(y_max - y_min) < 0.1:
            if y_max >= 0:
                y_max += 0.1
            if y_min <= 0:
                y_min -= 0.1

        ax.set_ylim(y_min, y_max)

        # Add spanner above mediator bars to show explained effect
        if n_components > 0:
            # Calculate spanner position and height
            mediator_start = 1  # First mediator position
            mediator_end = n_components  # Last mediator position

            # Position spanner appropriately based on the chart orientation
            if direct_effect >= 0:
                # For positive initial difference, place spanner above highest bar
                highest_bar_top = max([bottom + height for bottom, height in zip(bar_bottoms, bar_heights)])
                spanner_y = highest_bar_top + (y_max - highest_bar_top) * 0.2
            else:
                # For negative initial difference, place spanner below lowest bar
                lowest_bar_bottom = min(bar_bottoms)
                spanner_y = lowest_bar_bottom + (y_min - lowest_bar_bottom) * 0.2

            # Draw horizontal line spanning mediator bars
            ax.plot(
                [mediator_start - 0.3, mediator_end + 0.3],
                [spanner_y, spanner_y],
                color="#2E4A87",
                linewidth=2,
            )  # Navy blue for better appearance

            # Draw vertical lines at ends
            tick_height = (y_max - y_min) * 0.015
            ax.plot(
                [mediator_start - 0.3, mediator_start - 0.3],
                [spanner_y - tick_height, spanner_y + tick_height],
                color="#2E4A87",
                linewidth=2,
            )
            ax.plot(
                [mediator_end + 0.3, mediator_end + 0.3],
                [spanner_y - tick_height, spanner_y + tick_height],
                color="#2E4A87",
                linewidth=2,
            )

            # Add explained effect label
            spanner_center = (mediator_start + mediator_end) / 2
            if annotate_shares:
                share_of_direct = (explained_effect / direct_effect) * 100
                spanner_label = (
                    f"Explained Effect: {explained_effect:.3f} ({share_of_direct:.1f}%)"
                )
            else:
                spanner_label = f"Explained Effect: {explained_effect:.3f}"

            ax.text(
                spanner_center,
                spanner_y + tick_height * 2.5 if direct_effect >=0 else spanner_y - tick_height * 2.5,
                spanner_label,
                ha="center",
                va="bottom" if direct_effect >= 0 else "top",
                color="#2E4A87",
                fontweight="bold",
                fontsize=10,
            )

        # Add value labels on bars
        for i, (pos, height, bottom, val, bar_type) in enumerate(
            zip(positions, bar_heights, bar_bottoms, bar_values, bar_types)
        ):
            # Calculate label position (middle of bar)
            label_y = bottom + height / 2

            # Format label based on bar type
            if bar_type == "initial":
                label = f"{val:.3f}"
                ax.text(
                    pos,
                    label_y,
                    label,
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                    fontsize=10,
                )

            elif bar_type in ["mediator_green", "mediator_red"]:
                if annotate_shares:
                    # Create multi-line label: Absolute - Total % - Explained %
                    lines = [f"{val:.3f}"]

                    if direct_effect != 0:
                        total_share = (val / direct_effect) * 100
                        lines.append(f"({total_share:.1f}%)")

                    if explained_effect != 0:
                        explained_share = (val / explained_effect) * 100
                        lines.append(f"({explained_share:.1f}%)")

                    # Smart spacing
                    reference_spacing = 0.12
                    max_spacing_for_bar = height * 0.35
                    actual_spacing = min(reference_spacing, max_spacing_for_bar)

                    if len(lines) == 1:
                        ax.text(pos, label_y, lines[0], ha="center", va="center", color="black", fontweight="bold", fontsize=10)
                    elif len(lines) == 2:
                        ax.text(pos, label_y + actual_spacing / 2, lines[0], ha="center", va="center", color="black", fontweight="bold", fontsize=10)
                        ax.text(pos, label_y - actual_spacing / 2, lines[1], ha="center", va="center", color="black", fontweight="bold", fontsize=10)
                    elif len(lines) == 3:
                        ax.text(pos, label_y + actual_spacing, lines[0], ha="center", va="center", color="black", fontweight="bold", fontsize=10)
                        ax.text(pos, label_y, lines[1], ha="center", va="center", color="black", fontweight="bold", fontsize=10)
                        ax.text(pos, label_y - actual_spacing, lines[2], ha="center", va="center", color="#2E4A87", fontweight="bold", fontsize=10)
                else:
                    ax.text(pos, label_y, f"{val:.3f}", ha="center", va="center", color="black", fontweight="bold", fontsize=10)

            elif bar_type == "final":
                if annotate_shares:
                    share_of_direct = (full_effect / direct_effect) * 100
                    label = f"{val:.3f}\n({share_of_direct:.1f}%)"
                else:
                    label = f"{val:.3f}"
                ax.text(
                    pos,
                    label_y,
                    label,
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                    fontsize=10,
                )

            else:
                label = f"{val:.3f}"
                ax.text(
                    pos,
                    label_y,
                    label,
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                    fontsize=10,
                )

        # Customize the plot with reordered labels
        plot_labels = (
            ["Initial Difference"]
            + [display_labels.get(med, med).replace("_", " ") for med in ordered_mediator_names]
            + ["Final Difference"]
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(plot_labels, rotation=45, ha="right")

        if title is None:
            title = f"Decomposition of {self.decomp_var} by Covariates"

        if annotate_shares:
            title += (
                "\n(Normalized shares in parentheses for the decomposition section)"
            )

        ax.set_title(title, fontsize=14, pad=20)
        ax.set_ylabel("Difference (units)", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.8)

        # Add notes if provided
        if notes is not None:
            plt.figtext(
                0.02,
                0.02,
                notes,
                fontsize=9,
                style="italic",
                wrap=True,
                ha="left",
                va="bottom",
            )
            # Adjust layout to make room for notes
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)
        elif annotate_shares:
            # Default explanation note when showing shares
            default_note = "Mediator bars show: Absolute effect, (% of total effect), (% of explained effect in navy)."
            plt.figtext(
                0.02,
                0.02,
                default_note,
                fontsize=9,
                style="italic",
                wrap=True,
                ha="left",
                va="bottom",
            )
            plt.tight_layout()
            plt.subplots_adjust(
                bottom=0.20
            )  # More space to avoid overlap with x-labels
        else:
            # No notes by default
            plt.tight_layout()
        plt.show()


def _decompose_arg_check(
    type: str,
    has_weights: bool,
    is_iv: bool,
    method: str,
) -> None:
    "Check arguments for decomposition."
    supported_decomposition_types = ["gelbach"]

    if type not in supported_decomposition_types:
        raise ValueError(
            f"'type' {type} is not in supported types {supported_decomposition_types}."
        )

    if has_weights:
        raise NotImplementedError(
            "Decomposition is currently not supported for models with weights."
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
