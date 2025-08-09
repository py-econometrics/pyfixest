from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.sparse import hstack, spmatrix, vstack
from scipy.sparse.linalg import lsqr
from tqdm import tqdm


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
        return [
            *["direct_effect", "full_effect", "explained_effect", "unexplained_effect"],
            *self.mediator_effects.keys(),
        ]

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

    def summary_df(self) -> pd.DataFrame:
        """Create a summary DataFrame with all three formats."""
        return pd.DataFrame(
            {
                "Absolute": self.to_dict(),
                "Relative to Explained": self.to_dict("explained"),
                "Relative to Direct": self.to_dict("direct"),
            }
        )


@dataclass
class GelbachDecomposition:
    """
    Linear Mediation Model.

    Implements the Gelbach (2016) decomposition method to decompose the effect of a
    treatment variable into direct and indirect (mediated) components. The method
    compares coefficients from a "short" regression (outcome on treatment) with a
    "long" regression (outcome on treatment plus mediators).

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
    contribution_dict : dict
        Point estimates of direct, indirect, and total effects.
    contribution_dict_relative_explained : dict
        Effects relative to explained effect.
    contribution_dict_relative_direct : dict
        Effects relative to direct effect.

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
        if self.x1_vars is None:
            x1_variables = [self.decomp_var]
        else:
            x1_variables = [self.decomp_var, *self.x1_vars]

        # build index for all variables in X1: decomp_var, x1_vars
        x1_indices = [self.coefnames.index(var) for var in x1_variables]
        self.mask = np.ones(len(self.coefnames), dtype=bool)
        self.mask[x1_indices] = False

        self.mediator_names = [
            name for name in self.coefnames if self.mask[self.coefnames.index(name)]
        ]
        self.intercept_in_mediator_idx = self.mediator_names.index("Intercept")

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

            # Backward compatibility - provide the old dictionary interface
            self.contribution_dict = self.results.absolute
            self.contribution_dict_relative_explained = (
                self.results.relative_to_explained
            )
            self.contribution_dict_relative_direct = self.results.relative_to_direct

            # Prepare cluster bootstrap if relevant
            self.X_dict = {}
            self.Y_dict = {}

            if self.unique_clusters is not None and not self.only_coef:
                for g in self.unique_clusters:
                    cluster_idx = np.where(self.cluster_df == g)[0]
                    self.X_dict[g] = self.X[cluster_idx]
                    self.Y_dict[g] = self.Y[cluster_idx]

            return {
                "contribution_dict": self.contribution_dict,
                "contribution_dict_relative_explained": self.contribution_dict_relative_explained,
                "contribution_dict_relative_direct": self.contribution_dict_relative_direct,
                "results": self.results,
            }

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

            return {
                "contribution_dict": bootstrap_results.absolute,
                "contribution_dict_relative_explained": bootstrap_results.relative_to_explained,
                "contribution_dict_relative_direct": bootstrap_results.relative_to_direct,
            }

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
        self._bootstrap_absolute_df = pd.DataFrame(
            [d["contribution_dict"] for d in _bootstrapped]
        )
        self._bootstrap_relative_explained_df = pd.DataFrame(
            [d["contribution_dict_relative_explained"] for d in _bootstrapped]
        )
        self._bootstrap_relative_direct_df = pd.DataFrame(
            [d["contribution_dict_relative_direct"] for d in _bootstrapped]
        )

        # compute ci
        self._absolute_ci = pd.DataFrame(
            {
                "ci_lower": np.percentile(
                    self._bootstrap_absolute_df, 100 * (alpha / 2), axis=0
                ),
                "ci_upper": np.percentile(
                    self._bootstrap_absolute_df, 100 * (1 - alpha / 2), axis=0
                ),
            },
            index=self._bootstrap_absolute_df.columns,
        )
        self._absolute_ci = self._absolute_ci.astype(float)

        self._relative_explained_ci = pd.DataFrame(
            {
                "ci_lower": np.percentile(
                    self._bootstrap_relative_explained_df, 100 * (alpha / 2), axis=0
                ),
                "ci_upper": np.percentile(
                    self._bootstrap_relative_explained_df, 100 * (1 - alpha / 2), axis=0
                ),
            },
            index=self._bootstrap_relative_explained_df.columns,
        )
        self._relative_explained_ci = self._relative_explained_ci.astype(float)

        self._relative_direct_ci = pd.DataFrame(
            {
                "ci_lower": np.percentile(
                    self._bootstrap_relative_direct_df, 100 * (alpha / 2), axis=0
                ),
                "ci_upper": np.percentile(
                    self._bootstrap_relative_direct_df, 100 * (1 - alpha / 2), axis=0
                ),
            },
            index=self._bootstrap_relative_direct_df.columns,
        )
        self._relative_direct_ci = self._relative_direct_ci.astype(float)

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

        # Create structured results
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
            Which panels to include. One of 'all', 'Levels (units)',
            'Share of Explained Effect', 'Share of Full Effect', by default "all".

        Returns
        -------
        pd.DataFrame
            A tidy DataFrame with the decomposition results.
        """
        # Convert scalar dictionaries to DataFrames with proper index
        contribution_df = pd.DataFrame(
            list(self.contribution_dict.items()), columns=["effect", "coefficients"]
        ).set_index("effect")

        contribution_relative_explained_df = pd.DataFrame(
            list(self.contribution_dict_relative_explained.items()),
            columns=["effect", "coefficients"],
        ).set_index("effect")

        contribution_relative_direct_df = pd.DataFrame(
            list(self.contribution_dict_relative_direct.items()),
            columns=["effect", "coefficients"],
        ).set_index("effect")

        if not self.only_coef:
            contribution_df = pd.concat([contribution_df, self._absolute_ci], axis=1)
            contribution_relative_explained_df = pd.concat(
                [contribution_relative_explained_df, self._relative_explained_ci],
                axis=1,
            )
            contribution_relative_direct_df = pd.concat(
                [contribution_relative_direct_df, self._relative_direct_ci], axis=1
            )

        contribution_df["panels"] = np.repeat("Levels (units)", len(contribution_df))
        contribution_relative_explained_df["panels"] = np.repeat(
            "Share of Explained Effect", len(contribution_relative_explained_df)
        )
        contribution_relative_direct_df["panels"] = np.repeat(
            "Share of Full Effect", len(contribution_relative_direct_df)
        )

        if panels == "all":
            return pd.concat(
                [
                    contribution_df,
                    contribution_relative_direct_df,
                    contribution_relative_explained_df,
                ],
                axis=0,
            )
        elif panels == "Levels (units)":
            return contribution_df
        elif panels == "Share of Explained Effect":
            return contribution_relative_explained_df
        elif panels == "Share of Full Effect":
            return contribution_relative_direct_df
        else:
            raise ValueError(
                f"The 'panels' parameter must be one of 'all', 'Levels (units)', 'Share of Explained Effect', 'Share of Full Effect'. Got '{panels}'."
            )

    def _prepare_etable_df(self, digits: int = 3, panels: str = "all") -> pd.DataFrame:
        """
        Prepare a DataFrame formatted for etable output.

        Parameters
        ----------
        digits : int, optional
            Number of digits to display in the summary table, by default 3.
        panels : str, optional
            Which panels to include. One of 'all', 'Levels (units)', 'Share of Explained Effect', 'Share of Full Effect'

        Returns
        -------
        pd.DataFrame
            Multi-index DataFrame with estimated coefficients with the following columns:
            "direct_effect", "full_effect", "explained_effect"
            If `only_coef` is False, also includes "ci_lower" and "ci_upper" for the confidence intervals.
            First level of index is the 'panels' type, second level is the covariates/groups.
        """
        mediators = list(self.combine_covariates_dict.keys())
        df = self.tidy(panels="all").round(digits)

        panels_to_include = df["panels"].unique() if panels == "all" else [panels]

        results = {}

        for panels_name in panels_to_include:
            df_sub = df[df["panels"] == panels_name].copy()

            summary_data = {}

            main_effects_row = {}
            main_effects_ci_row = {}

            for effect in ["direct_effect", "full_effect", "explained_effect"]:
                if effect in df_sub.index:
                    coef = df_sub.loc[effect, "coefficients"]
                    main_effects_row[effect] = f"{coef:.{digits}f}"

                    if not self.only_coef and "ci_lower" in df_sub.columns:
                        ci_lower = df_sub.loc[effect, "ci_lower"]
                        ci_upper = df_sub.loc[effect, "ci_upper"]
                        main_effects_ci_row[effect] = (
                            f"[{ci_lower:.{digits}f}, {ci_upper:.{digits}f}]"
                        )
                    else:
                        main_effects_ci_row[effect] = "-"
                else:
                    main_effects_row[effect] = "-"
                    main_effects_ci_row[effect] = "-"

            summary_data[self.decomp_var] = main_effects_row

            if not self.only_coef:
                summary_data[f"{self.decomp_var}_ci"] = (
                    main_effects_ci_row  # Empty name for CI row
                )

            for mediator in mediators:
                if mediator in df_sub.index:
                    coef = df_sub.loc[mediator, "coefficients"]

                    summary_data[mediator] = {
                        "direct_effect": "-",
                        "full_effect": "-",
                        "explained_effect": f"{coef:.{digits}f}",
                    }

                    if not self.only_coef and "ci_lower" in df_sub.columns:
                        ci_lower = df_sub.loc[mediator, "ci_lower"]
                        ci_upper = df_sub.loc[mediator, "ci_upper"]
                        ci_str = f"[{ci_lower:.{digits}f}, {ci_upper:.{digits}f}]"

                        summary_data[f"{mediator}_ci"] = {
                            "direct_effect": "-",
                            "full_effect": "-",
                            "explained_effect": ci_str,
                        }

            # replace
            if panels_name == "Share of Full Effect" and not self.only_coef:
                # don't print CIs as they are [1,1]
                summary_data[f"{self.decomp_var}_ci"]["direct_effect"] = "-"
            elif panels_name == "Share of Explained Effect":
                summary_data[self.decomp_var]["direct_effect"] = "-"
                summary_data[self.decomp_var]["full_effect"] = "-"

                # delete CIs fully
                if not self.only_coef:
                    summary_data.pop(f"{self.decomp_var}_ci")

            # Convert to DataFrame
            summary_df = pd.DataFrame(summary_data).T
            summary_df.columns = ["direct_effect", "full_effect", "explained_effect"]

            summary_df.index = pd.Index(
                ["" if name.endswith("_ci") else name for name in summary_df.index]
            )
            results[panels_name] = summary_df

        if panels == "all":
            return pd.concat(results, axis=0)
        else:
            return results[panels]

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

        panels_arg_to_label = {
            "levels": "Levels (units)",
            "share_full": "Share of Full Effect",
            "share_explained": "Share of Explained Effect",
        }

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

        res = self._prepare_etable_df(panels="all")

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
