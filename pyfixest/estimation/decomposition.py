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
class GelbachDecomposition:
    """
    Linear Mediation Model.

    Initial implementation by Apoorva Lal at
    https://gist.github.com/apoorvalal/e7dc9f3e52dcd9d51854b28b3e8a7ba4.
    """

    # Either use the original single-parameter approach
    # param: Optional[str] = None

    # Or use the new multi-parameter approach
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
            raise ValueError(f"{self.decomp_var} is not in the coefnames.")
        if self.x1_vars is not None:
            for var in self.x1_vars:
                if var not in self.coefnames:
                    raise ValueError(f"{var} is not in the coefnames.")
        if self.x1_vars is not None and self.decomp_var in self.x1_vars:
            raise ValueError(
                "The decomposition variable cannot be included in the x1_vars argument."
            )

    def _check_combine_covariates(self):
        # Check that each value in self.combine_covariates_dict is in self.mediator_names
        for _, values in self.combine_covariates_dict.items():
            if not isinstance(values, list):
                raise TypeError("Values in combine_covariates_dict must be lists.")
            for v in values:
                if v not in self.mediator_names:
                    raise ValueError(f"{v} is not in the mediator names.")

        # Check for overlap in values between different keys
        all_values = {
            k: set([v] if isinstance(v, str) else v)
            for k, v in self.combine_covariates_dict.items()
        }
        for key1, values1 in all_values.items():
            for key2, values2 in all_values.items():
                if key1 != key2 and values1 & values2:
                    overlap = values1 & values2
                    raise ValueError(f"{overlap} is in both {key1} and {key2}.")

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
                self.contribution_dict,
                self.contribution_dict_relative_explained,
                self.contribution_dict_relative_direct,
            ) = results

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
                contribution_dict,
                contribution_dict_relative_explained,
                contribution_dict_relative_direct,
            ) = results

            return {
                "contribution_dict": contribution_dict,
                "contribution_dict_relative_explained": contribution_dict_relative_explained,
                "contribution_dict_relative_direct": contribution_dict_relative_direct,
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
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]:
        "Run the Gelbach decomposition."
        N = X1.shape[0]

        # Compute direct effect
        direct_effect = lsqr(X1, Y, atol=self.atol, btol=self.btol)[0]
        direct_effect_array = np.array([direct_effect[self.decomp_var_in_X1_idx]])

        # Compute beta_full and beta2
        beta_full = lsqr(X, Y, atol=self.atol, btol=self.btol)[0]
        beta2 = beta_full[self.mask]

        # Initialize contribution_dict: a dictionary to store the absolute and relative contributions of each covariate
        (
            contribution_dict,
            contribution_dict_relative_explained,
            contribution_dict_relative_direct,
        ) = {}, {}, {}

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
                contribution_dict[name] = np.array([delta[i]])
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
                contribution_dict[name] = np.array([np.sum(delta[variable_idx])])

        contribution_dict["explained_effect"] = np.sum(
            list(contribution_dict.values()), keepdims=True
        ).flatten()
        contribution_dict["unexplained_effect"] = (
            direct_effect_array - contribution_dict["explained_effect"]
        ).flatten()
        contribution_dict["direct_effect"] = direct_effect_array

        contribution_dict["full_effect"] = beta_full[self.decomp_var_in_X_idx].flatten()

        for name, value in contribution_dict.items():
            if contribution_dict["explained_effect"] != 0:
                contribution_dict_relative_explained[name] = (
                    value / contribution_dict["explained_effect"]
                )
            else:
                contribution_dict_relative_explained[name] = np.nan

            if contribution_dict["direct_effect"] != 0:
                contribution_dict_relative_direct[name] = (
                    value / contribution_dict["direct_effect"]
                )
            else:
                contribution_dict_relative_direct[name] = np.nan

        results = (
            direct_effect_array,
            beta_full,
            beta2,
            contribution_dict,
            contribution_dict_relative_explained,
            contribution_dict_relative_direct,
        )

        return results

    def tidy(self, stats="all") -> pd.DataFrame:
        """
        Tidy the Gelbach decomposition output into a DataFrame.

        Args
        ----
        stats : str, optional
            Which stats of summary to include. One of 'all', 'absolute', 'relative (vs explained)', 'relative (vs full)'
            "all" includes all of the ladder statistics. "absolute" includes only the absolute differences of the short and
            long regression as well as contributions of mediatios. "relative (vs explained)" normalizes all contributions
            relative to the "explained" effect. "relative (vs full)" normalizes all contributions relative to the "full" effect.

        Returns
        -------
        pd.DataFrame
            DataFrame with the tidy Gelbach decomposition output.
        """
        contribution_df = pd.DataFrame(self.contribution_dict).T
        contribution_relative_explained_df = pd.DataFrame(
            self.contribution_dict_relative_explained
        ).T
        contribution_relative_direct_df = pd.DataFrame(
            self.contribution_dict_relative_direct
        ).T

        contribution_df.columns = ["coefficients"]
        contribution_relative_explained_df.columns = ["coefficients"]
        contribution_relative_direct_df.columns = ["coefficients"]

        if not self.only_coef:
            contribution_df = pd.concat([contribution_df, self._absolute_ci], axis=1)
            contribution_relative_explained_df = pd.concat(
                [contribution_relative_explained_df, self._relative_explained_ci],
                axis=1,
            )
            contribution_relative_direct_df = pd.concat(
                [contribution_relative_direct_df, self._relative_direct_ci], axis=1
            )

        contribution_df["stats"] = np.repeat("Levels (units)", len(contribution_df))
        contribution_relative_explained_df["stats"] = np.repeat(
            "Share of Explained Effect", len(contribution_relative_explained_df)
        )
        contribution_relative_direct_df["stats"] = np.repeat(
            "Share of Full Effect", len(contribution_relative_direct_df)
        )

        if stats == "all":
            return pd.concat(
                [
                    contribution_df,
                    contribution_relative_direct_df,
                    contribution_relative_explained_df,
                ],
                axis=0,
            )
        elif stats == "Levels (units)":
            return contribution_df
        elif stats == "Share of Explained Effect":
            return contribution_relative_explained_df
        elif stats == "Share of Full Effect":
            return contribution_relative_direct_df
        else:
            raise ValueError(
                f"stats must be one of 'all', 'Levels (units)', 'Share of Explained Effect', 'Share of Full Effect', got {stats}"
            )

    def _prepare_etable_df(self, digits: int = 4, stats="all") -> pd.DataFrame:
        """
        Prepare a DataFrame formatted for etable output.

        Parameters
        ----------
        digits : int, optional
            Number of digits to display in the summary table, by default 4.
        stats : str, optional
            Which stats to include. One of 'all', 'Levels (units)', 'Share of Explained Effect', 'Share of Full Effect'

        Returns
        -------
        pd.DataFrame
            Multi-index DataFrame with estimated coefficients with the following columns:
            "direct_effect", "full_effect", "explained_effect"
            If `only_coef` is False, also includes "ci_lower" and "ci_upper" for the confidence intervals.
            First level of index is the 'stats' type, second level is the covariates/groups.
        """
        mediators = list(self.combine_covariates_dict.keys())
        df = self.tidy(stats="all").round(digits)

        stats_to_include = df["stats"].unique() if stats == "all" else [stats]

        results = {}

        for stats_name in stats_to_include:
            df_sub = df[df["stats"] == stats_name].copy()

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
                summary_data[f"{self.decomp_var}_ci"] = main_effects_ci_row  # Empty name for CI row

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
            if stats_name == "Share of Full Effect" and not self.only_coef:
                # don't print CIs as they are [1,1]
                summary_data[f"{self.decomp_var}_ci"]["direct_effect"] = "-"
            elif stats_name == "Share of Explained Effect":

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
            results[stats_name] = summary_df

        if stats == "all":
            return pd.concat(results, axis=0)
        else:
            return results[stats]

    def summary(self, digits: int = 3, stats: str = "all") -> None:
        """
        Print summary of Gelbach decomposition results.

        Parameters
        ----------
        digits : int, optional
            Number of digits to display, by default 3.
        stats : str, optional
            Which stats to include. One of 'all', 'Levels (units)', 'Share of Explained Effect', 'Share of Full Effect'
        """
        print("###")
        print("")
        print("Gelbach Decomposition Results")
        print(f"Decomposition variable: {self.decomp_var}")
        if self.x1_vars is not None:
            print(f"Control variables: {', '.join(self.x1_vars)}")
        print("")

        df = self._prepare_etable_df(digits=digits, stats=stats)
        print(df.to_markdown(floatfmt=f".{digits}f"))
        print("---")

    def etable(
        self,
        stats: str = "all",
        caption: Optional[str] = None,
        model_heads: Optional[list[str]] = None,
        rgroup_sep: Optional[str] = None,
        add_notes: Optional[str] = None,
        **kwargs,
    ) -> Union[pd.DataFrame, str, None]:
        """
        Create a GT (great tables) or tex table for the Gelbach decomposition output.

        Args
        ----
        stats : Union[str, list[str]], optional
            Which stats of summary to include. One or more of 'Levels (units)', 'Share of Full Effect', 'Share of Explained Effect'.
        caption: str, optional
            Caption for the table.
        model_heads: list[str], optional
            A list of length 3 with the column names. If None, the default column names are used.
        rgroup_sep : Optional[str]
            Whether group names are separated by lines. The default is "t".
            When output type = 'gt', the options are 'tb', 't', 'b', or '', i.e.
            you can specify whether to have a line above, below, both or none.
            When output type = 'tex' no line will be added between the row groups
            when rgroup_sep is '' and otherwise a line before the group name will be added.
        add_notes : str, optional
            Additional notes to add to the table.
        **kwargs : dict, optional
            Additional arguments to pass to the make_table function. You can add table notes, captions, etc.
            See the make_table function for more details.

        Returns
        -------
        A table in the specified format.

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
        gb.etable(model_heads = ["Full Difference", "Unexplained Difference", "Explained Difference"])
        """
        from pyfixest.report.make_table import make_table

        if model_heads is not None:
            if len(model_heads) != 3:
                raise ValueError("model_heads must be a list of length 3.")

        if stats == "all":
            stats_list = ["Levels (units)", "Share of Full Effect", "Share of Explained Effect"]
        else:
            if isinstance(stats, str):
                stats_list = [stats]
            else:
                stats_list = stats

        for stat in stats_list:
            if stat not in ["Levels (units)", "Share of Full Effect", "Share of Explained Effect"]:
                raise ValueError(f"stats must be one or more of 'Levels (units)', 'Share of Full Effect', 'Share of Explained Effect'. Got {stat}.")

        res = self._prepare_etable_df(stats="all")

        # Filter by stats types (which are in the index, not columns)
        if isinstance(res.index, pd.MultiIndex):
            # Filter rows by stats type (first level of index)
            mask = res.index.get_level_values(0).isin(stats_list)
            res_sub = res.loc[mask, :]
        else:
            # If not MultiIndex, return the full result
            res_sub = res

        if self.x1_vars is not None:
            default_model_notes = [
                f"Col 1: Adjusted Difference (by { "+".join(self.x1_vars)}) (Direct Effect): Coefficient on {self.decomp_var} in short regression.",
                f"Col 2: Adjusted Difference (Full Effect): Coefficient on {self.decomp_var} in long regression.",
                f"Col 3: Explained Difference (Explained Effect): Difference in coefficients of {self.decomp_var} in short and long regression.",
            ]

        else:
            default_model_notes = [
                f"Col 1: Raw Difference (Direct Effect): Coefficient on {self.decomp_var} in short regression.",
                f"Col 2: Adjusted Difference (Full Effect): Coefficient on {self.decomp_var} in long regression.",
                f"Col 3: Explained Difference (Explained Effect): Difference in coefficients of {self.decomp_var} in short and long regression.",
            ]

        panel = 0
        if "Levels (units)" in stats_list:
            panel += 1
            default_model_notes.append(f"Panel {panel}: Levels (units).")
        if "Share of Full Effect" in stats_list:
            panel += 1
            default_model_notes.append(f"Panel {panel}: Share of Full Effect: Levels normalized by coefficient of the short regression.")
        if "Share of Explained Effect" in stats_list:
            panel += 1
            default_model_notes.append(f"Panel {panel}: Share of Explained Effect: Levels normalized by coefficient of the long regression.")

        default_model_heads = [
            "Initial Difference",
            "Adjusted Difference",
            "Explained Difference",
        ]

        res_sub.columns = model_heads if model_heads is not None else default_model_heads


        notes = f"""
            Decomposition variable: {self.decomp_var}.
        """

        if self.x1_vars is not None:
            notes += f"""
            Control Variables: {", ".join(self.x1_vars)}.
            """

        notes += "\n".join(default_model_notes)

        if add_notes is not None:
            notes += f"""
            {add_notes}
            """

        if not self.only_coef:
            notes += f"""
                CIs are computed using B = {self.B} bootstrap replications
            """
            if self.cluster_df is None:
                notes += " using iid sampling."
            else:
                notes += f" using clustered sampling by {self.cluster_df.name}."

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
