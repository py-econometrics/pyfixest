from dataclasses import dataclass, field
from typing import Any, Optional

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

        self._check_combine_covariates()

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
            self.decomp_var_in_X1_idx = self.names_X1.index(self.decomp_var)

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
            ) = results

            # Prepare cluster bootstrap if relevant
            self.X_dict = {}
            self.Y_dict = {}

            if self.unique_clusters is not None and not self.only_coef:
                for g in self.unique_clusters:
                    cluster_idx = np.where(self.cluster_df == g)[0]
                    self.X_dict[g] = self.X[cluster_idx]
                    self.Y_dict[g] = self.Y[cluster_idx]

            return self.contribution_dict

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

            _, _, _, contribution_dict = results

            return contribution_dict

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

        self._bootstrapped = {
            key: np.concatenate([d[key] for d in _bootstrapped])
            for key in _bootstrapped[0]
        }

        self.ci = {
            key: np.percentile(
                self._bootstrapped[key],
                100 * np.array([alpha / 2, 1 - alpha / 2]),
                axis=0,
            )
            for key in self._bootstrapped
        }

    def summary(self, digits: int = 4) -> pd.DataFrame:
        """
        Summary Table for Total, Mediated and Direct Effects.

        Parameters
        ----------
        digits : int, optional
            Number of digits to display in the summary table, by default 4.
        """
        mediators = list(self.combine_covariates_dict.keys())

        # round all values in self.contribution_dict and self.ci to the specified number of digits

        contribution_dict = self.contribution_dict.copy()

        for key in contribution_dict:
            contribution_dict[key] = np.round(contribution_dict[key], digits)

        rows = []
        rows.append(
            [
                f"{contribution_dict['direct_effect'].item():.{digits}f}",
                f"{contribution_dict['full_effect'].item():.{digits}f}",
                f"{contribution_dict['explained_effect'].item():.{digits}f}",
            ]
        )

        if not self.only_coef:
            ci = self.ci.copy()
            for key in contribution_dict:
                ci[key] = np.round(ci[key], digits)

            rows.append(
                [
                    f"[{ci['direct_effect'][0]:.{digits}f}, {ci['direct_effect'][1]:.{digits}f}]",
                    f"[{ci['full_effect'][0]:.{digits}f}, {ci['full_effect'][1]:.{digits}f}]",
                    f"[{ci['explained_effect'][0]:.{digits}f}, {ci['explained_effect'][1]:.{digits}f}]",
                ]
            )

        for mediator in mediators:
            rows.append(["", "", f"{contribution_dict[mediator].item():.{digits}f}"])
            if not self.only_coef:
                rows.append(
                    [
                        "",
                        "",
                        f"[{ci[mediator][0]:.{digits}f}, {ci[mediator][1]:.{digits}f}]",
                    ]
                )

        if not self.only_coef:
            index = [self.decomp_var, ""] + [
                item for mediator in mediators for item in [f"{mediator}", ""]
            ]
        else:
            index = [self.decomp_var] + [mediator for mediator in mediators]

        columns = ["direct_effect", "full_effect", "explained_effect"]

        self.summary_table = (
            pd.DataFrame(rows, index=index, columns=columns).fillna("").T
        )

        return self.summary_table

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
    ]:
        "Run the Gelbach decomposition."
        N = X1.shape[0]

        # Compute direct effect
        direct_effect = lsqr(X1, Y, atol=self.atol, btol=self.btol)[0]
        direct_effect_array = np.array([direct_effect[self.decomp_var_in_X1_idx]])

        # Compute beta_full and beta2
        beta_full = lsqr(X, Y, atol=self.atol, btol=self.btol)[0]
        beta2 = beta_full[self.mask]

        # Initialize contribution_dict: a dictionary to store the contribution of each covariate
        contribution_dict = {}

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

        # Compute explained and unexplained effects
        contribution_dict["explained_effect"] = np.sum(
            list(contribution_dict.values()), keepdims=True
        ).flatten()
        contribution_dict["unexplained_effect"] = (
            direct_effect_array - contribution_dict["explained_effect"]
        ).flatten()
        contribution_dict["direct_effect"] = direct_effect_array
        contribution_dict["full_effect"] = beta_full[
            self.decomp_var_in_X1_idx
        ].flatten()

        # Collect all created elements into a tuple
        results = (
            direct_effect_array,
            beta_full,
            beta2,
            contribution_dict,
        )

        return results


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
