from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

# from joblib import Parallel, delayed
from tqdm import tqdm


@dataclass
class GelbachDecomposition:
    """
    Linear Mediation Model.

    Initial implementation by Apoorva Lal at
    https://gist.github.com/apoorvalal/e7dc9f3e52dcd9d51854b28b3e8a7ba4.
    """

    agg: bool
    param: str
    coefnames: list[str]
    nthreads: int = -1
    cluster_df: Optional[pd.Series] = None
    combine_covariates: dict[str, list[str]] = None

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
        param_idx = self.coefnames.index(self.param)
        self.intercept_idx = self.coefnames.index("Intercept")
        self.coefnames_no_intercept = self.coefnames[~self.intercept_idx]
        self.mask = np.ones(len(self.coefnames), dtype=bool)
        self.mask[param_idx] = False

        self.mediator_names = [
            name for name in self.coefnames if self.param not in name
        ]
        self.intercept_in_mediator_idx = self.mediator_names.index("Intercept")

        # Handle clustering setup if cluster_df is provided
        if self.cluster_df is not None:
            self.unique_clusters = self.cluster_df.unique()
            self.cluster_dict = {
                cluster: self.cluster_df[self.cluster_df == cluster].index
                for cluster in self.unique_clusters
            }
        else:
            self.unique_clusters = None
            self.cluster_dict = None

        if self.combine_covariates is None:
            self.combine_covariates = {
                x: x for x in self.mediator_names if x != "Intercept"
            }

        self._check_combine_covariates()

        self.contribution_dict = {
            key: 0 for key in self.combine_covariates if key != "Intercept"
        }

    def _check_combine_covariates(self):
        # Check that each value in self.combine_covariates is in self.mediator_names
        for key, values in self.combine_covariates.items():
            if isinstance(values, str):
                values = [values]  # Convert to list for consistent handling
            for v in values:
                if v not in self.mediator_names:
                    raise ValueError(f"{v} is not in the mediator names.")

        # Check for overlap in values between different keys
        all_values = {
            k: set([v] if isinstance(v, str) else v)
            for k, v in self.combine_covariates.items()
        }
        for key1, values1 in all_values.items():
            for key2, values2 in all_values.items():
                if key1 != key2 and values1 & values2:  # Check intersection
                    overlap = values1 & values2
                    raise ValueError(f"{overlap} is in both {key1} and {key2}.")

    def fit(self, X: np.ndarray, Y: np.ndarray, store: bool = True):
        "Fit Linear Mediation Model."
        if store:
            self.X = X
            self.N = X.shape[0]
            self.X1 = self.X[:, ~self.mask]
            self.X1 = np.concatenate([np.ones((self.N, 1)), self.X1], axis=1)
            self.names_X1 = ["Intercept"] + [self.param]
            self.param_in_X1_idx = self.names_X1.index(self.param)

            self.X2 = self.X[:, self.mask]
            self.Y = Y

            self.direct_effect = np.linalg.lstsq(self.X1, self.Y, rcond=[1])[
                0
            ].flatten()[self.param_in_X1_idx]
            self.direct_effect = np.array([self.direct_effect])

            # Gelbach Method:
            self.gamma = np.linalg.lstsq(self.X1[:, 1:], self.X2, rcond=1)[0].flatten()
            self.beta_full = np.linalg.lstsq(self.X, self.Y, rcond=1)[0].flatten()
            self.beta2 = self.beta_full[self.mask].flatten()
            self.delta = self.gamma * self.beta2.flatten()

            for name, covariates in self.combine_covariates.items():
                variable_idx = (
                    self.mediator_names.index(covariates)
                    if isinstance(covariates, str)
                    else [self.mediator_names.index(cov) for cov in covariates]
                )
                self.contribution_dict[name] = np.array(
                    [np.sum(self.delta[variable_idx])]
                )

            self.contribution_dict["explained_effect"] = np.sum(
                list(self.contribution_dict.values()), keepdims=True
            ).flatten()
            self.contribution_dict["unexplained_effect"] = (
                self.direct_effect - self.contribution_dict["explained_effect"]
            ).flatten()
            self.contribution_dict["direct_effect"] = self.direct_effect
            self.contribution_dict["full_effect"] = self.beta_full[
                self.param_in_X1_idx
            ].flatten()

            # prepare bootstrap in first iteration
            if self.cluster_df is not None:
                for g in self.unique_clusters:
                    cluster_idx = np.where(self.cluster_df == g)[0]
                    self.X_dict[g] = self.X[cluster_idx]
                    self.Y_dict[g] = self.Y[cluster_idx]

            return self.contribution_dict

        else:
            # Gelbach Method:

            contribution_dict = {
                key: 0 for key in self.combine_covariates if key != "Intercept"
            }

            X1 = np.concatenate([np.ones((self.N, 1)), X[:, ~self.mask]], axis=1)
            X2 = X[:, self.mask]

            direct_effect = np.linalg.lstsq(X1, Y, rcond=1)[0].flatten()[
                self.param_in_X1_idx
            ]
            direct_effect = np.array([direct_effect])

            gamma = np.linalg.lstsq(X2, X1[:, 1:], rcond=1)[0].flatten()
            beta_full = np.linalg.lstsq(X, Y, rcond=1)[0].flatten()
            beta2 = beta_full[self.mask].flatten()
            delta = gamma * beta2

            for name, covariates in self.combine_covariates.items():
                variable_idx = (
                    self.mediator_names.index(covariates)
                    if isinstance(covariates, str)
                    else [self.mediator_names.index(cov) for cov in covariates]
                )
                contribution_dict[name] = np.array([np.sum(delta[variable_idx])])

            contribution_dict["explained_effect"] = np.sum(
                list(contribution_dict.values()), keepdims=True
            ).flatten()
            contribution_dict["unexplained_effect"] = (
                direct_effect - contribution_dict["explained_effect"]
            ).flatten()
            contribution_dict["direct_effect"] = direct_effect
            contribution_dict["full_effect"] = beta_full[self.param_in_X1_idx].flatten()

            return np.concatenate(list(contribution_dict.values()))

    def bootstrap(self, rng: np.random.Generator, B: int = 1_000, alpha: float = 0.05):
        "Bootstrap Confidence Intervals for Total, Mediated and Direct Effects."
        self.alpha = alpha
        self.B = B

        # self._bootstrapped = np.c_[
        #    Parallel(n_jobs=self.nthreads)(
        #        delayed(self._bootstrap)(rng=rng) for _ in tqdm(range(B))
        #    )
        # ]

        self._bootstrapped = np.c_[[self._bootstrap(rng=rng) for _ in tqdm(range(B))]]

        self.ci = np.percentile(
            self._bootstrapped, 100 * np.array([alpha / 2, 1 - alpha / 2]), axis=0
        )

    def summary(self) -> pd.DataFrame:
        "Summary Table for Total, Mediated and Direct Effects."
        lb = self.alpha / 2
        ub = 1 - lb
        index = ["Estimate", f"{lb*100:.1f}%", f"{ub*100:.1f}%"]

        mediators = list(self.combine_covariates.keys())
        n_df = len(mediators) + 1
        index = [self.param] + mediators

        direct_effect = np.full(n_df, np.nan)
        direct_effect[0] = self.contribution_dict["direct_effect"]
        full_effect = np.full(n_df, np.nan)
        full_effect[0] = self.contribution_dict["full_effect"]
        explained_effect = np.full(n_df, np.nan)
        explained_effect[0] = self.contribution_dict["explained_effect"]

        for i, mediator in enumerate(mediators):
            explained_effect[i + 1] = self.contribution_dict[mediator]

        self.summary_table = (
            pd.DataFrame(
                {
                    "direct_effect": direct_effect,
                    "full_effect": full_effect,
                    "explained_effect": explained_effect,
                },
                index=index,
            )
            .fillna("")
            .T
        )

        return self.summary_table

    def _bootstrap(self, rng: np.random.Generator):
        "Run a single bootstrap iteration."
        if self.cluster_df is not None:
            idx = rng.choice(
                self.unique_clusters, len(self.unique_clusters), replace=True
            )

            X_list = []
            Y_list = []

            for g in idx:
                X_list.append(self.X_dict[g])
                Y_list.append(self.Y_dict[g])

            X = np.concatenate(X_list)
            Y = np.concatenate(Y_list)

        else:
            idx = rng.choice(self.N, self.N)
            X = self.X[idx]
            Y = self.Y[idx]

        return self.fit(X=X, Y=Y, store=False)
