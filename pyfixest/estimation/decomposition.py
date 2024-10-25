from typing import Any, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


class LinearMediation:
    """
    Linear Mediation Model.

    Initial implementation by Apoorva Lal at
    https://gist.github.com/apoorvalal/e7dc9f3e52dcd9d51854b28b3e8a7ba4.
    """

    def __init__(
        self,
        agg: bool,
        param: str,
        coefnames: list[str],
        nthreads: int = -1,
        cluster_df: Optional[pd.Series] = None,
    ):
        self.cluster_dict: Optional[dict[Any, Any]]
        self.unique_clusters = Optional[np.ndarray[Any, Any]]

        self.param = param

        self.cluster_df = cluster_df
        if self.cluster_df is not None:
            self.unique_clusters = self.cluster_df.unique()
            self.cluster_dict = {
                cluster: self.cluster_df[self.cluster_df == cluster].index
                for cluster in self.unique_clusters
            }
        else:
            self.unique_clusters = None
            self.cluster_dict = None

        self.coefnames = coefnames
        # Get the names of the mediator variables
        self.mediator_names = [name for name in coefnames if param not in name]
        self.agg = agg
        self.nthreads = nthreads

        # prepare dicts for cluster bootstrapping


        self.X_dict: dict[Any, Any] = {}
        self.W_dict: dict[Any, Any] = {}
        self.y_dict: dict[Any, Any] = {}

    def fit(self, X: np.ndarray, W: np.ndarray, y: np.ndarray, store: bool = True):
        "Fit Linear Mediation Model."
        if store:
            self.X = X
            self.W = W
            self.y = y
            self.N = X.shape[0]
            self.Xk = X.shape[1]
            self.Wk = W.shape[1]

            self.beta_tilde = np.linalg.lstsq(X, y, rcond=1)[0]
            self.delta_tilde = np.linalg.lstsq(X, W, rcond=1)[0]
            self.gamma_tilde = np.linalg.lstsq(W, y, rcond=1)[0]
            self.total_effect = self.beta_tilde.flatten()
            self.mediated_effect = (
                (self.delta_tilde @ self.gamma_tilde).flatten()
                if self.agg
                else self.delta_tilde.flatten() * self.gamma_tilde.flatten()
            )
            self.direct_effect = self.total_effect - np.sum(self.mediated_effect)

            # prepare bootstrap in first iteration
            for g in self.unique_clusters:
                cluster_idx = np.where(self.cluster_df == g)[0]
                self.X_dict[g] = self.X[cluster_idx]
                self.W_dict[g] = self.W[cluster_idx]
                self.y_dict[g] = self.y[cluster_idx]

        else:
            beta_tilde = np.linalg.lstsq(X, y, rcond=1)[0]
            delta_tilde = np.linalg.lstsq(X, W, rcond=1)[0]
            gamma_tilde = np.linalg.lstsq(W, y, rcond=1)[0]
            total_effect = beta_tilde.flatten()
            mediated_effect = (
                (delta_tilde @ gamma_tilde).flatten()
                if self.agg
                else delta_tilde.flatten() * gamma_tilde.flatten()
            )
            direct_effect = total_effect - np.sum(mediated_effect)
            return np.concatenate(
                [total_effect, mediated_effect, direct_effect]
            ).flatten()

    def bootstrap(self, rng: np.random.Generator, B: int = 1_000, alpha: float = 0.05):
        "Bootstrap Confidence Intervals for Total, Mediated and Direct Effects."
        self.alpha = alpha
        self.B = B

        self._bootstrapped = np.c_[
            Parallel(n_jobs=self.nthreads)(
                delayed(self._bootstrap)(rng=rng) for _ in tqdm(range(B))
            )
        ]
        self.ci = np.percentile(
            self._bootstrapped, 100 * np.array([alpha / 2, 1 - alpha / 2]), axis=0
        )

    def summary(self) -> pd.DataFrame:

        "Summary Table for Total, Mediated and Direct Effects."
        effects = np.concatenate(
            [self.total_effect, self.mediated_effect, self.direct_effect], axis=0
        )
        summary_arr = np.concatenate([effects.reshape(1, -1), self.ci], axis=0)

        if self.agg:
            columns = (
                ["Total Effect"]
                + ["Mediated Effect"]
                + [f"Direct Effect: {self.param}"]
            )
        else:
            columns = (
                ["Total Effect"]
                + [f"Mediated Effect: {var}" for var in self.mediator_names]
                + [f"Direct Effect: {self.param}"]
            )

        lb = self.alpha / 2
        ub = 1 - lb
        index = ["Estimate", f"{lb*100:.1f}%", f"{ub*100:.1f}%"]

        self.summary_table = pd.DataFrame(
            summary_arr,
            columns=columns,
            index=index,
        ).T

        return self.summary_table

    def _bootstrap(self, rng: np.random.Generator):
        "Run a single bootstrap iteration."
        if self.cluster_df is not None:
            idx = rng.choice(
                self.unique_clusters, len(self.unique_clusters), replace=True
            )

            X_list = []
            W_list = []
            y_list = []

            for g in idx:

                X_list.append(self.X_dict[g])
                W_list.append(self.W_dict[g])
                y_list.append(self.y_dict[g])

            X = np.concatenate(X_list)
            W = np.concatenate(W_list)
            y = np.concatenate(y_list)

        else:
            idx = rng.choice(self.N, self.N)
            X = self.X[idx]
            W = self.W[idx]
            y = self.y[idx]

        return self.fit(X, W, y, store=False)
