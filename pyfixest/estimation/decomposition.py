from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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
        self.mask = np.ones(len(self.coefnames), dtype=bool)
        self.mask[param_idx] = False

        self.mediator_names = [name for name in self.coefnames if self.param not in name]

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

    def fit(self, X: np.ndarray, Y: np.ndarray, store: bool = True):
        "Fit Linear Mediation Model."
        if store:

            self.X = X
            self.X1 = self.X[:,self.mask]
            self.X2 = self.X[:,~self.mask]
            self.Y = Y
            self.N = X.shape[0]

            # Gelbach Method:
            self.gamma = np.linalg.lstsq(self.X2, self.X1, rcond=1)[0]
            self.beta_full = np.linalg.lstsq(self.X, self.Y, rcond = 1)[0]
            self.beta2 = self.beta_full[self.mask]
            self.delta = self.gamma * self.beta2.flatten()

            if self.agg:
                self.delta = np.array([np.sum(self.delta)])

            # prepare bootstrap in first iteration
            if self.cluster_df is not None:
                for g in self.unique_clusters:
                    cluster_idx = np.where(self.cluster_df == g)[0]
                    self.X_dict[g] = self.X[cluster_idx]
                    self.Y_dict[g] = self.Y[cluster_idx]

            return self.delta.flatten()

        else:
            # Gelbach Method:
            X1 = X[:,self.mask]
            X2 = X[:,~self.mask]
            gamma = np.linalg.lstsq(X2, X1, rcond=1)[0]
            beta_full = np.linalg.lstsq(X, Y, rcond = 1)[0]
            beta2 = beta_full[self.mask]
            delta = gamma * beta2.flatten()

            if self.agg:
                delta = np.array([np.sum(delta)])

            return delta.flatten()

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
        #import pdb; pdb.set_trace()
        summary_arr = np.concatenate([self.delta.reshape(1,-1), self.ci], axis = 0)

        lb = self.alpha / 2
        ub = 1 - lb
        index = ["Estimate", f"{lb*100:.1f}%", f"{ub*100:.1f}%"]

        columns = ["delta (agg):"] if self.agg else [f"delta {x}:" for x in self.mediator_names]

        summary_table = pd.DataFrame(
            summary_arr,
            columns=columns,
            index=index,
        ).T

        # drop intercept
        self.summary_table = summary_table[~summary_table.index.str.contains("Intercept", regex=True)]
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

        return self.fit(X = X, Y = Y, store=False)
