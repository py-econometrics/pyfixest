# from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm


class LinearMediation:
    """
    Linear Mediation Model.

    Initial implementation by Apoorva Lal at
    https://gist.github.com/apoorvalal/e7dc9f3e52dcd9d51854b28b3e8a7ba4.
    """

    def __init__(self, agg, param, coefnames):
        self.param = param
        self.coefnames = coefnames
        # Get the names of the mediator variables
        self.mediator_names = [name for name in coefnames if param not in name]
        self.agg = agg
        pass

    def fit(self, X, W, y, store=True):
        """Fit Linear Mediation Model
        Args:
            X (2D Array): Treatment variable matrix (N x K)
            W (2D Array): Mediator variable matrix (N x L)
            y (1D Array): Outcome variable array (N x 1)
            store (bool, optional): Store estimates in class? Defaults to True. Same method is used for bootstrapping with False.
        """
        self.N = X.shape[0]
        self.Xk = X.shape[1]
        self.Wk = W.shape[1]
        self.X = X
        self.W = W
        self.y = y

        if store:
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

    def bootstrap(self, rng, B=1_000, alpha=0.05):
        "Bootstrap Confidence Intervals for Total, Mediated and Direct Effects."
        self.alpha = alpha
        self.B = B
        # self._bootstrapped = Parallel(n_jobs=-1)(
        #    delayed(self._bootstrap)() for _ in range(B)
        # )

        self._bootstrapped = np.c_[[self._bootstrap(rng=rng) for _ in tqdm(range(B))]]
        self.ci = np.percentile(
            self._bootstrapped, 100 * np.array([alpha / 2, 1 - alpha / 2]), axis=0
        )

    def summary(self):
        "Summary Table for Total, Mediated and Direct Effects."
        effects = np.concatenate(
            [self.total_effect, self.mediated_effect, self.direct_effect], axis=0
        )
        summary_arr = np.concatenate([effects.reshape(1, -1), self.ci], axis=0)
        self.summary_table = pd.DataFrame(
            summary_arr,
            columns=["Total Effect:"]
            + [f"Mediated Effect: {var}" for var in self.mediator_names]
            + [f"Direct Effect: {self.param}"],
            index=[
                "Estimate",
                f"CI Lower ({self.alpha/2})",
                f"CI Upper ({1-self.alpha/2})",
            ],
        ).T

        return self.summary_table

    def _bootstrap(self, rng):
        "Run a single bootstrap iteration."
        idx = rng.choice(self.N, self.N)
        X = self.X[idx]
        W = self.W[idx]
        y = self.y[idx]
        return self.fit(X, W, y, store=False)
