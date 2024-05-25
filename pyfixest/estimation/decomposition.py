import numpy as np
import pandas as pd

from joblib import Parallel, delayed


class CovariateDecomposition:
    def __init__(self):
        pass

    def fit(self, X, W, y, store=True):
        """Fit Linear Decomposition / Mediation Model.
        Args:
            X (2D Array): Treatment variable matrix (N x K)
            W (2D Array): Mediator variable matrix (N x L)
            y (1D Array): Outcome variable array (N x 1)
            store (bool, optional): Store estimates in class? Defaults to True.
                                    Same method is used for bootstrapping with False.
        """
        if store:
            self.beta_tilde = np.linalg.lstsq(X, y, rcond=1)[0]
            self.delta_tilde = np.linalg.lstsq(X, W, rcond=1)[0]
            self.gamma_tilde = np.linalg.lstsq(W, y, rcond=1)[0]
            self.total_effect, self.mediated_effect = (
                self.beta_tilde,
                self.delta_tilde @ self.gamma_tilde,
            )
            self.direct_effect = self.total_effect - self.mediated_effect
        else:
            beta_tilde = np.linalg.lstsq(X, y, rcond=1)[0]
            delta_tilde = np.linalg.lstsq(X, W, rcond=1)[0]
            gamma_tilde = np.linalg.lstsq(W, y, rcond=1)[0]
            total_effect, mediated_effect = beta_tilde, delta_tilde @ gamma_tilde
            direct_effect = total_effect - mediated_effect
            return total_effect, mediated_effect, direct_effect

    def bootstrap(self, B=1_000, alpha=0.05):
        """Bootstrap Confidence Intervals for Total, Mediated and Direct Effects."""
        self.alpha = alpha
        self.B = B
        self._bootstrapped = Parallel(n_jobs=-1)(
            delayed(self._bootstrap)() for _ in range(B)
        )
        self._bootstrapped = np.c_[self._bootstrapped]
        self.ci = np.percentile(
            self._bootstrapped, 100 * np.array([alpha / 2, 1 - alpha / 2]), axis=0
        )

    def summary(self):
        """Summary Table for Total, Mediated and Direct Effects."""
        self.total_effects_summary = np.c_[self.total_effect, self.ci[:, : self.K].T]
        self.mediated_effects_summary = np.c_[
            self.mediated_effect, self.ci[:, (self.K) : (self.K + self.K)].T
        ]
        self.direct_effects_summary = np.c_[self.direct_effect, self.ci[:, -self.K :].T]
        # summmary table omits intercept and handles single treatment
        # else use *_effects_summary arrays yourself
        self.summary_table = pd.DataFrame(
            {
                "Total Effect": self.total_effects_summary[1, :],
                "Mediated Effect": self.mediated_effects_summary[1, :],
                "Direct Effect": self.direct_effects_summary[1, :],
            },
            index=[
                "Estimate",
                f"CI Lower ({self.alpha/2})",
                f"CI Upper ({1-self.alpha/2})",
            ],
        )
        return self.summary_table

    def _bootstrap(self):
        """One replication of bootstrap."""
        idx = np.random.choice(self.N, self.N)
        X = self.X[idx]
        W = self.W[idx]
        y = self.y[idx]
        return self.fit(X, W, y, store=False)
