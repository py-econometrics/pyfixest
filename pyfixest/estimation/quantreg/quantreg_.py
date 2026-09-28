import warnings
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, solve_triangular

from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.fit_statistics import FitStatistics
from pyfixest.estimation.internals.literals import QuantregMethodOptions
from pyfixest.estimation.internals.model_state import (
    FittedValues,
    ModelDescription,
    QuantregEstimationOptions,
    SampleSplit,
    VcovSpec,
    WithinLinearData,
)
from pyfixest.estimation.internals.retention import require_retained
from pyfixest.estimation.internals.vcov_utils import VcovTerm
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.quantreg.frisch_newton_ip import (
    QuantregSolution,
    frisch_newton_solver,
)
from pyfixest.estimation.quantreg.vcov_ import (
    vcov_crv1_qreg,
    vcov_hetero_qreg,
    vcov_iid_qreg,
    vcov_nid_qreg,
)
from pyfixest.utils.dev_utils import _find_stack_level


class Quantreg(Feols):
    """
    Quantile regression model.

    Returned by
    [quantreg()](/reference/estimation.api.quantreg.quantreg.qmd). Fits the
    conditional quantile of the outcome instead of the conditional mean, which
    allows the effect of a covariate to differ across the outcome distribution.
    Estimated via the interior point algorithm of Portnoy and Koenker (1997),
    [Statistical Science](https://doi.org/10.1214/ss/1030037960).

    Examples
    --------
    ```{python}
    import pyfixest as pf

    data = pf.get_data()

    fit = pf.quantreg("Y ~ X1 + X2", data, quantile=0.5)
    fit.tidy()
    ```

    Several quantiles can be estimated in one call.
    [qplot()](/reference/report.qplot.qmd) plots the resulting coefficients.

    ```{python}
    fits = pf.quantreg("Y ~ X1 + X2", data, quantile=[0.25, 0.5, 0.75])
    pf.etable(fits)
    ```

    See the [quantile regression tutorial](/tutorials/quantile-regression.qmd)
    for details.

    Attributes
    ----------
    solution : QuantregSolution
        Output of the interior point solver: coefficients, convergence flag,
        number of iterations, and final primal and dual iterates. With
        `multi_method="cfm2"`, the solver runs only for the central quantile;
        the other quantiles get a single Newton step and have no solution.
        Dropped by `lean=True`.
    """

    options: QuantregEstimationOptions
    # Quantile loss fit: no single least-squares solve to shortcut.
    _closed_form_ols = False

    # Set in get_fit().
    solution: QuantregSolution

    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        *,
        options: QuantregEstimationOptions,
        lookup_demeaned_data: dict[frozenset[int], DemeanedData],
        sample_split: SampleSplit | None = None,
    ) -> None:
        super().__init__(
            FixestFormula=FixestFormula,
            data=data,
            options=options,
            lookup_demeaned_data=lookup_demeaned_data,
            sample_split=sample_split,
        )

        warnings.warn(
            """
           The Quantile Regression implementation is experimental and may change in future releases.
           But mostly, we expect the API to remain unchanged.
           """,
            FutureWarning,
            stacklevel=_find_stack_level(),
        )

        self.capabilities = replace(
            self.capabilities,
            crv3_inference=False,
            hac_inference=False,
            multiway_clustering=False,
            wildboottest=False,
            cluster_causal_variance=False,
            decomposition=False,
            fixed_effect_recovery=False,
            randomization_inference=False,
            sherman_morrison_update=False,
        )

    def _describe_model(self, **kwargs: Any) -> ModelDescription:
        """Name the quantile solver and append the quantile to the model name."""
        description = super()._describe_model(**kwargs)
        return replace(
            description,
            method=f"quantreg_{self.options.method}",
            model_name=f"{description.model_name} (q = {self.options.quantile})",
        )

    def to_array(self):
        "Publish quantile-regression arrays from the formula state."
        response = self.model_matrix.dependent.to_numpy(dtype=np.float64)
        design = self.model_matrix.independent.to_numpy(dtype=np.float64)
        self.within_data = WithinLinearData(response=response, design=design)

    def drop_multicol_vars(self):
        """Remove collinear regressors using the same rank check as OLS.

        Quantile models do not support fixed effects, so within_data holds
        the original response and regressor arrays without demeaning.
        """
        self._set_within_data(self._drop_multicollinear_within_data(self.within_data))

    def prepare_model_matrix(self):
        "Prepare model inputs for estimation."
        super().prepare_model_matrix()

        if self.model_matrix.fixed_effects is not None:
            raise NotImplementedError(
                "Fixed effects are not yet supported for Quantile Regression."
            )

    def get_fit(self) -> None:
        """Fit a quantile regression model using the interior point method."""
        self.to_array()
        self.drop_multicol_vars()

        self._publish_solution(
            self._solve(
                X=self.within_data.design,
                Y=self.within_data.response,
                q=self.options.quantile,
            )
        )

    def _solve(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        q: float,
        beta_init: np.ndarray | None = None,
    ) -> QuantregSolution:
        """Solve quantile `q` with the solver and tolerances in `self.options`.

        Each solve seeds a fresh generator from `options.seed`, so a solve
        does not depend on the solves that ran before it.
        """
        options = self.options
        if options.method == "pfn":
            return self.fit_qreg_pfn(
                X=X,
                Y=Y,
                q=q,
                tol=options.quantile_tol,
                maxiter=options.quantile_maxiter,
                beta_init=beta_init,
                rng=np.random.default_rng(options.seed),
            )
        return self.fit_qreg_fn(
            X=X,
            Y=Y,
            q=q,
            tol=options.quantile_tol,
            maxiter=options.quantile_maxiter,
            beta_init=beta_init,
        )

    def _publish_solution(self, solution: QuantregSolution) -> None:
        """Publish an interior point solution and the coefficients it holds."""
        self.solution = solution
        self._publish_coefficients(solution.beta)

    def _publish_coefficients(self, beta_hat: np.ndarray) -> None:
        """Publish coefficients with the fitted values and residuals they imply."""
        self._beta_hat = beta_hat
        fitted = self.within_data.design @ beta_hat
        self.fitted_values = FittedValues(link=fitted, response=fitted)
        self._u_hat = self.within_data.response.flatten() - fitted

    def _fit_statistics(self) -> FitStatistics:
        """Leave the goodness-of-fit measures of a quantile fit undefined."""
        return FitStatistics()

    def fit_qreg_fn(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        q: float,
        tol: float | None = None,
        maxiter: int | None = None,
        beta_init: np.ndarray | None = None,
    ) -> QuantregSolution:
        """Fit a quantile regression model using the Frisch-Newton Interior Point Solver."""
        N, _ = X.shape
        if tol is None:
            tol = 1e-06
        if maxiter is None:
            maxiter = N

        # compute cholesky once outside of FN loop
        _chol, _ = cho_factor(X.T @ X, lower=True, check_finite=False)
        _chol = np.atleast_2d(_chol)
        _P = solve_triangular(_chol, X.T, lower=True, check_finite=False)

        solution = frisch_newton_solver(
            A=X.T,
            b=(1 - q) * X.T @ np.ones(N),
            c=-Y,
            u=np.ones(N),
            q=q,
            tol=tol,
            max_iter=maxiter,
            backoff=0.9995,
            beta_init=beta_init,
            chol=cast(np.ndarray, _chol),
            P=cast(np.ndarray, _P),
        )

        if not solution.has_converged:
            warnings.warn(
                f"The Frisch-Newton Interior Point solver has not converged after {solution.iterations} iterations.",
                RuntimeWarning,
                stacklevel=_find_stack_level(),
            )

        return solution

    def fit_qreg_pfn(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        q: float,
        m: float | None = None,
        tol: float | None = None,
        maxiter: int | None = None,
        beta_init: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        eta: float | None = None,
    ) -> QuantregSolution:
        """Fit a quantile regression model using the Frisch-Newton Interior Point Solver with pre-processing."""
        N, k = X.shape
        if tol is None:
            tol = 1e-06
        if maxiter is None:
            maxiter = N
        if rng is None:
            rng = np.random.default_rng()
        if beta_init is None:
            beta_init = np.zeros(k)
        if m is None:
            m = 0.8
        if eta is None:
            eta = 2 / 3

        max_bad_fixups = 3
        n_bad_fixups = 0

        has_converged = False
        compute_beta_init = beta_init is None

        # m constant should be set by user
        m = 0.8
        n_init = int(np.ceil((k * N) ** (eta)))
        M = int(np.maximum(N, np.ceil(m * n_init)))

        while not has_converged:
            if compute_beta_init:
                # get initial sample
                idx_init = rng.choice(N, size=n_init, replace=False)
                beta_hat_init = self.fit_qreg_fn(
                    X[idx_init, :], Y[idx_init], q=q, tol=tol, maxiter=maxiter
                ).beta

            else:
                beta_hat_init = beta_init

            r_init = Y.flatten() - X @ beta_hat_init
            # conservative estimate of sigma^2
            z = np.sqrt(np.dot(r_init, r_init) / (N - k))
            rz = r_init / z

            ql = max(0, q - M / (2 * N))
            qu = min(1, q + M / (2 * N))

            JL = rz < np.quantile(rz, ql)
            JH = rz > np.quantile(rz, qu)

            while not has_converged and n_bad_fixups < max_bad_fixups:
                keep = ~(JL | JH)
                X_sub = X[keep, :]
                Y_sub = Y[keep, :]

                if np.any(JL):
                    X_neg = np.sum(X[JL, :], axis=0)
                    Y_neg = np.sum(Y[JL])
                    X_sub = np.concatenate([X_sub, X_neg.reshape((1, self._k))], axis=0)
                    Y_sub = np.concatenate([Y_sub, Y_neg.reshape((1, 1))], axis=0)
                if np.any(JH):
                    X_pos = np.sum(X[JH, :], axis=0)
                    Y_pos = np.sum(Y[JH])
                    X_sub = np.concatenate([X_sub, X_pos.reshape(1, self._k)], axis=0)
                    Y_sub = np.concatenate([Y_sub, Y_pos.reshape((1, 1))], axis=0)

                # solve the modified problem
                solution = self.fit_qreg_fn(X=X_sub, Y=Y_sub, q=q)
                beta_hat = solution.beta

                r = Y.flatten() - X @ beta_hat

                # count wrong predictions and get their indices
                mis_L = JL & (r > 0)
                mis_H = JH & (r < 0)
                n_bad = np.sum(mis_L) + np.sum(mis_H)

                if n_bad == 0:
                    has_converged = True
                    break
                elif n_bad > 0.1 * M:
                    warnings.warn(
                        "Too many bad fixups. Doubling m.",
                        RuntimeWarning,
                        stacklevel=_find_stack_level(),
                    )
                    n_init = min(N, 2 * n_init)
                    M = int(np.ceil(m * n_init))
                    n_bad_fixups += 1
                    compute_beta_init = True
                    break

                else:
                    JL = JL & ~mis_L
                    JH = JH & ~mis_H

        if not has_converged:
            warnings.warn(
                "The Frisch-Newton Interior Point solver with preprocessing has not converged after 3 bad fixups.",
                RuntimeWarning,
                stacklevel=_find_stack_level(),
            )

        return solution

    def _vcov_iid(self) -> VcovTerm:
        vcov = vcov_iid_qreg(
            X=self.within_data.design,
            Y=self.within_data.response,
            u_hat=self._u_hat,
            q=self.options.quantile,
            N=self.sample_info.n_rows,
        )
        return VcovTerm(vcov=vcov, meat=None)

    def _vcov_hetero(self, *, vcov_type_detail: str) -> VcovTerm:
        vcov = vcov_hetero_qreg(
            X=self.within_data.design,
            Y=self.within_data.response,
            u_hat=self._u_hat,
            q=self.options.quantile,
            N=self.sample_info.n_rows,
        )
        return VcovTerm(vcov=vcov, meat=None)

    def _check_vcov_support(self, spec: VcovSpec) -> None:
        """Accept ``"nid"``, which only quantile regression supports."""
        if spec.vcov_type != "nid":
            super()._check_vcov_support(spec)

    def _vcov_nid(self) -> VcovTerm:
        """
        Compute nonparametric IID (NID) vcov matrix using the Hall-Sheather bandwidth
        as developed in Hendricks and Koenker (1991).
        Note: the estimator is actually heteroskedasticity robust, despite its name.
        'nid' stands for 'non-iid'.
        For details, see page 80 in Koenker's "Quantile Regression" (2005) book.
        """
        vcov = vcov_nid_qreg(
            X=self.within_data.design,
            Y=self.within_data.response,
            beta_hat=self._beta_hat,
            q=self.options.quantile,
            N=self.sample_info.n_rows,
            method=cast(QuantregMethodOptions, self.model.method),
            fit=self._solve,
        )
        return VcovTerm(vcov=vcov, meat=None)

    def _vcov_crv1(self, clustid: np.ndarray, cluster_col: np.ndarray) -> VcovTerm:
        """
        Implement cluster robust variance estimator for quantile regression following
        Parente and Santos Silva, 2016. Multiway clustering is rejected by
        ``vcov()`` through ``capabilities.multiway_clustering``.
        """
        vcov = vcov_crv1_qreg(
            X=self.within_data.design,
            u_hat=self._u_hat,
            q=self.options.quantile,
            clustid=clustid,
            cluster_col=cluster_col,
        )
        return VcovTerm(vcov=vcov, meat=None)

    def _finalize_fit(self) -> None:
        """Skip the OLS Wald test; quantile regression runs none at fit time."""

    @property
    def objective_value(self):
        "Compute the total loss of the quantile regression model."
        require_retained(self, "objective_value", "_u_hat")
        return np.sum(np.abs(self._u_hat) * (self.options.quantile - (self._u_hat < 0)))
