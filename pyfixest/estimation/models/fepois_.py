from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy.special import gammaln

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.capabilities import (
    FREQUENCY_WEIGHTS,
    WEIGHTED,
    Capabilities,
    supported,
    unless,
)
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.families import POISSON
from pyfixest.estimation.internals.literals import (
    EstimatorKind,
    SolverOptions,
)
from pyfixest.estimation.models.feglm_ import Feglm


class Fepois(Feglm):
    """
    Estimate a Poisson regression model.

    Non user-facing class to estimate a Poisson regression model via Iterated
    Weighted Least Squares (IWLS).

    Inherits from the Feglm class. Users should not directly instantiate this class,
    but rather use the [fepois()](/reference/estimation.api.fepois.fepois.qmd) function.
    IRLS residualization is orchestrated by ``Feglm`` through the shared
    ``DemeanCache`` supplied by the estimation runner.

    The method implements the algorithm from Stata's `ppmlhdfe` module.

    Attributes
    ----------
    _Y : np.ndarray
        Final within-scale IRLS working response. It is not multiplied by the
        square root of the working weights.
    _X : np.ndarray
        Final within-scale IRLS design. It is not multiplied by the square root
        of the working weights.
    _fe : pd.DataFrame or None
        Formula-scale fixed effects.
    _weights : np.ndarray
        Compatibility alias containing observation weights only.
    _irls_weights : np.ndarray
        Final IRLS working weights.
    coefnames : list[str]
        Names of the coefficients in the design matrix X.
    drop_singletons : bool
        Whether to drop singleton fixed effects.
    collin_tol : float
        Tolerance level for the detection of collinearity.
    maxiter : Optional[int], default=25
        Maximum number of iterations for the IRLS algorithm.
    tol : Optional[float], default=1e-08
        Tolerance level for the convergence of the IRLS algorithm.
    solver : str, optional.
        The solver to use for the regression. Can be "np.linalg.lstsq",
        "np.linalg.solve", "scipy.linalg.solve" and "scipy.sparse.linalg.lsqr".
        Defaults to "scipy.linalg.solve".
    demeaner : Optional[AnyDemeaner]
        Resolved typed demeaner configuration.
    fixef_tol: float, default = 1e-06.
        Tolerance level for the convergence of the demeaning algorithm.
    context : int or Mapping[str, Any]
        A dictionary containing additional context variables to be used by
        formulaic during the creation of the model matrix. This can include
        custom factorization functions, transformations, or any other
        variables that need to be available in the formula environment.
    weights_name : Optional[str]
        Name of the weights variable.
    weights_type : Optional[str]
        Type of weights variable.
    _data: pd.DataFrame
        The data frame used in the estimation. Deleted if arguments `lean = True`
        or `store_data = False`.

    Examples
    --------
    `Fepois` is returned by
    [fepois()](/reference/estimation.api.fepois.fepois.qmd) and is not
    constructed directly. Post-estimation methods are inherited from
    [Feols](/reference/estimation.models.feols_.Feols.qmd).

    ```{python}
    import pyfixest as pf

    data = pf.get_data(model="Fepois")
    fit = pf.fepois("Y ~ X1 + X2 | f1", data)

    fit.tidy()
    ```
    """

    _estimator: ClassVar[EstimatorKind] = "fepois"
    # The fast randomization-inference path resamples a linear design; the
    # Poisson working arrays are not one, so RI always replays `fepois()`.
    _ritest_forces_slow_algorithm: ClassVar[bool] = True
    # Poisson adds the two paths whose refits the class can replay: the
    # longstanding CRV3 jackknife and randomization inference, which both
    # re-estimate through the public Poisson API rather than reusing the IRLS
    # working arrays.
    _capabilities: ClassVar[Capabilities] = Capabilities(
        crv3=supported(),
        hac=unless(FREQUENCY_WEIGHTS),
        ritest=unless(WEIGHTED),
        fixef=supported(),
        predict=supported(),
    )

    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        ssc_dict: dict[str, str | bool],
        drop_singletons: bool,
        drop_intercept: bool,
        weights: str | None,
        weights_type: str | None,
        collin_tol: float,
        lookup_demeaned_data: dict[frozenset[int], DemeanedData],
        tol: float,
        maxiter: int,
        solver: SolverOptions = "np.linalg.solve",
        demeaner: AnyDemeaner | None = None,
        lookup_preconditioner: dict[frozenset[int], Preconditioner] | None = None,
        context: int | Mapping[str, Any] = 0,
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        sample_split_var: str | None = None,
        sample_split_value: str | int | None = None,
        separation_check: list[str] | None = None,
        offset: str | None = None,
    ) -> None:
        super().__init__(
            FixestFormula=FixestFormula,
            data=data,
            ssc_dict=ssc_dict,
            drop_singletons=drop_singletons,
            drop_intercept=drop_intercept,
            weights=weights,
            weights_type=weights_type,
            collin_tol=collin_tol,
            lookup_demeaned_data=lookup_demeaned_data,
            tol=tol,
            maxiter=maxiter,
            solver=solver,
            store_data=store_data,
            copy_data=copy_data,
            lean=lean,
            sample_split_var=sample_split_var,
            sample_split_value=sample_split_value,
            separation_check=separation_check,
            context=context,
            demeaner=demeaner,
            lookup_preconditioner=lookup_preconditioner,
            family=POISSON,
        )

        # Poisson-specific overrides on top of the Feglm-set defaults.
        self._method = "fepois"
        self._offset_name = offset

    def get_fit(self) -> None:
        "Fit via Feglm IRLS, then add Poisson-specific post-fit summary stats."
        y_orig = self._model_matrix.dependent.to_numpy().flatten()
        observation_weights = self._observation_weights.values
        user_weights = (
            np.ones_like(y_orig, dtype=np.float64)
            if observation_weights is None
            else observation_weights
        )

        super().get_fit()

        self._y_hat_null = np.full_like(
            y_orig, np.average(y_orig, weights=user_weights), dtype=float
        )

        self._loglik = np.sum(
            user_weights
            * (
                y_orig * np.log(self._Y_hat_response)
                - self._Y_hat_response
                - gammaln(y_orig + 1)
            )
        )

        # cant replicate fixest atm
        if self._has_weights:
            self._loglik_null = None
            self._pseudo_r2 = None
        else:
            self._loglik_null = np.sum(
                user_weights
                * (
                    y_orig * np.log(self._y_hat_null)
                    - self._y_hat_null
                    - gammaln(y_orig + 1)
                )
            )
            self._pseudo_r2 = 1 - (self._loglik / self._loglik_null)
        self._pearson_chi2 = np.sum(
            user_weights * (y_orig - self._Y_hat_response) ** 2 / self._Y_hat_response
        )

        self.deviance = self._family.deviance(
            y_orig, self._Y_hat_response, user_weights
        )

    def _estimation_refit_kwargs(self) -> dict[str, Any]:
        """Return the full Poisson estimation contract for data-changing refits."""
        kwargs = super()._estimation_refit_kwargs()
        kwargs.update(
            {
                "offset": self._offset_name,
                "iwls_tol": self.tol,
                "iwls_maxiter": self.maxiter,
                "separation_check": (
                    None
                    if self.separation_check is None
                    else list(self.separation_check)
                ),
            }
        )
        return kwargs

    def _crv3_refit(self, data: pd.DataFrame) -> Fepois:
        """Replay Poisson estimation for one leave-one-cluster-out sample."""
        # lazy loading to avoid circular import
        fixest_module = import_module("pyfixest.estimation")
        return fixest_module.fepois(
            fml=self._fml,
            data=data,
            vcov="iid",
            **self._estimation_refit_kwargs(),
        )

    def _clear_attributes(self) -> None:
        """Apply GLM cleanup and discard the Poisson null-fit array when lean."""
        super()._clear_attributes()
        if self._lean and hasattr(self, "_y_hat_null"):
            del self._y_hat_null
