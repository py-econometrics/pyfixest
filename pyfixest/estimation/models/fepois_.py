from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from pyfixest.core.demean import Preconditioner
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.families import POISSON
from pyfixest.estimation.internals.fit_statistics import poisson_fit_statistics
from pyfixest.estimation.internals.model_state import (
    GlmEstimationOptions,
    ModelDescription,
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
    model_matrix : ModelMatrix
        Model frames containing the response, regressors, and other model inputs.
    observation_weights : ObservationWeights
        User-scale observation weights.
    working_state : GlmWorkingState
        Final within-scale IRLS design, response, weights, predictors and residuals.
    fitted_values : FittedValues
        The linear predictor and the response mean of the final IRLS iteration.
    sandwich : SandwichComponents
        IRLS scores, the Hessian X' W X with the final working weights, and its inverse.
    coefnames : list[str]
        Names of the coefficients in the design matrix X.
    options : GlmEstimationOptions
        The estimation options the model was built with, including the IRLS
        `maxiter` and `tol`, the separation check, and the offset.
    _data: pd.DataFrame
        The data frame used in the estimation. None if arguments `lean = True` or
        `store_data = False`.

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

    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        *,
        options: GlmEstimationOptions,
        lookup_demeaned_data: dict[frozenset[int], DemeanedData],
        lookup_preconditioner: dict[frozenset[int], Preconditioner] | None = None,
        sample_split_var: str | None = None,
        sample_split_value: str | int | None = None,
    ) -> None:
        super().__init__(
            FixestFormula=FixestFormula,
            data=data,
            options=options,
            lookup_demeaned_data=lookup_demeaned_data,
            lookup_preconditioner=lookup_preconditioner,
            sample_split_var=sample_split_var,
            sample_split_value=sample_split_value,
            family=POISSON,
        )

        # Poisson-specific overrides on top of the Feglm-set defaults.
        self.capabilities = replace(
            self.capabilities,
            crv3_inference=True,
            cluster_causal_variance=False,
            decomposition=False,
            randomization_inference=True,
        )

    def _describe_model(self, **kwargs: Any) -> ModelDescription:
        """Name the Poisson estimation function."""
        return replace(super()._describe_model(**kwargs), method="fepois")

    def get_fit(self) -> None:
        "Fit via Feglm IRLS, then add the Poisson likelihood measures."
        super().get_fit()
        y_orig = self.model_matrix.dependent.to_numpy().flatten()
        # ``None`` is the allocation-free unweighted path shared with the rest
        # of the estimation core; no vector of ones is materialised.
        observation_weights = self.observation_weights.values
        self.fitstat = poisson_fit_statistics(
            y=y_orig,
            mu=self.working_state.mu,
            weights=observation_weights,
            deviance=self._family.deviance(
                y_orig, self.working_state.mu, observation_weights
            ),
        )
