from __future__ import annotations

from typing import ClassVar

from pyfixest.estimation.internals.families import POISSON, GlmFamily
from pyfixest.estimation.internals.fit_statistics import (
    FitStatistics,
    poisson_fit_statistics,
)
from pyfixest.estimation.internals.model_state import Capabilities
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

    _family: ClassVar[GlmFamily] = POISSON
    _declared_capabilities: ClassVar[Capabilities] = Capabilities(
        covariance_update=True,
        crv3_inference=True,
        hac_inference=True,
        multiway_clustering=True,
        wildboottest=False,
        cluster_causal_variance=False,
        decomposition=False,
        prediction=True,
        fixed_effect_recovery=True,
        randomization_inference=True,
        sherman_morrison_update=False,
        anytime_valid_inference=False,
    )

    def _fit_statistics(self) -> FitStatistics:
        "Add the Poisson likelihood measures to the deviance."
        # ``None`` is the allocation-free unweighted path shared with the rest
        # of the estimation core; no vector of ones is materialised.
        return poisson_fit_statistics(
            y=self.model_matrix.dependent.to_numpy().flatten(),
            mu=self.working_state.mu,
            weights=self.observation_weights.values,
            deviance=super()._fit_statistics().deviance,
        )
