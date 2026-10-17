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
    """Fitted Poisson-regression result returned by `pf.fepois()`.

    Access as ``fit = pf.fepois(formula, data)``. ``pf.feglm(...,
    family="poisson")`` returns the same result type. Use those functions to
    construct a fit. Estimation uses iteratively reweighted least squares.

    Attributes
    ----------
    options : GlmEstimationOptions
        Access as ``fit.options``: shared estimation settings, IRLS controls,
        separation check, acceleration, and offset expression.
    working_state : GlmWorkingState
        Final IRLS design, response, weights, predictors, and residuals.
    fitted_values : FittedValues
        Linear predictor and response mean in retained estimation-row order.
    sandwich : SandwichComponents
        IRLS scores, weighted Hessian, and its inverse.
    fitstat : FitStatistics
        Deviance, likelihood measures, and Pearson statistic. Null likelihood
        and pseudo R-squared are unavailable for weighted fits.
    convergence : bool
        Whether IRLS converged.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.fepois("Y ~ X1 + X2 | f1", pf.get_data(model="Fepois"))
    fit.tidy()
    fit.predict(type="response")[:3]
    ```

    Notes
    -----
    Use ``fit.coef().index`` for coefficient names. Shared result attributes and
    methods are listed in the
    [fitted-result overview](/reference-guide/fitted-results.qmd). Poisson
    supports fixed-effect recovery and unweighted randomization inference,
    but rejects wild bootstrap, causal cluster variance, decomposition,
    coefficient updates, and SAVI. See
    [model support](/reference-guide/model-support.qmd).

    Predictions default to the link scale. Recovered fixed effects are on that
    scale too. Prediction standard errors are unsupported, and interval
    arguments currently have no effect. Storage options can remove the
    arrays needed by later operations; see
    [model storage](/reference-guide/model-storage.qmd).

    See Also
    --------
    [fepois()](/reference/estimation.api.fepois.fepois.qmd),
    [Feglm](/reference/estimation.models.feglm_.Feglm.qmd),
    [model support](/reference-guide/model-support.qmd).
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
