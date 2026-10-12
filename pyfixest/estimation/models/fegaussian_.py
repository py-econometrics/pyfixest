from __future__ import annotations

from typing import ClassVar

from pyfixest.estimation.internals.families import GAUSSIAN, GlmFamily
from pyfixest.estimation.internals.fit_statistics import (
    FitStatistics,
    linear_fit_statistics,
)
from pyfixest.estimation.internals.vcov_ import vcov_iid_ols
from pyfixest.estimation.internals.vcov_utils import VcovTerm
from pyfixest.estimation.models.feglm_ import Feglm


class Fegaussian(Feglm):
    "Class for the estimation of a fixed-effects GLM with normal errors."

    _family: ClassVar[GlmFamily] = GAUSSIAN

    def _vcov_iid(self) -> VcovTerm:
        # we set gaussian glms to match pf.feols exactly
        vcov = vcov_iid_ols(
            residuals=self.working_state.working_residuals,
            bread=self.sandwich.bread,
            N=self.sample_info.n_obs,
            weights=self.observation_weights.values,
        )
        return VcovTerm(vcov=vcov, meat=None)

    def _fit_statistics(self) -> FitStatistics:
        """Add the linear fit statistics to the deviance.

        Gaussian fits retain their demeaned response and residuals in
        working_state rather than the linear model's within_data and _u_hat.
        The identity link puts those arrays in the units of Y, so they can
        be passed to the same kernel used for OLS. The original response
        comes from model_matrix for the overall R².
        """
        working_state = self.working_state
        return linear_fit_statistics(
            Y=self.model_matrix.dependent.to_numpy(),
            Y_within=working_state.working_response_within.reshape((-1, 1)),
            residuals=working_state.response_residuals,
            weights=self.observation_weights.values,
            N=self.sample_info.n_obs,
            k=self._k,
            k_fe=self.fixef_counts.fixef_dof,
            has_intercept=not self.options.drop_intercept,
            has_fixef=self.model.has_fixef,
            deviance=super()._fit_statistics().deviance,
        )
