from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.capabilities import (
    FREQUENCY_WEIGHTS,
    Capabilities,
    supported,
    unless,
)
from pyfixest.estimation.formula.model_matrix import ModelMatrix
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.families import GlmFamily
from pyfixest.estimation.internals.fit_glm_ import fit_glm_irls
from pyfixest.estimation.internals.literals import EstimatorKind
from pyfixest.estimation.internals.separation import check_for_separation
from pyfixest.estimation.internals.vcov_ import vcov_iid_glm
from pyfixest.estimation.models.base_regression_ import BaseRegression


class Feglm(BaseRegression):
    """
    Base class for the estimation of a fixed-effects GLM model.

    Returned by [feglm()](/reference/estimation.api.feglm.feglm.qmd). Fixed
    effects are handled via iteratively reweighted least squares with demeaning,
    following Stammann (2018),
    [arXiv:1707.01815](https://arxiv.org/pdf/1707.01815). The family is set with
    the `family` argument and implemented by a subclass. `poisson` dispatches to
    [Fepois](/reference/estimation.models.fepois_.Fepois.qmd).

    A GLM is a leaf of
    [BaseRegression](/reference/estimation.models.base_regression_.BaseRegression.qmd),
    not of `Feols`: IRLS leaves a working response, a working design, and
    working weights behind, so the post-estimation methods written against a
    single linear estimating equation are not defined here. See
    `capabilities()` for what a given fit supports.

    Examples
    --------
    ```{python}
    import numpy as np
    import pyfixest as pf

    data = pf.get_data()
    data["Y_bin"] = np.where(data["Y"] > 0, 1, 0)

    fit = pf.feglm("Y_bin ~ X1 + X2 | f1", data, family="logit")
    fit.tidy()
    ```
    """

    _estimator: ClassVar[EstimatorKind] = "feglm"
    # IRLS leaves a working response, a working design, and working weights
    # behind. Features that read those arrays as an OLS fit would resample or
    # refit the wrong quantities, so a GLM keeps only the covariance and
    # post-estimation paths written against its working state.
    _capabilities: ClassVar[Capabilities] = Capabilities(
        hac=unless(FREQUENCY_WEIGHTS),
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
        solver: Literal[
            "np.linalg.lstsq",
            "np.linalg.solve",
            "scipy.linalg.solve",
            "scipy.sparse.linalg.lsqr",
        ],
        family: GlmFamily,
        demeaner: AnyDemeaner | None = None,
        lookup_preconditioner: dict[frozenset[int], Preconditioner] | None = None,
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        sample_split_var: str | None = None,
        sample_split_value: str | int | None = None,
        separation_check: list[str] | None = None,
        context: int | Mapping[str, Any] = 0,
        accelerate: bool = True,
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
            solver=solver,
            store_data=store_data,
            copy_data=copy_data,
            lean=lean,
            sample_split_var=sample_split_var,
            sample_split_value=sample_split_value,
            context=context,
            demeaner=demeaner,
            lookup_preconditioner=lookup_preconditioner,
        )

        _glm_input_checks(
            drop_singletons=drop_singletons,
            tol=tol,
            maxiter=maxiter,
        )

        self.maxiter = maxiter
        self.tol = tol
        self.convergence = False
        self.separation_check = separation_check
        self._accelerate = accelerate

        self._Y_hat_response = np.empty(0)
        self.deviance = None

        self._method = "feglm"
        self._family = family
        self._inference_dist = family.inference_dist

    def prepare_model_matrix(self) -> ModelMatrix:
        "Prepare model inputs for estimation."
        model_matrix = super().prepare_model_matrix()

        # check for separation
        na_separation: list[int] = []
        if (
            self._fe is not None
            and self.separation_check is not None
            and self.separation_check  # not an empty list
        ):
            na_separation = check_for_separation(
                Y=model_matrix.dependent,
                X=model_matrix.independent,
                fe=model_matrix.fixed_effects,
                fml=self._fml,
                data=self._data,
                demeaner=self._demeaner,
                methods=self.separation_check,
            )

        if na_separation:
            self._data.drop(na_separation, axis=0, inplace=True)
            model_matrix = model_matrix.without_rows(na_separation)
            self._publish_model_matrix(model_matrix)

            self.n_separation_na = len(na_separation)
            # possible to have dropped fixed effects level due to separation
            self._n_fe = np.sum(self._k_fe > 1) if self._k_fe is not None else 0

        return model_matrix

    def get_fit(self) -> None:
        "Fit the GLM via IRLS and write results onto self.* attributes."
        model_matrix = self._model_matrix
        response = model_matrix.dependent.to_numpy()
        design = model_matrix.independent.to_numpy()
        fixed_effects = (
            None
            if model_matrix.fixed_effects is None
            else model_matrix.fixed_effects.to_numpy()
        )
        offset = (
            None
            if model_matrix.offset is None
            else model_matrix.offset.to_numpy().reshape((-1, 1))
        )
        self._offset = offset

        def _demean(
            v: np.ndarray, X: np.ndarray, weights: np.ndarray, tol: float
        ) -> tuple[np.ndarray, np.ndarray]:
            return self.residualize(
                v=v,
                X=X,
                flist=fixed_effects,
                weights=weights,
                tol=tol,
            )

        fit = fit_glm_irls(
            X=design,
            Y=response,
            family=self._family,
            demean=_demean,
            coefnames=self._coefnames,
            collin_tol=self._collin_tol,
            accelerate=self._accelerate and fixed_effects is not None,
            offset=offset,
            weights=self._observation_weights.values,
            solver=self._solver,
            maxiter=self.maxiter,
            tol=self.tol,
            fixef_tol=self._fixef_tol,
        )

        self._coefnames = fit.coefnames
        self._collin_vars = fit.collin_vars
        self._collin_index = fit.collin_index

        working_state = fit.working_state
        self._working_state = working_state
        design_within = working_state.design_within
        self._X_is_empty = design_within.shape[1] == 0
        self._k = design_within.shape[1]

        self._beta_hat = fit.beta
        self._Y_hat_response = working_state.mu
        self._Y_hat_link = working_state.eta

        self._u_hat_response = working_state.response_residuals
        self._u_hat_working = working_state.working_residuals
        self._u_hat = self._u_hat_working

        weighted_working_residuals = (
            working_state.working_weights * working_state.working_residuals
        )
        self._scores = design_within * weighted_working_residuals[:, None]

        weighted_design = working_state.working_weights[:, None] * design_within
        self._tZX = design_within.T @ weighted_design
        self._tZXinv = np.linalg.inv(self._tZX)

        self._hessian = self._tZX.copy()
        self.deviance = fit.deviance
        self.convergence = fit.converged
        if self.convergence:
            self._convergence = True

    # Read-only views on the final IRLS state. A GLM never builds the linear
    # `_within_data`, so these override the base aliases rather than extend them.

    @property
    def _Y(self) -> NDArray[np.float64]:
        """Within-scale IRLS working response."""
        return self._working_state.working_response_within

    @property
    def _X(self) -> NDArray[np.float64]:
        """Within-scale IRLS design, not premultiplied by working weights."""
        return self._working_state.design_within

    @property
    def _Z(self) -> NDArray[np.float64]:
        """The IRLS design; a GLM has no instruments."""
        return self._working_state.design_within

    @property
    def _irls_weights(self) -> NDArray[np.float64]:
        """Final IRLS working weights, never the observation weights."""
        return self._working_state.working_weights

    @property
    def _Xbeta(self) -> NDArray[np.float64]:
        """Linear predictor as a column vector."""
        return self._working_state.eta.reshape(-1, 1)

    def _clear_attributes(self) -> None:
        """Apply base cleanup and discard large GLM fit arrays when lean."""
        super()._clear_attributes()
        if self._lean:
            # The IRLS aliases are read-only views, so the working state
            # backing them is what has to go.
            for attr in (
                "_working_state",
                "_u_hat_response",
                "_u_hat_working",
                "_offset",
            ):
                if hasattr(self, attr):
                    delattr(self, attr)

    def _leverage_weights(self) -> np.ndarray:
        """Return final GLM working weights for HC2/HC3 leverage."""
        return self._working_state.working_weights

    def _fixef_weights(self) -> np.ndarray | None:
        """Return working weights when weighted FE recovery historically used them."""
        return self._working_state.working_weights if self._has_weights else None

    def _vcov_iid(self):
        return vcov_iid_glm(bread=self._bread)

    def resid(self, type: str = "response") -> np.ndarray:
        """
        Return residuals from a fitted GLM.

        Parameters
        ----------
        type : str, optional
            The type of residuals to return. Either "response" (default) or
            "working".

        Returns
        -------
        np.ndarray
            A flat array with the requested residuals.
        """
        if type not in ("response", "working"):
            raise ValueError("type must be one of 'response' or 'working'.")
        self._require_fit_arrays(
            "resid",
            arrays="the response and working residual arrays",
            remedy="Refit with lean=False to access residuals.",
        )
        if type == "response":
            return self._u_hat_response.flatten()
        return self._u_hat_working.flatten()

    def residualize(
        self,
        v: np.ndarray,
        X: np.ndarray,
        flist: np.ndarray | None,
        weights: np.ndarray,
        tol: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        "Residualize v and X by flist using weights."
        self._require_fit_arrays("residualize", arrays="the demeaning caches")
        if flist is None:
            return v, X

        effective_demeaner = self._demeaner.with_tol(tol)
        vX_tilde = self._demean_cache.demean_array(
            x=np.c_[v, X],
            flist=flist,
            weights=weights.flatten(),
            na_index=self._na_index,
            demeaner=effective_demeaner,
        )
        return vX_tilde[:, 0], vX_tilde[:, 1:]

    def _fixef_residual_target(
        self, response: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Residualize the estimated linear predictor instead of the response.

        Equation (5.2) in Stammann (2018),
        [arXiv:1707.01815](http://arxiv.org/abs/1707.01815). `_Y_hat_link`
        carries the offset as part of eta, so the offset is subtracted here:
        `_sumFE` then holds the pure fixed-effect contribution and `predict()`
        can add the offset back from `newdata` without double-counting it.
        """
        eta = self._Y_hat_link
        if self._offset_name is None:
            return eta
        assert self._offset is not None
        return eta - self._offset.flatten()

    def _response_from_link(
        self, link_predictor: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Apply the family's inverse link to the linear predictor."""
        return self._family.inv_link(link_predictor)

    def _check_dependent_variable(self) -> None:
        "Validate the dependent variable according to the family's constraints."
        self._family.check_y(self._model_matrix.dependent.to_numpy())

    def _validate_response(self) -> None:
        """Apply family-specific response validation after matrix preparation."""
        self._check_dependent_variable()


def _glm_input_checks(drop_singletons: bool, tol: float, maxiter: int) -> None:
    if not isinstance(drop_singletons, bool):
        raise TypeError("drop_singletons must be logical.")
    if not isinstance(tol, (int, float)):
        raise TypeError("tol must be numeric.")
    if tol <= 0 or tol >= 1:
        raise AssertionError("tol must be between 0 and 1.")
    if not isinstance(maxiter, int):
        raise TypeError("maxiter must be integer.")
    if maxiter <= 0:
        raise AssertionError("maxiter must be greater than 0.")
