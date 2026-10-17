from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pyfixest.core.demean import Preconditioner
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.collinearity import drop_multicollinear_variables
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.fit_ import fit_iv
from pyfixest.estimation.internals.fit_statistics import FitStatistics
from pyfixest.estimation.internals.model_state import (
    Capabilities,
    CollinearityCheck,
    EstimationOptions,
    FirstStage,
    FirstStageDiagnostics,
    FittedValues,
    ModelDescription,
    SampleSplit,
    WithinIvData,
    WithinLinearData,
)
from pyfixest.estimation.internals.retention import require_retained
from pyfixest.estimation.internals.vcov_ import meat_hetero
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.refit import refit
from pyfixest.utils.utils import get_ssc


class Feiv(Feols):
    """Fitted instrumental-variables result returned by `pf.feols()`.

    Access as ``fit = pf.feols("Y ~ X + [D ~ Z] | fe", data)``. The
    instrument block selects two-stage least squares instead of OLS. Construct
    this result through ``feols()``.

    Attributes
    ----------
    first_stage : FirstStage
        Access as ``fit.first_stage``: first-stage coefficients, within-scale
        predictions and residuals, fitted first-stage model, excluded-instrument
        names, and instrument-strength diagnostics.
    collinearity_instruments : CollinearityCheck
        Instrument columns retained or dropped by the rank check.
    within_data : WithinIvData
        Demeaned structural regressors, instruments, endogenous regressor, and
        response. Arrays are not multiplied by square-root observation weights.
    sandwich : SandwichComponents
        Weighted scores from the first-stage projection, 2SLS Hessian, and bread.
    capabilities : Capabilities
        Supported inference and post-estimation features.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X2 + [X1 ~ Z1] | f1", pf.get_data())
    fit.tidy()
    fit.first_stage.diagnostics.f_stat
    ```

    Inspect the first-stage regression with ``fit.first_stage.model.tidy()``.
    Run ``fit.IV_Diag()`` to publish additional instrument-strength diagnostics.

    Notes
    -----
    Shared result attributes and coefficient/reporting operations are described
    in the [fitted-result overview](/reference-guide/fitted-results.qmd).
    IV fits do not support prediction, fixed-effect recovery, CRV3, wild
    bootstrap, randomization inference, causal cluster variance, decomposition,
    coefficient updates, or SAVI. See
    [model support](/reference-guide/model-support.qmd).

    The first-stage model follows the same storage policy as the second stage;
    ``lean=True`` removes arrays needed to recompute effective F statistics.
    See [model storage](/reference-guide/model-storage.qmd).

    See Also
    --------
    [feols()](/reference/estimation.api.feols.feols.qmd),
    [FirstStage](/reference/estimation.state.FirstStage.qmd),
    [instrumental variables tutorial](/tutorials/instrumental-variables.qmd).
    """

    # Set in _fit_first_stage().
    first_stage: FirstStage
    # Set in get_fit().
    collinearity_instruments: CollinearityCheck
    # Two-stage fit: no single least-squares solve to shortcut.
    _closed_form_ols = False

    _declared_capabilities: ClassVar[Capabilities] = Capabilities(
        covariance_update=True,
        crv3_inference=False,
        hac_inference=True,
        multiway_clustering=True,
        wildboottest=False,
        cluster_causal_variance=False,
        decomposition=False,
        prediction=False,
        fixed_effect_recovery=False,
        randomization_inference=False,
        sherman_morrison_update=False,
        anytime_valid_inference=False,
    )

    # Constructor and methods implementation...
    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        *,
        options: EstimationOptions,
        lookup_demeaned_data: dict[frozenset[int], DemeanedData],
        lookup_preconditioner: dict[frozenset[int], Preconditioner] | None = None,
        sample_split: SampleSplit | None = None,
    ) -> None:
        super().__init__(
            FixestFormula=FixestFormula,
            data=data,
            options=options,
            lookup_demeaned_data=lookup_demeaned_data,
            lookup_preconditioner=lookup_preconditioner,
            sample_split=sample_split,
        )

    def _publish_model_matrix(self, model_matrix):
        """Publish the base model-matrix state plus the instrument names."""
        super()._publish_model_matrix(model_matrix)
        self._coefnames_z: list[str] = model_matrix.instruments.columns.tolist()

    def _describe_model(self, **kwargs: Any) -> ModelDescription:
        """Describe the second stage of an instrumental-variable fit."""
        return replace(super()._describe_model(**kwargs), is_iv=True)

    def _demean(self) -> WithinIvData:
        """Return second-stage and full instrument arrays on within scale."""
        linear_data = super()._demean()
        endogenous_frame = self.model_matrix.endogenous
        instrument_frame = self.model_matrix.instruments
        assert endogenous_frame is not None
        assert instrument_frame is not None
        endogenous = endogenous_frame.to_numpy(dtype=np.float64)
        instruments = instrument_frame.to_numpy(dtype=np.float64)
        fixed_effects = self.model_matrix.fixed_effects
        if fixed_effects is not None:
            endogenous, instruments, _ = self._demean_cache.demean_yx(
                endogenous,
                instruments,
                y_keys=self.model_matrix.column_keys("endogenous"),
                x_keys=self.model_matrix.column_keys("instruments"),
                fe=fixed_effects.to_numpy(),
                weights=self.observation_weights.values,
                na_index=self.sample_info.dropped_row_index,
                demeaner=self.options.demeaner,
            )
        return WithinIvData(
            response=linear_data.response,
            design=linear_data.design,
            instruments=instruments,
            endogenous=endogenous,
        )

    def _drop_multicollinear_within_data(
        self, within_data: WithinLinearData
    ) -> WithinLinearData:
        """Drop collinear columns from the second-stage design and the instruments."""
        within_data = super()._drop_multicollinear_within_data(within_data)
        assert isinstance(within_data, WithinIvData)
        instruments, collinearity = drop_multicollinear_variables(
            within_data.instruments,
            self._coefnames_z,
            self.options.collin_tol,
        )
        self.collinearity_instruments = collinearity
        self._coefnames_z = list(collinearity.coefnames)
        return replace(within_data, instruments=instruments)

    def get_fit(self) -> None:
        """Fit a IV model using a 2SLS estimator."""
        within_data = self._drop_multicollinear_within_data(self._demean())
        # Narrow the base return type so `within_data.instruments` type-checks.
        assert isinstance(within_data, WithinIvData)
        self._set_within_data(within_data)
        fit = fit_iv(
            X=within_data.design,
            Z=within_data.instruments,
            Y=within_data.response,
            weights=self.observation_weights.values,
            solver=self.options.solver,
        )

        self._beta_hat = fit.beta
        self._u_hat = fit.residuals
        self.sandwich = fit.sandwich

        # The response minus the residual carries the fixed-effect
        # contribution, which `design @ beta_hat` alone would omit.
        fitted = self.model_matrix.dependent.to_numpy().flatten() - self.resid()
        self.fitted_values = FittedValues(link=fitted, response=fitted)

    def _fit_statistics(self) -> FitStatistics:
        """Leave the goodness-of-fit measures of a 2SLS fit undefined."""
        return FitStatistics()

    def _fit_first_stage(self) -> None:
        """Fit the first-stage regression and publish it as `first_stage`."""
        require_retained(self, "_fit_first_stage", "_data")
        # The excluded instruments are the instrument-matrix columns that are
        # not also second-stage regressors, kept in instrument-matrix order.
        exogenous = set(self._coefnames)
        instruments = tuple(
            str(name) for name in self._coefnames_z if name not in exogenous
        )

        # As in fixest, the first stage uses the second stage's rows and options.
        model1 = refit(
            self,
            formula=self.model.parsed_formula.as_first_stage(),
            data=self._data,
            vcov=self.variance_covariance.spec,
            same_sample=True,
        )

        self.first_stage = FirstStage(
            coefficients=model1._beta_hat,
            # note that model1.within_data.design is demeaned
            fitted_values=model1.within_data.design @ model1._beta_hat,
            residuals=model1._u_hat,
            model=model1,
            instruments=instruments,
            diagnostics=first_stage_f_test(
                model=model1,
                instrument_positions=_instrument_positions(
                    model=model1, instruments=instruments
                ),
            ),
        )

    def _finalize_fit(self) -> None:
        """Fit and retain the first-stage model after second-stage inference."""
        self._fit_first_stage()

    def _clear_attributes(self) -> None:
        """Apply the retention policy to this model and its first stage."""
        first_stage = getattr(self, "first_stage", None)
        if first_stage is not None:
            # the first stage replays this model's storage options
            first_stage.model._clear_attributes()
        super()._clear_attributes()

    def IV_Diag(self, statistics: list[str] | None = None):
        """Compute weak-instrument diagnostics for the first stage.

        Recomputes the first-stage F-statistic and the effective F-statistic
        (Olea and Pflueger, 2013) and stores them in
        `fit.first_stage.diagnostics`. The method returns `None`.

        Parameters
        ----------
        statistics : list[str], optional
            Diagnostic statistics to compute, out of `"f_stat"` and
            `"effective_f"`. Currently the argument is only validated: both
            statistics are always computed, and other values raise a
            `ValueError`.

        Notes
        -----
        Effective F recomputation needs retained first-stage within-scale data
        and weights. Refit with ``lean=False`` if those arrays were discarded.

        The first-stage F-statistic is adjusted to the model's vcov
        specification.

        Examples
        --------
        ```{python}
        import numpy as np
        import pandas as pd
        import pyfixest as pf

        rng = np.random.default_rng(1)
        n = 1000

        z = rng.binomial(1, 0.5, size=n)
        d = 0.5 * z + rng.normal(size=n)
        c1 = rng.normal(size=n)
        y = 1.0 + 1.5 * d + 0.8 * c1 + rng.normal(size=n)
        data = pd.DataFrame({"y": y, "d": d, "z": z, "c1": c1})

        fit_iv = pf.feols("y ~ c1 + [d ~ z]", data=data, vcov="hetero")
        fit_iv.IV_Diag()
        fit_iv.first_stage.diagnostics
        ```

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd).
        """
        # Set default statistics
        iv_diag_stat = ["f_stat", "effective_f"]

        # Set statistics allowed in the current version
        iv_diag_stat_allowed = ["f_stat", "effective_f"]

        # Check whether there is unsupported statistics.
        if statistics:
            invalid_stats = [
                stat for stat in statistics if stat not in iv_diag_stat_allowed
            ]

            if invalid_stats:
                raise ValueError(
                    f"Statistics not supported: {invalid_stats}."
                    f"You should specify from the following list of statistics {iv_diag_stat_allowed}"
                )

            iv_diag_stat += statistics

        self.IV_weakness_test(iv_diag_stat)

    def IV_weakness_test(self, iv_diag_statistics: list[str] | None = None) -> None:
        """Implement IV weakness test (F-test).

        This method covers hetero-robust and clustered-robust F statistics.
        It republishes `first_stage` with two updated statistics:

        - `first_stage.diagnostics.f_stat`: F statistic of the first stage
        - `first_stage.diagnostics.eff_f`: effective F statistic
          (Olea and Pflueger 2013) of the first stage

        Notes
        -----
        `f_stat` is adjusted to the specification of vcov.

        Parameters
        ----------
        iv_diag_statistics : list, optional
            Names to recompute: ``"f_stat"`` and/or ``"effective_f"``.
            The default ``None`` performs no update. Unknown names are ignored.

        Returns
        -------
        None
            Republishes the selected fields in ``fit.first_stage.diagnostics``.

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd).
        """
        iv_diag_statistics = iv_diag_statistics or []

        if "f_stat" in iv_diag_statistics:
            published = self.first_stage
            diagnostics = first_stage_f_test(
                model=published.model,
                instrument_positions=_instrument_positions(
                    model=published.model, instruments=published.instruments
                ),
            )
            self.first_stage = replace(
                published,
                diagnostics=replace(diagnostics, eff_f=published.diagnostics.eff_f),
            )

        if "effective_f" in iv_diag_statistics:
            self.eff_F()

    def eff_F(self) -> None:
        """Compute and store the effective first-stage F statistic.

        Uses Olea and Pflueger (2013),
        [Journal of Business & Economic Statistics](https://doi.org/10.1080/00401706.2013.806694).
        For an iid fit, computes a heteroskedasticity-robust instrument covariance;
        otherwise uses the first-stage model's current covariance.

        Returns
        -------
        None
            Updates ``fit.first_stage.diagnostics.eff_f``.

        Notes
        -----
        Requires retained first-stage within-scale arrays, observation weights,
        and sandwich components. Refit with ``lean=False`` when needed.

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd).
        """
        published = self.first_stage
        model = published.model
        require_retained(model, "eff_F", "within_data", "observation_weights")

        instrument_positions = _instrument_positions(
            model=model, instruments=published.instruments
        )

        if model.variance_covariance.spec.vcov_type_detail == "iid":
            observation_weights = model.observation_weights.values
            hetero_meat = meat_hetero(
                sandwich=model.sandwich,
                X=model.within_data.design,
                frequency_weights=(
                    observation_weights.reshape((-1, 1))
                    if observation_weights is not None
                    and model.options.weights_type == "fweights"
                    else None
                ),
                normal_equation_weights=observation_weights,
                vcov_type_detail="hetero",
            )
            bread = model.sandwich.bread
            correction = get_ssc(
                model.options.ssc,
                model._dof_counts(G=model.sample_info.n_obs),
                vcov_type="hetero",
            )
            vcv = bread @ (hetero_meat * correction.adj) @ bread
        else:
            vcv = model.variance_covariance.vcov

        eff_f = effective_f_statistic(
            pi_hat=model._beta_hat[instrument_positions],
            instruments_within=model.within_data.design[:, instrument_positions],
            instrument_vcov=vcv[np.ix_(instrument_positions, instrument_positions)],
            weights=model.observation_weights.values,
        )
        self.first_stage = replace(
            published, diagnostics=replace(published.diagnostics, eff_f=eff_f)
        )


def _instrument_positions(*, model: Feols, instruments: tuple[str, ...]) -> list[int]:
    """Locate the excluded instruments among the first-stage coefficients."""
    coefnames = list(model._coefnames)
    return [coefnames.index(instrument) for instrument in instruments]


def first_stage_f_test(
    *, model: Feols, instrument_positions: list[int]
) -> FirstStageDiagnostics:
    r"""Test that the excluded instruments are jointly irrelevant.

    Wald test of

    H0 : \beta_{z_1} = 0 & ... & \beta_{z_{p_iv}} = 0
         where z_1, ..., z_{p_iv} are the excluded instruments
    H1 : H0 does not hold

    under the first stage's own covariance estimator, so the statistic is
    heteroskedasticity- or cluster-robust whenever that estimator is.

    Parameters
    ----------
    model : Feols
        The fitted first-stage model.
    instrument_positions : list[int]
        Positions of the excluded instruments among `model`'s coefficients.

    Returns
    -------
    FirstStageDiagnostics
        The F statistic and its p-value; `eff_f` is not computed here.
    """
    # Pad an identity matrix of size p_iv by p_iv with zeros to select the
    # excluded instruments out of the first stage's k coefficients.
    p_iv = len(instrument_positions)
    R = np.zeros((p_iv, model._k))
    R[:, instrument_positions] = np.eye(p_iv)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wald = model.wald_test(R=R)
    return FirstStageDiagnostics(
        f_stat=wald.f_statistic, p_value=wald.pvalue, eff_f=None
    )


def effective_f_statistic(
    *,
    pi_hat: NDArray[np.float64],
    instruments_within: NDArray[np.float64],
    instrument_vcov: NDArray[np.float64],
    weights: NDArray[np.float64] | None,
) -> float:
    """Compute the effective F statistic of Olea and Pflueger (2013).

    With the excluded instruments Z on within scale, the first-stage
    coefficients pi on those instruments, and their covariance Sigma,

        F_eff = pi' Q_zz pi / trace(Sigma Q_zz),   Q_zz = Z' W Z.

    See [Olea and Pflueger
    (2013)](https://doi.org/10.1080/00401706.2013.806694).

    Parameters
    ----------
    pi_hat : NDArray[np.float64]
        First-stage coefficients on the excluded instruments, shape (p_iv,).
    instruments_within : NDArray[np.float64]
        Within-scale excluded instruments, shape (n_rows, p_iv).
    instrument_vcov : NDArray[np.float64]
        Heteroskedasticity-robust covariance of `pi_hat`, shape (p_iv, p_iv).
    weights : NDArray[np.float64] or None
        Observation weights, or `None` for an unweighted fit.

    Returns
    -------
    float
        The effective F statistic.
    """
    Z = instruments_within
    Q_zz = Z.T @ Z if weights is None else Z.T @ (weights[:, None] * Z)
    return float((pi_hat.T @ Q_zz @ pi_hat) / np.sum(np.diag(instrument_vcov @ Q_zz)))
