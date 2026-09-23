from __future__ import annotations

import warnings
from dataclasses import replace
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import LsmrDemeaner
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.collinearity import drop_multicollinear_variables
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.fit_ import fit_iv
from pyfixest.estimation.internals.model_state import (
    CollinearityCheck,
    EstimationOptions,
    FirstStage,
    FirstStageDiagnostics,
    FittedValues,
    ModelDescription,
    WithinIvData,
    WithinLinearData,
)
from pyfixest.estimation.internals.retention import require_retained
from pyfixest.estimation.internals.vcov_ import meat_hetero
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.utils.utils import get_ssc


class Feiv(Feols):
    """
    Non user-facing class to estimate an IV model using a 2SLS estimator.

    Inherits from the Feols class. Users should not directly instantiate this class,
    but rather use the [feols()](/reference/estimation.api.feols.feols.qmd)
    function. This class constructs the second-stage and instrument within
    arrays through the shared ``DemeanCache`` supplied by the estimation runner.

    Parameters
    ----------
    FixestFormula : Formula
        Parsed fixest formula, including the first stage.
    data : pd.DataFrame
        Estimation data, already converted to pandas and reindexed.
    options : EstimationOptions
        Every estimation option the fit is built with, assembled from the
        `EstimationConfig` by the estimation planner.
    lookup_demeaned_data : dict[frozenset[int], DemeanedData]
        Demeaning cache shared across the models of one cache block.
    lookup_preconditioner : Optional[dict[frozenset[int], Preconditioner]]
        Preconditioner cache shared across the models of one cache block.
    sample_split_var : Optional[str]
        Name of the sample-split variable, or ``None`` for the full sample.
    sample_split_value : Optional[str | int]
        Value of `sample_split_var` this model is fitted on.

    Attributes
    ----------
    _Z : np.ndarray
        Processed instruments after handling multicollinearity.
    _weights_type_feiv : str
        Type of the weights variable defined in Feiv class.
        Either "aweights" for analytic weights or "fweights"
        for frequency weights.
    _coefnames_z : list
        Names of coefficients for Z after handling multicollinearity.
    collinearity_instruments : CollinearityCheck
        Names and column mask of the instruments dropped by the rank check,
        set in get_fit().
    capabilities : Capabilities
        Inference and post-estimation features this model class supports.
    sandwich : SandwichComponents
        Weighted scores of the first-stage projection X_hat, the 2SLS Hessian
        X_hat' W X_hat, and its inverse, set in get_fit().
    _beta_hat : np.ndarray
        Estimated regression coefficients.
    fitted_values : FittedValues
        In-sample predictions on the link and the response scale, set in
        get_fit().
    _u_hat : np.ndarray
        Residuals of the regression model.
    first_stage : FirstStage
        First-stage regression fitted after second-stage inference: the
        coefficients pi_hat, the fitted values X_hat, the residuals v_hat, the
        fitted first-stage model, the excluded instruments, and the
        instrument-strength `diagnostics`.
    _data: pd.DataFrame
        The data frame used in the estimation. None if arguments `lean = True` or
        `store_data = False`.


    Raises
    ------
    ValueError
        If Z is not a two-dimensional array.

    Examples
    --------
    `Feiv` is returned by [feols()](/reference/estimation.api.feols.feols.qmd)
    when the formula includes an IV part, i.e.
    `depvar ~ exog | fe | endog ~ instrument`.

    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X2 | f1 | X1 ~ Z1", pf.get_data())
    fit.tidy()
    ```

    The first stage F-statistic is stored on the fitted object.

    ```{python}
    fit.first_stage.diagnostics.f_stat
    ```

    See the
    [instrumental variables tutorial](/tutorials/instrumental-variables.qmd) for
    details.
    """

    # Set in _fit_first_stage().
    first_stage: FirstStage
    # Set in get_fit().
    collinearity_instruments: CollinearityCheck

    # Constructor and methods implementation...
    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        *,
        options: EstimationOptions,
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
        )

        self.capabilities = replace(
            self.capabilities,
            crv3_inference=False,
            wildboottest=False,
            cluster_causal_variance=False,
            decomposition=False,
        )

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
                y_names=tuple(endogenous_frame.columns),
                x_names=tuple(instrument_frame.columns),
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
        assert self._coefnames_z is not None
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

    def _fit_first_stage(self) -> None:
        """Fit the first-stage regression and publish it as `first_stage`."""
        require_retained(self, "_fit_first_stage", "_data")
        # The excluded instruments are the instrument-matrix columns that are
        # not also second-stage regressors, kept in instrument-matrix order.
        exogenous = set(self._coefnames)
        instruments = tuple(
            str(name) for name in self._coefnames_z if name not in exogenous
        )

        fixest_module = import_module("pyfixest.estimation")
        fit_ = fixest_module.feols

        fml_first_stage = self.model.fixest_formula.first_stage
        # Append fixed effects manually since fml_first_stage doesn't include them
        # (see Formula.fml_first_stage docstring for explanation)
        if self.model.has_fixef and fml_first_stage is not None:
            fml_first_stage += f" | {self.model.fixef}"

        # Type hint to reflect that vcov_detail can be either a dict or a str
        vcov_detail: dict[str, str] | str

        spec = self.variance_covariance.spec
        if spec.is_clustered:
            vcov_detail = {spec.vcov_type_detail: spec.clustervar[0]}
        else:
            vcov_detail = spec.vcov_type_detail

        demeaner = self.options.demeaner
        cached_pre = self._demean_cache.lookup_preconditioner.get(
            self.sample_info.dropped_row_index
        )
        if isinstance(demeaner, LsmrDemeaner) and cached_pre is not None:
            demeaner = replace(demeaner, preconditioner=cached_pre)

        # Do first stage regression
        model1 = fit_(
            fml=fml_first_stage,
            data=self._data,
            vcov=vcov_detail,
            weights=self.options.weights,
            weights_type=self.options.weights_type,
            collin_tol=self.options.collin_tol,
            solver=self.options.solver,
            demeaner=demeaner,
        )

        # Ensure model1 is of type Feols
        if not isinstance(model1, Feols):
            raise TypeError("The first stage model must be of type Feols")

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
        """Apply the parent's retention policy to the retained first stage."""
        first_stage = getattr(self, "first_stage", None)
        if first_stage is not None:
            model = first_stage.model
            # The first stage is fitted in full because `first_stage` is built
            # from its within data and residuals; it takes over the parent's
            # storage options once those values have been read.
            model.options = replace(
                model.options,
                store_data=self.options.store_data,
                lean=self.options.lean,
            )
            model._clear_attributes()
        super()._clear_attributes()

    def IV_Diag(self, statistics: list[str] | None = None):
        """Implement IV diagnostic tests.

        Notes
        -----
        This method covers diagnostic tests related with IV regression.
        We currently have IV weak tests only. More test will be updated
        in future updates!

        Parameters
        ----------
        statistics : list[str], optional
            List of IV diagnostic statistics

        Example
        -------
        The following is an example usage of this method:

            ```{python}

            import numpy as np
            import pandas as pd
            from pyfixest.estimation import feols

            # Set random seed for reproducibility
            np.random.seed(1)

            # Number of observations
            n = 1000

            # Simulate the data
            # Instrumental variable
            z = np.random.binomial(1, 0.5, size=n)
            z2 = np.random.binomial(1, 0.5, size=n)

            # Endogenous variable
            d = 0.5 * z + 1.5 * z2 + np.random.normal(size=n)

            # Control variables
            c1 = np.random.normal(size=n)
            c2 = np.random.normal(size=n)

            # Outcome variable
            y = 1.0 + 1.5 * d + 0.8 * c1 + 0.5 * c2 + np.random.normal(size=n)

            # Cluster variable
            cluster = np.random.randint(1, 50, size=n)
            weights = np.random.uniform(1, 3, size=n)

            # Create a DataFrame
            data = pd.DataFrame({
                'd': d,
                'y': y,
                'z': z,
                'z2': z2,
                'c1': c1,
                'c2': c2,
                'cluster': cluster,
                'weights': weights
            })

            vcov_detail = "iid"

            # Fit OLS model
            fit_ols = feols("y ~ 1 + d + c1 + c2", data=data, vcov=vcov_detail)

            # Fit IV model
            fit_iv = feols("y ~ 1 + c1 + c2 | d ~ z", data=data,
                     vcov=vcov_detail,
                     weights="weights")
            F_stat_pf = fit_iv.first_stage.diagnostics.f_stat
            fit_iv.IV_Diag()
            F_stat_eff_pf = fit_iv.first_stage.diagnostics.eff_f

            print("(Unadjusted) F stat :", F_stat_pf)
            print("Effective F stat :", F_stat_eff_pf)

            ```
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
            List of IV weakness statistics

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
        """Compute Effective F stat (Olea and Pflueger 2013)."""
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
