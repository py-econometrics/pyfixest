"""Omitted-variable-bias sensitivity analysis for fitted OLS models."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import t

from pyfixest.utils.utils import get_ssc

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from pyfixest.estimation.models.feols_ import Feols


ScalarOrArray = float | np.ndarray
NumericInput = float | Sequence[float] | np.ndarray | pd.Series


@dataclass(frozen=True, slots=True)
class SensitivityStatistics:
    """Sensitivity statistics for one treatment coefficient."""

    estimate: float
    standard_error: float
    degrees_of_freedom: int
    partial_r2: float
    partial_f2: float
    robustness_value: float
    robustness_value_alpha: float

    @property
    def t_statistic(self) -> float:
        """Return the IID t statistic for the treatment coefficient."""
        return self.estimate / self.standard_error

    def to_dict(self) -> dict[str, float | int]:
        """Return the statistics as a plain dictionary."""
        return {
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "degrees_of_freedom": self.degrees_of_freedom,
            "partial_r2": self.partial_r2,
            "partial_f2": self.partial_f2,
            "robustness_value": self.robustness_value,
            "robustness_value_alpha": self.robustness_value_alpha,
        }


@dataclass(frozen=True, slots=True)
class _IidInference:
    estimates: np.ndarray
    standard_errors: np.ndarray
    t_statistics: np.ndarray
    degrees_of_freedom: int


def compute_partial_r2(
    t_statistic: float | np.ndarray, degrees_of_freedom: int
) -> ScalarOrArray:
    """Compute coefficient partial R-squared from an IID t statistic."""
    dof = _validate_degrees_of_freedom(degrees_of_freedom)
    t_value = np.asarray(t_statistic, dtype=float)
    result = t_value**2 / (t_value**2 + dof)
    return _scalar_or_array(result)


def compute_partial_f2(
    t_statistic: float | np.ndarray, degrees_of_freedom: int
) -> ScalarOrArray:
    """Compute coefficient partial Cohen's f-squared from an IID t statistic."""
    dof = _validate_degrees_of_freedom(degrees_of_freedom)
    t_value = np.asarray(t_statistic, dtype=float)
    result = t_value**2 / dof
    return _scalar_or_array(result)


def compute_robustness_value(
    t_statistic: float | np.ndarray,
    degrees_of_freedom: int,
    q: float = 1.0,
    alpha: float = 1.0,
) -> ScalarOrArray:
    """Compute the Cinelli-Hazlett robustness value.

    This follows the reference implementation in R's ``sensemakr`` package,
    including its non-binding and low-degrees-of-freedom branches.
    """
    dof = _validate_degrees_of_freedom(degrees_of_freedom)
    q = _validate_nonnegative_scalar(q, "q")
    alpha = _validate_alpha(alpha)
    t_value = np.asarray(t_statistic, dtype=float)

    fq = q * np.abs(t_value) / np.sqrt(dof)
    f_critical = abs(t.ppf(alpha / 2, dof - 1)) / np.sqrt(dof - 1)
    fqa = fq - f_critical

    result = np.zeros_like(fq, dtype=float)
    binding = fqa > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        result[binding] = 2 / (1 + np.sqrt(1 + 4 / np.square(fqa[binding])))

    if f_critical > 0:
        non_binding = binding & (fq > 1 / f_critical)
        result[non_binding] = (np.square(fq[non_binding]) - f_critical**2) / (
            1 + np.square(fq[non_binding])
        )

    result = np.clip(result, 0, 1)
    return _scalar_or_array(result)


def compute_bias(
    r2dz_x: NumericInput,
    r2yz_dx: NumericInput,
    standard_error: float,
    degrees_of_freedom: int,
) -> ScalarOrArray:
    """Compute omitted-variable bias under the partial-R-squared parameterization."""
    r2_treatment, r2_outcome = _validate_partial_r2_inputs(r2dz_x, r2yz_dx)
    standard_error = _validate_positive_scalar(standard_error, "standard_error")
    dof = _validate_degrees_of_freedom(degrees_of_freedom)
    bias_factor = np.sqrt((r2_outcome * r2_treatment) / (1 - r2_treatment))
    result = bias_factor * standard_error * np.sqrt(dof)
    return _scalar_or_array(result)


def compute_adjusted_estimate(
    r2dz_x: NumericInput,
    r2yz_dx: NumericInput,
    estimate: float,
    standard_error: float,
    degrees_of_freedom: int,
    *,
    reduce: bool = True,
) -> ScalarOrArray:
    """Compute a bias-adjusted coefficient estimate."""
    if not isinstance(reduce, bool):
        raise TypeError("reduce must be a boolean.")
    estimate = _validate_finite_scalar(estimate, "estimate")
    bias = np.asarray(
        compute_bias(
            r2dz_x=r2dz_x,
            r2yz_dx=r2yz_dx,
            standard_error=standard_error,
            degrees_of_freedom=degrees_of_freedom,
        )
    )
    direction = -1 if reduce else 1
    result = np.sign(estimate) * (abs(estimate) + direction * bias)
    return _scalar_or_array(result)


def compute_adjusted_se(
    r2dz_x: NumericInput,
    r2yz_dx: NumericInput,
    standard_error: float,
    degrees_of_freedom: int,
) -> ScalarOrArray:
    """Compute a bias-adjusted standard error."""
    r2_treatment, r2_outcome = _validate_partial_r2_inputs(r2dz_x, r2yz_dx)
    standard_error = _validate_positive_scalar(standard_error, "standard_error")
    dof = _validate_degrees_of_freedom(degrees_of_freedom)
    result = (
        np.sqrt((1 - r2_outcome) / (1 - r2_treatment))
        * standard_error
        * np.sqrt(dof / (dof - 1))
    )
    return _scalar_or_array(result)


def compute_adjusted_t(
    r2dz_x: NumericInput,
    r2yz_dx: NumericInput,
    estimate: float,
    standard_error: float,
    degrees_of_freedom: int,
    *,
    reduce: bool = True,
    h0: float = 0.0,
) -> ScalarOrArray:
    """Compute a bias-adjusted t statistic."""
    h0 = _validate_finite_scalar(h0, "h0")
    adjusted_estimate = np.asarray(
        compute_adjusted_estimate(
            r2dz_x=r2dz_x,
            r2yz_dx=r2yz_dx,
            estimate=estimate,
            standard_error=standard_error,
            degrees_of_freedom=degrees_of_freedom,
            reduce=reduce,
        )
    )
    adjusted_se = np.asarray(
        compute_adjusted_se(
            r2dz_x=r2dz_x,
            r2yz_dx=r2yz_dx,
            standard_error=standard_error,
            degrees_of_freedom=degrees_of_freedom,
        )
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (adjusted_estimate - h0) / adjusted_se
    return _scalar_or_array(result)


@dataclass(frozen=True, slots=True)
class SensitivityAnalysis:
    """Sensitivity analysis for one treatment coefficient in an OLS model.

    Use :meth:`pyfixest.estimation.models.feols_.Feols.sensitivity_analysis`
    instead of constructing this class directly.

    Parameters
    ----------
    model : Feols
        A fitted, unweighted OLS model.
    treatment : str
        Name of the treatment coefficient to analyze.
    """

    model: Feols
    treatment: str

    def __post_init__(self) -> None:
        _validate_sensitivity_model(self.model, self.treatment)
        if self.model._vcov_type != "iid":
            warnings.warn(
                "Sensitivity analysis uses IID standard errors and residual degrees "
                "of freedom, matching sensemakr; the fitted model's covariance "
                "estimator is ignored.",
                UserWarning,
                stacklevel=3,
            )

    def partial_r2(self, covariate: str | None = None) -> float:
        """Return a coefficient's partial R-squared using IID inference.

        Parameters
        ----------
        covariate : str, optional
            Coefficient to evaluate. Defaults to the configured treatment.
        """
        inference = self._iid_inference()
        index = self._coefficient_index(covariate)
        return float(
            compute_partial_r2(
                inference.t_statistics[index], inference.degrees_of_freedom
            )
        )

    def partial_f2(self, covariate: str | None = None) -> float:
        """Return a coefficient's partial Cohen's f-squared using IID inference."""
        inference = self._iid_inference()
        index = self._coefficient_index(covariate)
        return float(
            compute_partial_f2(
                inference.t_statistics[index], inference.degrees_of_freedom
            )
        )

    def robustness_value(
        self,
        covariate: str | None = None,
        *,
        q: float = 1.0,
        alpha: float = 1.0,
    ) -> float:
        """Return the robustness value for a coefficient."""
        inference = self._iid_inference()
        index = self._coefficient_index(covariate)
        return float(
            compute_robustness_value(
                inference.t_statistics[index],
                inference.degrees_of_freedom,
                q=q,
                alpha=alpha,
            )
        )

    def sensitivity_stats(
        self, *, q: float = 1.0, alpha: float = 0.05
    ) -> SensitivityStatistics:
        """Return sensitivity statistics for the configured treatment."""
        q = _validate_nonnegative_scalar(q, "q")
        alpha = _validate_alpha(alpha)
        inference = self._iid_inference()
        index = self._coefficient_index()
        t_statistic = float(inference.t_statistics[index])
        dof = inference.degrees_of_freedom
        return SensitivityStatistics(
            estimate=float(inference.estimates[index]),
            standard_error=float(inference.standard_errors[index]),
            degrees_of_freedom=dof,
            partial_r2=float(compute_partial_r2(t_statistic, dof)),
            partial_f2=float(compute_partial_f2(t_statistic, dof)),
            robustness_value=float(
                compute_robustness_value(t_statistic, dof, q=q, alpha=1)
            ),
            robustness_value_alpha=float(
                compute_robustness_value(t_statistic, dof, q=q, alpha=alpha)
            ),
        )

    def ovb_bounds(
        self,
        benchmark_covariates: str | Sequence[str],
        *,
        kd: float | Sequence[float] = 1.0,
        ky: float | Sequence[float] | None = None,
        alpha: float = 0.05,
        adjusted_estimates: bool = True,
        reduce: bool = True,
        h0: float = 0.0,
    ) -> pd.DataFrame:
        """Compute benchmark bounds on omitted-variable bias.

        Benchmark associations are computed from the fitted model's prepared
        design matrix. This preserves absorbed fixed effects, transformations,
        categorical encodings, the estimation sample, and ``store_data=False``.

        Parameters
        ----------
        benchmark_covariates : str or sequence of str
            Observed coefficient names used as confounding benchmarks.
        kd : float or sequence of float, optional
            Treatment-association multipliers. Defaults to 1.
        ky : float or sequence of float, optional
            Outcome-association multipliers. Defaults to ``kd``.
        alpha : float, optional
            Significance level for adjusted confidence intervals.
        adjusted_estimates : bool, optional
            Whether to add adjusted estimate, standard error, t statistic, and
            confidence interval columns.
        reduce : bool, optional
            Whether confounding moves the estimate toward zero.
        h0 : float, optional
            Null value used for adjusted t statistics.

        Returns
        -------
        pandas.DataFrame
            One row per benchmark and multiplier pair.
        """
        benchmarks = self._validate_benchmarks(benchmark_covariates)
        kd_values = _validate_multipliers(kd, "kd")
        ky_values = kd_values.copy() if ky is None else _validate_multipliers(ky, "ky")
        if kd_values.size != ky_values.size:
            raise ValueError("kd and ky must have the same length.")
        alpha = _validate_alpha(alpha)
        h0 = _validate_finite_scalar(h0, "h0")
        if not isinstance(adjusted_estimates, bool):
            raise TypeError("adjusted_estimates must be a boolean.")
        if not isinstance(reduce, bool):
            raise TypeError("reduce must be a boolean.")

        records: list[dict[str, str | float]] = []
        capped_benchmarks: list[str] = []
        for benchmark in benchmarks:
            r2yxj_dx = self.partial_r2(benchmark)
            r2dxj_x = self._treatment_benchmark_partial_r2(benchmark)
            for kd_value, ky_value in zip(kd_values, ky_values, strict=True):
                r2dz_x = kd_value * r2dxj_x / (1 - r2dxj_x)
                if r2dz_x >= 1:
                    raise ValueError(
                        f"kd={kd_value:g} implies r2dz_x >= 1 for benchmark "
                        f"{benchmark!r}; use a smaller kd."
                    )

                denominator = (1 - kd_value * r2dxj_x) * (1 - r2dxj_x)
                if denominator <= 0:
                    raise ValueError(
                        f"kd={kd_value:g} is not feasible for benchmark "
                        f"{benchmark!r}; use a smaller kd."
                    )
                r2zxj_xd = kd_value * r2dxj_x**2 / denominator
                if r2zxj_xd >= 1:
                    raise ValueError(
                        f"kd={kd_value:g} implies an impossible benchmark "
                        f"association for {benchmark!r}; use a smaller kd."
                    )

                r2yz_dx = (
                    ((np.sqrt(ky_value) + np.sqrt(r2zxj_xd)) / np.sqrt(1 - r2zxj_xd))
                    ** 2
                    * r2yxj_dx
                    / (1 - r2yxj_dx)
                )
                if r2yz_dx > 1:
                    r2yz_dx = 1.0
                    capped_benchmarks.append(benchmark)

                records.append(
                    {
                        "bound_label": f"{kd_value:g}x {benchmark}",
                        "benchmark_covariate": benchmark,
                        "treatment": self.treatment,
                        "kd": float(kd_value),
                        "ky": float(ky_value),
                        "r2dz_x": float(r2dz_x),
                        "r2yz_dx": float(r2yz_dx),
                    }
                )

        if capped_benchmarks:
            benchmark_list = ", ".join(dict.fromkeys(capped_benchmarks))
            warnings.warn(
                "The implied outcome-side partial R-squared exceeded 1 and was "
                f"capped at 1 for: {benchmark_list}.",
                RuntimeWarning,
                stacklevel=2,
            )

        bounds = pd.DataFrame.from_records(records)
        if adjusted_estimates:
            stats = self.sensitivity_stats()
            r2_treatment = bounds["r2dz_x"].to_numpy()
            r2_outcome = bounds["r2yz_dx"].to_numpy()
            bounds["adjusted_estimate"] = compute_adjusted_estimate(
                r2_treatment,
                r2_outcome,
                stats.estimate,
                stats.standard_error,
                stats.degrees_of_freedom,
                reduce=reduce,
            )
            bounds["adjusted_se"] = compute_adjusted_se(
                r2_treatment,
                r2_outcome,
                stats.standard_error,
                stats.degrees_of_freedom,
            )
            bounds["adjusted_t"] = compute_adjusted_t(
                r2_treatment,
                r2_outcome,
                stats.estimate,
                stats.standard_error,
                stats.degrees_of_freedom,
                reduce=reduce,
                h0=h0,
            )
            critical_value = abs(t.ppf(alpha / 2, stats.degrees_of_freedom))
            bounds["adjusted_lower_ci"] = (
                bounds["adjusted_estimate"] - critical_value * bounds["adjusted_se"]
            )
            bounds["adjusted_upper_ci"] = (
                bounds["adjusted_estimate"] + critical_value * bounds["adjusted_se"]
            )
        return bounds

    def bias(self, r2dz_x: NumericInput, r2yz_dx: NumericInput) -> ScalarOrArray:
        """Return omitted-variable bias for the configured treatment."""
        stats = self.sensitivity_stats()
        return compute_bias(
            r2dz_x,
            r2yz_dx,
            stats.standard_error,
            stats.degrees_of_freedom,
        )

    def adjusted_estimate(
        self,
        r2dz_x: NumericInput,
        r2yz_dx: NumericInput,
        *,
        reduce: bool = True,
    ) -> ScalarOrArray:
        """Return a bias-adjusted estimate for the configured treatment."""
        stats = self.sensitivity_stats()
        return compute_adjusted_estimate(
            r2dz_x,
            r2yz_dx,
            stats.estimate,
            stats.standard_error,
            stats.degrees_of_freedom,
            reduce=reduce,
        )

    def adjusted_se(self, r2dz_x: NumericInput, r2yz_dx: NumericInput) -> ScalarOrArray:
        """Return a bias-adjusted standard error for the configured treatment."""
        stats = self.sensitivity_stats()
        return compute_adjusted_se(
            r2dz_x,
            r2yz_dx,
            stats.standard_error,
            stats.degrees_of_freedom,
        )

    def adjusted_t(
        self,
        r2dz_x: NumericInput,
        r2yz_dx: NumericInput,
        *,
        reduce: bool = True,
        h0: float = 0.0,
    ) -> ScalarOrArray:
        """Return a bias-adjusted t statistic for the configured treatment."""
        stats = self.sensitivity_stats()
        return compute_adjusted_t(
            r2dz_x,
            r2yz_dx,
            stats.estimate,
            stats.standard_error,
            stats.degrees_of_freedom,
            reduce=reduce,
            h0=h0,
        )

    def summary(
        self,
        *,
        benchmark_covariates: str | Sequence[str] | None = None,
        kd: float | Sequence[float] = 1.0,
        ky: float | Sequence[float] | None = None,
        q: float = 1.0,
        alpha: float = 0.05,
        reduce: bool = True,
        decimals: int = 3,
    ) -> None:
        """Print a compact sensitivity-analysis summary."""
        if not isinstance(decimals, int) or decimals < 0:
            raise ValueError("decimals must be a nonnegative integer.")
        stats = self.sensitivity_stats(q=q, alpha=alpha)
        null_multiplier = 1 - q if reduce else 1 + q
        h0 = null_multiplier * stats.estimate

        print("Sensitivity Analysis to Unobserved Confounding\n")
        print(f"Model Formula: {self.model._fml}\n")
        print(f"Treatment: {self.treatment}")
        print(f"Null hypothesis: q = {q:g}, reduce = {reduce}")
        print(f"H0: tau = {h0:.{decimals}f}\n")
        print("Unadjusted estimate:")
        print(f"  Coefficient: {stats.estimate:.{decimals}f}")
        print(f"  IID standard error: {stats.standard_error:.{decimals}f}")
        print(f"  IID t-value: {stats.t_statistic:.{decimals}f}\n")
        print("Sensitivity statistics:")
        print(f"  Partial R2: {stats.partial_r2:.{decimals}f}")
        print(f"  Robustness value (q = {q:g}): {stats.robustness_value:.{decimals}f}")
        print(
            f"  Robustness value (q = {q:g}, alpha = {alpha:g}): "
            f"{stats.robustness_value_alpha:.{decimals}f}\n"
        )

        if benchmark_covariates is not None:
            print("Bounds on omitted variable bias:")
            bounds = self.ovb_bounds(
                benchmark_covariates,
                kd=kd,
                ky=ky,
                alpha=alpha,
                reduce=reduce,
                h0=h0,
            )
            print(
                bounds.to_string(
                    index=False,
                    float_format=lambda value: f"{value:.{decimals}f}",
                )
            )

    def plot(
        self,
        *,
        plot_type: str = "contour",
        sensitivity_of: str = "estimate",
        benchmark_covariates: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Figure:
        """Plot contour or extreme omitted-confounding scenarios.

        Parameters
        ----------
        plot_type : {"contour", "extreme"}, optional
            Diagnostic plot type.
        sensitivity_of : {"estimate", "t-value"}, optional
            Contour target. Extreme plots currently support estimates only.
        benchmark_covariates : str or sequence of str, optional
            Fitted coefficients shown as benchmark markers or rugs.
        **kwargs : object
            Additional options passed to the selected plotting function.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the requested diagnostic.
        """
        from pyfixest.report.visualize_sensitivity import (
            ovb_contour_plot,
            ovb_extreme_plot,
        )

        if plot_type == "contour":
            return ovb_contour_plot(
                self,
                sensitivity_of=sensitivity_of,  # type: ignore[arg-type]
                benchmark_covariates=benchmark_covariates,
                **kwargs,
            )
        if plot_type == "extreme":
            if sensitivity_of != "estimate":
                raise NotImplementedError(
                    "Extreme sensitivity plots currently support estimates only."
                )
            return ovb_extreme_plot(
                self,
                benchmark_covariates=benchmark_covariates,
                **kwargs,
            )
        raise ValueError("plot_type must be 'contour' or 'extreme'.")

    def _iid_inference(self) -> _IidInference:
        ssc, _, degrees_of_freedom = get_ssc(
            **self.model._make_ssc_kwargs(vcov_type="iid", G=1)
        )
        degrees_of_freedom = _validate_degrees_of_freedom(degrees_of_freedom)
        vcov = np.asarray(ssc, dtype=float) * self.model._vcov_iid()
        standard_errors = np.sqrt(np.diag(vcov))
        t_statistics = self.model._beta_hat / standard_errors
        return _IidInference(
            estimates=np.asarray(self.model._beta_hat, dtype=float),
            standard_errors=standard_errors,
            t_statistics=t_statistics,
            degrees_of_freedom=degrees_of_freedom,
        )

    def _coefficient_index(self, covariate: str | None = None) -> int:
        coefficient = self.treatment if covariate is None else covariate
        if not isinstance(coefficient, str):
            raise TypeError("covariate must be a coefficient name.")
        try:
            return self.model._coefnames.index(coefficient)
        except ValueError as exc:
            raise ValueError(
                f"Coefficient {coefficient!r} was not found in the fitted model."
            ) from exc

    def _validate_benchmarks(
        self, benchmark_covariates: str | Sequence[str]
    ) -> list[str]:
        benchmarks = (
            [benchmark_covariates]
            if isinstance(benchmark_covariates, str)
            else list(benchmark_covariates)
        )
        if not benchmarks:
            raise ValueError("benchmark_covariates must not be empty.")
        for benchmark in benchmarks:
            if not isinstance(benchmark, str):
                raise TypeError("Every benchmark covariate must be a coefficient name.")
            if benchmark == self.treatment:
                raise ValueError("A benchmark covariate cannot also be the treatment.")
            if benchmark == "Intercept":
                raise ValueError(
                    "The intercept cannot be used as a benchmark covariate."
                )
            self._coefficient_index(benchmark)
        return benchmarks

    def _treatment_benchmark_partial_r2(self, benchmark: str) -> float:
        treatment_index = self._coefficient_index()
        benchmark_index = self._coefficient_index(benchmark)
        design = np.asarray(self.model._X, dtype=float)
        controls = np.delete(design, [treatment_index, benchmark_index], axis=1)
        treatment = design[:, treatment_index]
        benchmark_values = design[:, benchmark_index]

        if controls.shape[1]:
            treatment = (
                treatment
                - controls @ np.linalg.lstsq(controls, treatment, rcond=None)[0]
            )
            benchmark_values = (
                benchmark_values
                - controls @ np.linalg.lstsq(controls, benchmark_values, rcond=None)[0]
            )

        treatment_ss = float(treatment @ treatment)
        benchmark_ss = float(benchmark_values @ benchmark_values)
        if treatment_ss <= 0 or benchmark_ss <= 0:
            raise ValueError(
                f"Benchmark {benchmark!r} has no residual variation after "
                "conditioning on the other fitted covariates."
            )
        partial_r2 = float(
            (treatment @ benchmark_values) ** 2 / (treatment_ss * benchmark_ss)
        )
        return float(np.clip(partial_r2, 0, 1))


def _validate_sensitivity_model(model: Feols, treatment: str) -> None:
    if model._method != "feols" or model._is_iv:
        raise NotImplementedError(
            "Sensitivity analysis is currently supported only for non-IV feols models."
        )
    if model._has_weights:
        raise NotImplementedError(
            "Sensitivity analysis does not currently support weighted feols models."
        )
    if model._lean:
        raise NotImplementedError(
            "Sensitivity analysis requires stored model matrices and is unavailable "
            "when lean=True. Refit with lean=False."
        )
    if not isinstance(treatment, str):
        raise TypeError("treatment must be a coefficient name.")
    if treatment == "Intercept":
        raise ValueError("treatment must name a non-intercept coefficient.")
    if treatment not in model._coefnames:
        raise ValueError(f"Treatment {treatment!r} was not found in the fitted model.")
    _validate_degrees_of_freedom(
        get_ssc(**model._make_ssc_kwargs(vcov_type="iid", G=1))[2]
    )


def _validate_partial_r2_inputs(
    r2dz_x: NumericInput, r2yz_dx: NumericInput
) -> tuple[np.ndarray, np.ndarray]:
    try:
        r2_treatment, r2_outcome = np.broadcast_arrays(
            np.asarray(r2dz_x, dtype=float), np.asarray(r2yz_dx, dtype=float)
        )
    except ValueError as exc:
        raise ValueError("r2dz_x and r2yz_dx must be broadcast-compatible.") from exc
    if not np.all(np.isfinite(r2_treatment)) or not np.all(np.isfinite(r2_outcome)):
        raise ValueError("Partial R-squared values must be finite.")
    if np.any((r2_treatment < 0) | (r2_treatment >= 1)):
        raise ValueError("r2dz_x must be in the interval [0, 1).")
    if np.any((r2_outcome < 0) | (r2_outcome > 1)):
        raise ValueError("r2yz_dx must be in the interval [0, 1].")
    return r2_treatment, r2_outcome


def _validate_multipliers(values: float | Sequence[float], name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc
    if array.ndim > 1:
        raise ValueError(f"{name} must be a scalar or one-dimensional sequence.")
    array = np.atleast_1d(array)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must contain finite, nonnegative values.")
    return array


def _validate_degrees_of_freedom(degrees_of_freedom: int) -> int:
    dof = int(degrees_of_freedom)
    if dof != degrees_of_freedom or dof < 2:
        raise ValueError(
            "Sensitivity analysis requires at least 2 residual degrees of freedom."
        )
    return dof


def _validate_alpha(alpha: float) -> float:
    alpha = _validate_finite_scalar(alpha, "alpha")
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be in the interval [0, 1].")
    return alpha


def _validate_nonnegative_scalar(value: float, name: str) -> float:
    value = _validate_finite_scalar(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _validate_positive_scalar(value: float, name: str) -> float:
    value = _validate_finite_scalar(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_finite_scalar(value: float, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a number.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _scalar_or_array(value: np.ndarray) -> ScalarOrArray:
    value = np.asarray(value, dtype=float)
    return float(value) if value.ndim == 0 else value


__all__ = [
    "SensitivityAnalysis",
    "SensitivityStatistics",
    "compute_adjusted_estimate",
    "compute_adjusted_se",
    "compute_adjusted_t",
    "compute_bias",
    "compute_partial_f2",
    "compute_partial_r2",
    "compute_robustness_value",
]
