"""Matplotlib visualizations for omitted-variable-bias sensitivity analysis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from pyfixest.estimation.post_estimation.sensitivity import SensitivityAnalysis


SensitivityTarget = Literal["estimate", "t-value"]


def ovb_contour_plot(
    sensitivity: SensitivityAnalysis,
    *,
    sensitivity_of: SensitivityTarget = "estimate",
    benchmark_covariates: str | Sequence[str] | None = None,
    kd: float | Sequence[float] = 1.0,
    ky: float | Sequence[float] | None = None,
    r2dz_x: float | Sequence[float] | None = None,
    r2yz_dx: float | Sequence[float] | None = None,
    manual_labels: str | Sequence[str] | None = None,
    reduce: bool = True,
    h0: float = 0.0,
    estimate_threshold: float = 0.0,
    t_threshold: float = 2.0,
    lim: float | None = None,
    lim_y: float | None = None,
    col_contour: str = "black",
    col_threshold: str = "#D62728",
    col_benchmark: str = "#D62728",
    label_text: bool = True,
    xlab: str | None = None,
    ylab: str | None = None,
    n_levels: int = 10,
    grid_size: int = 200,
    figsize: tuple[float, float] = (6, 6),
    ax: Axes | None = None,
) -> Figure:
    """Plot adjusted estimates or t values over partial-R-squared scenarios.

    Parameters
    ----------
    sensitivity : SensitivityAnalysis
        Configured analysis object.
    sensitivity_of : {"estimate", "t-value"}, optional
        Quantity represented by contour levels.
    benchmark_covariates : str or sequence of str, optional
        Fitted coefficients used to add benchmark markers.
    kd, ky : float or sequence of float, optional
        Benchmark multipliers passed to ``ovb_bounds()``.
    r2dz_x, r2yz_dx : float or sequence of float, optional
        Manual treatment- and outcome-side partial-R-squared marker positions.
        Both must be supplied and have equal length.
    manual_labels : str or sequence of str, optional
        Labels for manual markers.
    reduce : bool, optional
        Whether confounding moves the estimate toward zero.
    h0 : float, optional
        Null value used for adjusted t statistics.
    estimate_threshold, t_threshold : float, optional
        Highlighted contour level for each sensitivity target.
    lim, lim_y : float, optional
        Horizontal and vertical maxima, each strictly between zero and one.
    col_contour, col_threshold, col_benchmark : str, optional
        Matplotlib colors for regular contours, the highlighted threshold, and
        benchmark/manual markers.
    label_text : bool, optional
        Whether to annotate benchmark and manual markers.
    xlab, ylab : str, optional
        Axis labels.
    n_levels : int, optional
        Number of regular contour levels.
    grid_size : int, optional
        Number of grid points per dimension.
    figsize : tuple of float, optional
        Figure size used when ``ax`` is not supplied.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the contour plot.
    """
    if sensitivity_of not in ("estimate", "t-value"):
        raise ValueError("sensitivity_of must be 'estimate' or 't-value'.")
    _validate_plot_shape(n_levels=n_levels, grid_size=grid_size)

    bounds = _benchmark_bounds(
        sensitivity,
        benchmark_covariates=benchmark_covariates,
        kd=kd,
        ky=ky,
        reduce=reduce,
        h0=h0,
    )
    manual_x, manual_y, labels = _manual_bounds(
        r2dz_x=r2dz_x,
        r2yz_dx=r2yz_dx,
        manual_labels=manual_labels,
    )

    x_points = _combined_points(bounds, "r2dz_x", manual_x)
    y_points = _combined_points(bounds, "r2yz_dx", manual_y)
    lim = _resolve_limit(lim, x_points, default=0.4, name="lim")
    lim_y = _resolve_limit(lim_y, y_points, default=0.4, name="lim_y")

    grid_x = np.linspace(0, lim, grid_size)
    grid_y = np.linspace(0, lim_y, grid_size)
    r2_treatment, r2_outcome = np.meshgrid(grid_x, grid_y)
    if sensitivity_of == "estimate":
        values = np.asarray(
            sensitivity.adjusted_estimate(r2_treatment, r2_outcome, reduce=reduce)
        )
        threshold = float(estimate_threshold)
        benchmark_value_column = "adjusted_estimate"
        unadjusted_value = sensitivity.sensitivity_stats().estimate
    else:
        values = np.asarray(
            sensitivity.adjusted_t(
                r2_treatment,
                r2_outcome,
                reduce=reduce,
                h0=h0,
            )
        )
        threshold = float(t_threshold)
        benchmark_value_column = "adjusted_t"
        unadjusted_value = (
            sensitivity.sensitivity_stats().estimate - h0
        ) / sensitivity.sensitivity_stats().standard_error

    figure, axes = _figure_and_axes(ax=ax, figsize=figsize)
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    levels = np.linspace(minimum, maximum, n_levels)
    regular_levels = levels[~np.isclose(levels, threshold)]
    if regular_levels.size:
        contours = axes.contour(
            grid_x,
            grid_y,
            values,
            levels=regular_levels,
            colors=col_contour,
            linewidths=1,
        )
        axes.clabel(contours, inline=True, fontsize=8, fmt="%1.3g")

    if minimum <= threshold <= maximum:
        threshold_contour = axes.contour(
            grid_x,
            grid_y,
            values,
            levels=[threshold],
            colors=col_threshold,
            linewidths=1.5,
            linestyles="dashed",
        )
        axes.clabel(threshold_contour, inline=True, fontsize=8, fmt="%1.3g")

    axes.scatter(
        [0],
        [0],
        color="black",
        marker="^",
        label="Unadjusted estimate",
        zorder=4,
    )
    axes.annotate(
        f"Unadjusted\n({unadjusted_value:.3g})",
        (0, 0),
        xytext=(5, 5),
        textcoords="offset points",
    )

    if bounds is not None:
        axes.scatter(
            bounds["r2dz_x"],
            bounds["r2yz_dx"],
            color=col_benchmark,
            edgecolor="black",
            marker="D",
            label="Benchmark bounds",
            zorder=5,
        )
        if label_text:
            for row in bounds.itertuples(index=False):
                value = getattr(row, benchmark_value_column)
                axes.annotate(
                    f"{row.bound_label}\n({value:.3g})",
                    (row.r2dz_x, row.r2yz_dx),
                    xytext=(5, 5),
                    textcoords="offset points",
                )

    if manual_x is not None and manual_y is not None:
        axes.scatter(
            manual_x,
            manual_y,
            color=col_benchmark,
            edgecolor="black",
            marker="o",
            label="Manual bounds",
            zorder=5,
        )
        if label_text:
            for x_value, y_value, label in zip(manual_x, manual_y, labels, strict=True):
                axes.annotate(
                    label,
                    (x_value, y_value),
                    xytext=(5, 5),
                    textcoords="offset points",
                )

    axes.set_xlabel(xlab or r"Partial $R^2$ of confounder(s) with the treatment")
    axes.set_ylabel(ylab or r"Partial $R^2$ of confounder(s) with the outcome")
    axes.set_xlim(0, lim)
    axes.set_ylim(0, lim_y)
    axes.legend(frameon=False)
    figure.tight_layout()
    return figure


def ovb_extreme_plot(
    sensitivity: SensitivityAnalysis,
    *,
    benchmark_covariates: str | Sequence[str] | None = None,
    kd: float | Sequence[float] = 1.0,
    ky: float | Sequence[float] | None = None,
    r2yz_dx: Sequence[float] = (1.0, 0.75, 0.5),
    reduce: bool = True,
    threshold: float = 0.0,
    lim: float | None = None,
    ylim: tuple[float, float] | None = None,
    col_scenario: str = "black",
    col_threshold: str = "#D62728",
    col_benchmark: str = "#D62728",
    xlab: str | None = None,
    ylab: str | None = None,
    grid_size: int = 300,
    figsize: tuple[float, float] = (8, 4.8),
    ax: Axes | None = None,
) -> Figure:
    """Plot adjusted estimates under extreme outcome-confounding scenarios."""
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least 2.")
    outcome_strengths = _r2_vector(r2yz_dx, "r2yz_dx", upper_closed=True)
    bounds = _benchmark_bounds(
        sensitivity,
        benchmark_covariates=benchmark_covariates,
        kd=kd,
        ky=ky,
        reduce=reduce,
        h0=0,
        adjusted_estimates=False,
    )
    benchmark_points = (
        None if bounds is None else bounds["r2dz_x"].to_numpy(dtype=float)
    )
    lim = _resolve_limit(lim, benchmark_points, default=0.1, name="lim")
    treatment_strengths = np.linspace(0, lim, grid_size)
    figure, axes = _figure_and_axes(ax=ax, figsize=figsize)

    all_values: list[np.ndarray] = []
    for index, outcome_strength in enumerate(outcome_strengths):
        values = np.asarray(
            sensitivity.adjusted_estimate(
                treatment_strengths,
                outcome_strength,
                reduce=reduce,
            )
        )
        all_values.append(values)
        axes.plot(
            treatment_strengths,
            values,
            color=col_scenario,
            linewidth=max(1.0, 1.8 - 0.25 * index),
            linestyle="solid" if index == 0 else "dashed",
            label=f"Outcome partial R2: {outcome_strength:.0%}",
        )

    axes.axhline(
        threshold,
        color=col_threshold,
        linestyle="dashed",
        label="Estimate threshold",
    )
    value_array = np.concatenate(all_values)
    y_min = float(np.nanmin(value_array))
    y_max = float(np.nanmax(value_array))
    span = max(y_max - y_min, 1e-12)
    if benchmark_points is not None:
        axes.vlines(
            benchmark_points,
            y_min,
            y_min + 0.04 * span,
            color=col_benchmark,
            linewidth=2.5,
            label="Benchmark treatment bounds",
        )

    axes.set_xlabel(xlab or r"Partial $R^2$ of confounder(s) with the treatment")
    axes.set_ylabel(ylab or "Adjusted effect estimate")
    axes.set_xlim(0, lim)
    if ylim is None:
        axes.set_ylim(y_min - 0.05 * span, y_max + 0.05 * span)
    else:
        if len(ylim) != 2 or not np.all(np.isfinite(ylim)) or ylim[0] >= ylim[1]:
            raise ValueError("ylim must contain two finite, increasing values.")
        axes.set_ylim(*ylim)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.legend(frameon=False)
    figure.tight_layout()
    return figure


def _benchmark_bounds(
    sensitivity: SensitivityAnalysis,
    *,
    benchmark_covariates: str | Sequence[str] | None,
    kd: float | Sequence[float],
    ky: float | Sequence[float] | None,
    reduce: bool,
    h0: float,
    adjusted_estimates: bool = True,
) -> pd.DataFrame | None:
    if benchmark_covariates is None:
        return None
    return sensitivity.ovb_bounds(
        benchmark_covariates,
        kd=kd,
        ky=ky,
        reduce=reduce,
        h0=h0,
        adjusted_estimates=adjusted_estimates,
    )


def _manual_bounds(
    *,
    r2dz_x: float | Sequence[float] | None,
    r2yz_dx: float | Sequence[float] | None,
    manual_labels: str | Sequence[str] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    if r2dz_x is None and r2yz_dx is None:
        if manual_labels is not None:
            raise ValueError("manual_labels requires r2dz_x and r2yz_dx.")
        return None, None, []
    if r2dz_x is None or r2yz_dx is None:
        raise ValueError("r2dz_x and r2yz_dx must be supplied together.")
    treatment = _r2_vector(r2dz_x, "r2dz_x", upper_closed=False)
    outcome = _r2_vector(r2yz_dx, "r2yz_dx", upper_closed=True)
    if treatment.size != outcome.size:
        raise ValueError("r2dz_x and r2yz_dx must have the same length.")

    if manual_labels is None:
        labels = [f"Manual bound {index + 1}" for index in range(treatment.size)]
    elif isinstance(manual_labels, str):
        labels = [manual_labels]
    else:
        labels = list(manual_labels)
    if len(labels) != treatment.size or not all(
        isinstance(label, str) for label in labels
    ):
        raise ValueError("manual_labels must contain one string per manual bound.")
    return treatment, outcome, labels


def _r2_vector(
    values: float | Sequence[float], name: str, *, upper_closed: bool
) -> np.ndarray:
    try:
        array = np.atleast_1d(np.asarray(values, dtype=float))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty one-dimensional sequence.")
    invalid_upper = array > 1 if upper_closed else array >= 1
    closing = "]" if upper_closed else ")"
    if np.any((array < 0) | invalid_upper):
        raise ValueError(f"{name} must be in the interval [0, 1{closing}.")
    return array


def _combined_points(
    bounds: pd.DataFrame | None,
    column: str,
    manual: np.ndarray | None,
) -> np.ndarray | None:
    parts: list[np.ndarray] = []
    if bounds is not None:
        parts.append(bounds[column].to_numpy(dtype=float))
    if manual is not None:
        parts.append(manual)
    return np.concatenate(parts) if parts else None


def _resolve_limit(
    value: float | None,
    points: np.ndarray | None,
    *,
    default: float,
    name: str,
) -> float:
    if value is None:
        value = default if points is None else max(default, float(np.max(points)) * 1.2)
        value = min(value, 1 - 1e-9)
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric.") from exc
    if not np.isfinite(value) or value <= 0 or value >= 1:
        raise ValueError(f"{name} must be strictly between 0 and 1.")
    return value


def _validate_plot_shape(*, n_levels: int, grid_size: int) -> None:
    if not isinstance(n_levels, int) or n_levels < 2:
        raise ValueError("n_levels must be an integer of at least 2.")
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least 2.")


def _figure_and_axes(
    *, ax: Axes | None, figsize: tuple[float, float]
) -> tuple[Figure, Axes]:
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.get_figure(), ax


__all__ = ["ovb_contour_plot", "ovb_extreme_plot"]
