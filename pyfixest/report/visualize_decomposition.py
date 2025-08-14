"""
Visualization functions for Gelbach decomposition results.

This module provides standalone visualization functions for creating waterfall charts
from Gelbach decomposition data, separated from the main decomposition logic.
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class PlotConfig:
    """Configuration for coefplot styling and behavior."""

    # Plot dimensions
    figsize: tuple[int, int] = (12, 8)

    # Bar styling
    bar_width: float = 0.6
    bar_alpha: float = 0.8

    # Colors
    color_initial: str = "#1f77b4"  # Blue
    color_final: str = "#1f77b4"  # Blue
    color_mediator_green: str = "#2ca02c"  # Green (reducing effect)
    color_mediator_red: str = "#d62728"  # Red (increasing effect)
    color_spanner: str = "#2E4A87"  # Navy blue
    color_explained_text: str = "#2E4A87"  # Navy blue

    # Text styling
    fontsize: int = 10
    fontweight: str = "bold"

    # Layout constants
    spanner_linewidth: float = 2.0
    axis_padding_factor: float = 1.15
    min_axis_range: float = 0.1
    spacing_factor: float = 0.03
    spanner_position_factor: float = 0.2
    tick_height_factor: float = 0.015
    spanner_offset: float = 0.3
    max_bar_spacing_factor: float = 0.35
    outside_label_offset: float = 0.6


@dataclass
class BarData:
    """Data for a single bar in the waterfall chart."""

    position: int
    height: float
    bottom: float
    value: float
    bar_type: str  # "initial", "mediator_green", "mediator_red", "final"
    color: str
    name: Optional[str] = None  # For mediator bars


class MediatorInfo(NamedTuple):
    """Information about a mediator variable."""

    name: str
    value: float
    moves_toward_zero: bool


def create_decomposition_plot(
    decomposition_data: pd.DataFrame,
    depvarname: str,
    decomp_var: str,
    components_order: Optional[list[str]] = None,
    annotate_shares: bool = True,
    title: Optional[str] = None,
    figsize: Optional[tuple[int, int]] = None,
    keep: Optional[Union[list, str]] = None,
    drop: Optional[Union[list, str]] = None,
    exact_match: bool = False,
    labels: Optional[dict] = None,
    notes: Optional[str] = None,
) -> None:
    """
    Create a waterfall chart showing Gelbach decomposition results.

    This is a standalone function that creates decomposition plots without
    requiring a GelbachDecomposition instance.

    Parameters
    ----------
    decomposition_data : pd.DataFrame
        DataFrame with decomposition results from tidy() method.
    depvarname : str
        Name of the dependent variable.
    decomp_var : str
        Name of the decomposition variable.
    components_order : Optional[list[str]], optional
        Order of mediator components to display.
    annotate_shares : bool, optional
        Whether to show percentage shares in parentheses. Default True.
    title : Optional[str], optional
        Chart title. If None, uses default title.
    figsize : Optional[tuple[int, int]], optional
        Figure size (width, height) in inches. Default (12, 8).
    keep : Optional[Union[list, str]], optional
        Pattern for retaining mediator names.
    drop : Optional[Union[list, str]], optional
        Pattern for excluding mediator names.
    exact_match : bool, optional
        Whether to use exact match for keep/drop. Default False.
    labels : Optional[dict], optional
        Dictionary to relabel mediator variables.
    notes : Optional[str], optional
        Custom notes to display below the chart.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for coefplot. Install with: pip install matplotlib"
        )

    # Create configuration
    config = PlotConfig(figsize=figsize or (12, 8))

    # Prepare data for plotting
    plot_data = _prepare_plot_data(
        decomposition_data, components_order, keep, drop, exact_match, labels
    )

    # Create and configure the plot
    fig, ax = plt.subplots(figsize=config.figsize)

    # Draw the chart components
    _draw_bars(ax, plot_data, config)
    _draw_spanner(ax, plot_data, config, annotate_shares)
    _add_bar_labels(ax, plot_data, config, annotate_shares)

    # Finalize the plot
    _finalize_plot(
        ax, plot_data, config, depvarname, decomp_var, title, annotate_shares, notes
    )

    plt.show()


def _prepare_plot_data(
    decomposition_data: pd.DataFrame,
    components_order: Optional[list[str]],
    keep: Optional[Union[list, str]],
    drop: Optional[Union[list, str]],
    exact_match: bool,
    labels: Optional[dict],
) -> dict:
    """Prepare all data needed for plotting."""
    from pyfixest.utils.dev_utils import _select_order_coefs

    levels = decomposition_data[
        decomposition_data["panels"].str.lower().eq("levels (units)".lower())
    ].copy()

    if levels.empty:
        raise ValueError("No rows found with panels == 'Levels (units)'.")
    if "direct_effect" not in levels.index or "full_effect" not in levels.index:
        raise ValueError(
            "Expected 'direct_effect' and 'full_effect' in 'Levels (units)'."
        )

    # Extract key values
    direct_effect = pd.to_numeric(levels.loc["direct_effect", "coefficients"])
    full_effect = pd.to_numeric(levels.loc["full_effect", "coefficients"])

    # Filter and order mediators
    mediators = _filter_and_order_mediators(
        levels, components_order, keep, drop, exact_match, _select_order_coefs
    )

    # Apply labels
    display_labels = _apply_labels(mediators, labels)

    # Get mediator values and reorder by effect type
    mediator_info = _categorize_mediators(levels, mediators, direct_effect)

    # Create bar data
    bar_data = _create_bar_data(
        direct_effect, full_effect, mediator_info, display_labels
    )

    # Calculate axis limits
    y_min, y_max = _calculate_axis_limits(bar_data)

    return {
        "direct_effect": direct_effect,
        "full_effect": full_effect,
        "explained_effect": sum(info.value for info in mediator_info),
        "bar_data": bar_data,
        "y_min": y_min,
        "y_max": y_max,
        "display_labels": display_labels,
    }


def _filter_and_order_mediators(
    levels: pd.DataFrame,
    components_order: Optional[list[str]],
    keep: Optional[Union[list, str]],
    drop: Optional[Union[list, str]],
    exact_match: bool,
    select_order_coefs_func,
) -> list[str]:
    """Filter and order mediator variables."""
    # Get mediator components (exclude the key summary effects)
    exclude = {
        "direct_effect",
        "full_effect",
        "unexplained_effect",
        "explained_effect",
    }
    mediators = [e for e in levels.index if e not in exclude]

    # Apply keep/drop filtering
    if keep is not None or drop is not None:
        keep_list = keep if isinstance(keep, list) else ([keep] if keep else [])
        drop_list = drop if isinstance(drop, list) else ([drop] if drop else [])
        mediators = select_order_coefs_func(
            mediators, keep_list, drop_list, exact_match
        )

    # Apply user-specified order if provided
    if components_order:
        ordered_mediators = [c for c in components_order if c in mediators]
        ordered_mediators.extend([c for c in mediators if c not in components_order])
    else:
        ordered_mediators = mediators

    return ordered_mediators


def _apply_labels(mediators: list[str], labels: Optional[dict]) -> dict[str, str]:
    """Apply custom labels to mediator names."""
    if labels:
        return {med: labels.get(med, med) for med in mediators}
    else:
        return {med: med for med in mediators}


def _categorize_mediators(
    levels: pd.DataFrame, mediators: list[str], direct_effect: float
) -> list[MediatorInfo]:
    """Categorize and sort mediators by their effect direction."""
    # Get mediator values
    mediator_data = [
        (med, pd.to_numeric(levels.loc[med, "coefficients"])) for med in mediators
    ]

    # Separate red (move away from zero) and green (move toward zero) effects
    red_effects = []
    green_effects = []
    for name, val in mediator_data:
        moves_toward_zero = (np.sign(direct_effect) * np.sign(val)) >= 0
        mediator_info = MediatorInfo(name, val, moves_toward_zero)

        if moves_toward_zero:
            green_effects.append(mediator_info)
        else:
            red_effects.append(mediator_info)

    # Sort by decreasing absolute value
    red_effects.sort(key=lambda x: abs(x.value), reverse=True)
    green_effects.sort(key=lambda x: abs(x.value), reverse=True)

    # Reorder: red first (away from zero), then green (toward zero)
    return red_effects + green_effects


def _create_bar_data(
    direct_effect: float,
    full_effect: float,
    mediator_info: list[MediatorInfo],
    display_labels: dict[str, str],
) -> list[BarData]:
    """Create bar data for waterfall chart."""
    config = PlotConfig()  # Use default config for colors
    bar_data = []

    # Initial bar
    bar_data.append(
        BarData(
            position=0,
            height=abs(direct_effect),
            bottom=min(0, direct_effect),
            value=direct_effect,
            bar_type="initial",
            color=config.color_initial,
            name="Initial Difference",
        )
    )

    # Mediator bars
    cumulative_position = direct_effect
    for i, info in enumerate(mediator_info):
        old_pos = cumulative_position
        cumulative_position -= info.value

        bar_type = "mediator_green" if info.moves_toward_zero else "mediator_red"
        color = (
            config.color_mediator_green
            if info.moves_toward_zero
            else config.color_mediator_red
        )

        bar_data.append(
            BarData(
                position=i + 1,
                height=abs(info.value),
                bottom=min(old_pos, cumulative_position),
                value=info.value,
                bar_type=bar_type,
                color=color,
                name=display_labels.get(info.name, info.name),
            )
        )

    # Final bar
    bar_data.append(
        BarData(
            position=len(mediator_info) + 1,
            height=abs(full_effect),
            bottom=min(0, full_effect),
            value=full_effect,
            bar_type="final",
            color=config.color_final,
            name="Final Difference",
        )
    )

    return bar_data


def _calculate_axis_limits(bar_data: list[BarData]) -> tuple[float, float]:
    """Calculate appropriate axis limits for the plot."""
    config = PlotConfig()

    all_bar_tops = [bar.bottom + bar.height for bar in bar_data]
    all_bar_bottoms = [bar.bottom for bar in bar_data]

    y_min = min(min(all_bar_bottoms), 0) * config.axis_padding_factor
    y_max = max(max(all_bar_tops), 0) * config.axis_padding_factor

    # Ensure minimum range
    if abs(y_max - y_min) < config.min_axis_range:
        if y_max >= 0:
            y_max += config.min_axis_range
        if y_min <= 0:
            y_min -= config.min_axis_range

    return y_min, y_max


def _draw_bars(ax, plot_data: dict, config: PlotConfig) -> None:
    """Draw the waterfall bars."""
    bar_data = plot_data["bar_data"]

    positions = [bar.position for bar in bar_data]
    heights = [bar.height for bar in bar_data]
    bottoms = [bar.bottom for bar in bar_data]
    colors = [bar.color for bar in bar_data]

    ax.bar(
        positions,
        heights,
        bottom=bottoms,
        width=config.bar_width,
        color=colors,
        alpha=config.bar_alpha,
    )

    # Set axis limits
    ax.set_ylim(plot_data["y_min"], plot_data["y_max"])


def _draw_spanner(
    ax, plot_data: dict, config: PlotConfig, annotate_shares: bool
) -> None:
    """Draw the spanner showing explained effect."""
    bar_data = plot_data["bar_data"]
    mediator_bars = [bar for bar in bar_data if bar.bar_type.startswith("mediator")]

    if not mediator_bars:
        return

    direct_effect = plot_data["direct_effect"]
    explained_effect = plot_data["explained_effect"]
    y_min, y_max = plot_data["y_min"], plot_data["y_max"]

    # Calculate spanner position
    mediator_start = mediator_bars[0].position
    mediator_end = mediator_bars[-1].position

    # Position spanner based on chart orientation
    if direct_effect >= 0:
        highest_bar_top = max(bar.bottom + bar.height for bar in bar_data)
        spanner_y = (
            highest_bar_top + (y_max - highest_bar_top) * config.spanner_position_factor
        )
    else:
        lowest_bar_bottom = min(bar.bottom for bar in bar_data)
        spanner_y = (
            lowest_bar_bottom
            + (y_min - lowest_bar_bottom) * config.spanner_position_factor
        )

    # Draw horizontal line
    ax.plot(
        [mediator_start - config.spanner_offset, mediator_end + config.spanner_offset],
        [spanner_y, spanner_y],
        color=config.color_spanner,
        linewidth=config.spanner_linewidth,
    )

    # Draw vertical ticks
    tick_height = (y_max - y_min) * config.tick_height_factor
    for x_pos in [
        mediator_start - config.spanner_offset,
        mediator_end + config.spanner_offset,
    ]:
        ax.plot(
            [x_pos, x_pos],
            [spanner_y - tick_height, spanner_y + tick_height],
            color=config.color_spanner,
            linewidth=config.spanner_linewidth,
        )

    # Add label
    spanner_center = (mediator_start + mediator_end) / 2
    if annotate_shares:
        share_of_direct = (explained_effect / direct_effect) * 100
        spanner_label = (
            f"Explained Effect: {explained_effect:.3f} ({share_of_direct:.1f}%)"
        )
    else:
        spanner_label = f"Explained Effect: {explained_effect:.3f}"

    ax.text(
        spanner_center,
        spanner_y + tick_height * 2.5
        if direct_effect >= 0
        else spanner_y - tick_height * 2.5,
        spanner_label,
        ha="center",
        va="bottom" if direct_effect >= 0 else "top",
        color=config.color_explained_text,
        fontweight=config.fontweight,
        fontsize=config.fontsize,
    )


def _add_bar_labels(
    ax, plot_data: dict, config: PlotConfig, annotate_shares: bool
) -> None:
    """Add value labels to bars."""
    bar_data = plot_data["bar_data"]
    direct_effect = plot_data["direct_effect"]
    explained_effect = plot_data["explained_effect"]
    y_min, y_max = plot_data["y_min"], plot_data["y_max"]

    spacing_unit = (y_max - y_min) * config.spacing_factor

    for bar in bar_data:
        label_y = bar.bottom + bar.height / 2

        if bar.bar_type in ["initial", "final"]:
            _add_simple_label(ax, bar, label_y, config, annotate_shares, direct_effect)
        else:  # mediator bars
            _add_mediator_label(
                ax,
                bar,
                label_y,
                config,
                annotate_shares,
                direct_effect,
                explained_effect,
                spacing_unit,
            )


def _add_simple_label(
    ax,
    bar: BarData,
    label_y: float,
    config: PlotConfig,
    annotate_shares: bool,
    direct_effect: float,
) -> None:
    """Add label for initial or final bars."""
    if bar.bar_type == "final" and annotate_shares:
        share_of_direct = (bar.value / direct_effect) * 100
        label = f"{bar.value:.3f}\n({share_of_direct:.1f}%)"
    else:
        label = f"{bar.value:.3f}"

    ax.text(
        bar.position,
        label_y,
        label,
        ha="center",
        va="center",
        color="black",
        fontweight=config.fontweight,
        fontsize=config.fontsize,
    )


def _add_mediator_label(
    ax,
    bar: BarData,
    label_y: float,
    config: PlotConfig,
    annotate_shares: bool,
    direct_effect: float,
    explained_effect: float,
    spacing_unit: float,
) -> None:
    """Add label for mediator bars with smart placement."""
    if not annotate_shares:
        ax.text(
            bar.position,
            label_y,
            f"{bar.value:.3f}",
            ha="center",
            va="center",
            color="black",
            fontweight=config.fontweight,
            fontsize=config.fontsize,
        )
        return

    # Create multi-line labels
    lines = [f"{bar.value:.3f}"]

    if abs(direct_effect) > 1e-10:
        total_share = (bar.value / direct_effect) * 100.0
        lines.append(f"({total_share:.1f}%)")

    if abs(explained_effect) > 1e-10:
        explained_share = (bar.value / explained_effect) * 100.0
        lines.append(f"({explained_share:.1f}%)")

    # Decide placement
    min_needed_height = spacing_unit * (2 if len(lines) == 2 else 3)
    fits_inside = bar.height >= min_needed_height

    if fits_inside:
        _place_labels_inside(ax, bar, lines, label_y, spacing_unit, config)
    else:
        _place_labels_outside(ax, bar, lines, spacing_unit, config)


def _place_labels_inside(
    ax,
    bar: BarData,
    lines: list[str],
    label_y: float,
    spacing_unit: float,
    config: PlotConfig,
) -> None:
    """Place labels inside the bar."""
    actual_spacing = min(spacing_unit, bar.height * config.max_bar_spacing_factor)

    if len(lines) == 1:
        ax.text(
            bar.position,
            label_y,
            lines[0],
            ha="center",
            va="center",
            color="black",
            fontweight=config.fontweight,
            fontsize=config.fontsize,
        )
    elif len(lines) == 2:
        ax.text(
            bar.position,
            label_y + actual_spacing / 2,
            lines[0],
            ha="center",
            va="center",
            color="black",
            fontweight=config.fontweight,
            fontsize=config.fontsize,
        )
        ax.text(
            bar.position,
            label_y - actual_spacing / 2,
            lines[1],
            ha="center",
            va="center",
            color="black",
            fontweight=config.fontweight,
            fontsize=config.fontsize,
        )
    else:
        ax.text(
            bar.position,
            label_y + actual_spacing,
            lines[0],
            ha="center",
            va="center",
            color="black",
            fontweight=config.fontweight,
            fontsize=config.fontsize,
        )
        ax.text(
            bar.position,
            label_y,
            lines[1],
            ha="center",
            va="center",
            color="black",
            fontweight=config.fontweight,
            fontsize=config.fontsize,
        )
        ax.text(
            bar.position,
            label_y - actual_spacing,
            lines[2],
            ha="center",
            va="center",
            color=config.color_explained_text,
            fontweight=config.fontweight,
            fontsize=config.fontsize,
        )


def _place_labels_outside(
    ax, bar: BarData, lines: list[str], spacing_unit: float, config: PlotConfig
) -> None:
    """Place labels outside the bar to avoid overlap."""
    # Determine direction and starting position
    if bar.value >= 0:
        direction = -1  # Place below positive bars
        start_y = bar.bottom - (spacing_unit * config.outside_label_offset)
    else:
        direction = 1  # Place above negative bars
        start_y = (bar.bottom + bar.height) + (
            spacing_unit * config.outside_label_offset
        )

    # Place labels in order
    for j, text in enumerate(lines):
        y = start_y + direction * (j * spacing_unit)
        color = "black" if j == 0 or j == 1 else config.color_explained_text

        ax.text(
            bar.position,
            y,
            text,
            ha="center",
            va="bottom" if direction > 0 else "top",
            color=color,
            fontweight=config.fontweight,
            fontsize=config.fontsize,
        )


def _finalize_plot(
    ax,
    plot_data: dict,
    config: PlotConfig,
    depvarname: str,
    decomp_var: str,
    title: Optional[str],
    annotate_shares: bool,
    notes: Optional[str],
) -> None:
    """Finalize the plot with titles, labels, and styling."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for coefplot")

    bar_data = plot_data["bar_data"]

    # Set x-axis labels
    plot_labels = [bar.name.replace("_", " ") for bar in bar_data]
    positions = [bar.position for bar in bar_data]

    ax.set_xticks(positions)
    ax.set_xticklabels(plot_labels, rotation=45, ha="right")

    # Set title
    if title is None:
        title = f"Decomposition of {depvarname} on {decomp_var} by Covariates"

    if annotate_shares:
        title += "\n(Normalized shares in parentheses for the decomposition section)"

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_ylabel("Difference (units)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.8)

    # Add notes
    if notes is not None:
        plt.figtext(
            0.02,
            0.02,
            notes,
            fontsize=9,
            style="italic",
            wrap=True,
            ha="left",
            va="bottom",
        )
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
    elif annotate_shares:
        default_note = "Mediator bars show: Absolute effect, (% of total effect), (% of explained effect in navy)."
        plt.figtext(
            0.02,
            0.02,
            default_note,
            fontsize=9,
            style="italic",
            wrap=True,
            ha="left",
            va="bottom",
        )
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.20)
    else:
        plt.tight_layout()
