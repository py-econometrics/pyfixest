import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.contour import QuadContourSet

import pyfixest as pf

pytestmark = pytest.mark.plots


@pytest.fixture
def plot_analysis():
    rng = np.random.default_rng(20260827)
    n_obs = 180
    benchmark = rng.normal(size=n_obs)
    control = rng.normal(size=n_obs)
    treatment = 0.4 * benchmark + 0.2 * control + rng.normal(size=n_obs)
    outcome = 0.9 * treatment + 0.5 * benchmark - 0.2 * control + rng.normal(size=n_obs)
    data = pd.DataFrame(
        {
            "outcome": outcome,
            "treatment": treatment,
            "benchmark": benchmark,
            "control": control,
        }
    )
    fit = pf.feols("outcome ~ treatment + benchmark + control", data)
    return fit.sensitivity_analysis("treatment")


def _collection_with_label(axes, label):
    return next(
        collection for collection in axes.collections if collection.get_label() == label
    )


def test_contour_forwards_benchmarks_to_marker_artists(plot_analysis):
    expected = plot_analysis.ovb_bounds("benchmark", kd=[0.5, 1], ky=[1, 2])

    figure = plot_analysis.plot(
        plot_type="contour",
        benchmark_covariates="benchmark",
        kd=[0.5, 1],
        ky=[1, 2],
        grid_size=40,
    )
    axes = figure.axes[0]
    marker_artist = _collection_with_label(axes, "Benchmark bounds")

    np.testing.assert_allclose(
        marker_artist.get_offsets(),
        expected[["r2dz_x", "r2yz_dx"]].to_numpy(),
    )
    annotation_text = {text.get_text() for text in axes.texts}
    assert any("0.5x benchmark" in text for text in annotation_text)
    assert any("1x benchmark" in text for text in annotation_text)
    plt.close(figure)


def test_contour_adds_manual_bound_artists_and_labels(plot_analysis):
    figure = plot_analysis.plot(
        r2dz_x=[0.05, 0.1],
        r2yz_dx=[0.1, 0.2],
        manual_labels=["Plausible", "Severe"],
        col_contour="#123456",
        grid_size=40,
    )
    axes = figure.axes[0]
    marker_artist = _collection_with_label(axes, "Manual bounds")

    np.testing.assert_allclose(
        marker_artist.get_offsets(),
        np.array([[0.05, 0.1], [0.1, 0.2]]),
    )
    assert {"Plausible", "Severe"}.issubset({text.get_text() for text in axes.texts})
    assert any(
        isinstance(collection, QuadContourSet) for collection in axes.collections
    )
    plt.close(figure)


def test_t_value_contour_uses_bound_t_values(plot_analysis):
    expected = plot_analysis.ovb_bounds("benchmark", kd=0.5, ky=1, reduce=False, h0=0.2)
    figure = plot_analysis.plot(
        sensitivity_of="t-value",
        benchmark_covariates="benchmark",
        kd=0.5,
        ky=1,
        reduce=False,
        h0=0.2,
        grid_size=40,
    )
    axes = figure.axes[0]

    bound_annotation = next(
        text.get_text() for text in axes.texts if "0.5x benchmark" in text.get_text()
    )
    assert f"{expected.loc[0, 'adjusted_t']:.3g}" in bound_annotation
    plt.close(figure)


def test_extreme_plot_adds_scenario_lines_and_benchmark_rugs(plot_analysis):
    expected = plot_analysis.ovb_bounds(
        "benchmark", kd=[0.5, 1], ky=[0.5, 1], adjusted_estimates=False
    )
    figure = plot_analysis.plot(
        plot_type="extreme",
        benchmark_covariates="benchmark",
        kd=[0.5, 1],
        ky=[0.5, 1],
        r2yz_dx=[1, 0.5],
        grid_size=40,
    )
    axes = figure.axes[0]
    scenario_labels = {line.get_label() for line in axes.lines}
    rug_artist = _collection_with_label(axes, "Benchmark treatment bounds")

    assert {
        "Outcome partial R2: 100%",
        "Outcome partial R2: 50%",
        "Estimate threshold",
    }.issubset(scenario_labels)
    rug_segments = rug_artist.get_segments()
    np.testing.assert_allclose(
        [segment[0, 0] for segment in rug_segments], expected["r2dz_x"]
    )
    plt.close(figure)


def test_plot_reuses_supplied_axes(plot_analysis):
    original_figure, axes = plt.subplots()

    returned_figure = plot_analysis.plot(ax=axes, grid_size=40)

    assert returned_figure is original_figure
    assert returned_figure.axes == [axes]
    plt.close(returned_figure)


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"plot_type": "invalid"}, ValueError, "plot_type"),
        (
            {"plot_type": "extreme", "sensitivity_of": "t-value"},
            NotImplementedError,
            "estimates only",
        ),
        ({"sensitivity_of": "invalid"}, ValueError, "sensitivity_of"),
        ({"r2dz_x": 0.1}, ValueError, "supplied together"),
        (
            {"r2dz_x": [0.1, 0.2], "r2yz_dx": [0.1]},
            ValueError,
            "same length",
        ),
        ({"lim": 1}, ValueError, "strictly between"),
    ],
)
def test_plot_validates_options(plot_analysis, kwargs, error, message):
    with pytest.raises(error, match=message):
        plot_analysis.plot(grid_size=20, **kwargs)
