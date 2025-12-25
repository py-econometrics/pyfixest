"""Snapshot tests for coefplot() and iplot() output.

These tests use syrupy to capture and verify the exact visual output of
coefplot() and iplot() across matplotlib (SVG) and lets_plot (HTML) backends.

Run with: pixi run -e snapshot snapshot-test
Update snapshots with: pixi run -e snapshot snapshot-update
"""

import re
from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

import pyfixest as pf
from pyfixest.estimation.estimation import feols
from pyfixest.report.visualize import _HAS_LETS_PLOT, coefplot, iplot
from pyfixest.utils.utils import get_data

matplotlib.use("Agg")  # Non-interactive backend for testing


# ============================================================================
# Helper Functions
# ============================================================================


def figure_to_svg(fig: plt.Figure) -> str:
    """Convert matplotlib Figure to normalized SVG string."""
    buf = BytesIO()
    fig.savefig(buf, format="svg")
    buf.seek(0)
    svg = buf.read().decode("utf-8")
    plt.close(fig)  # Clean up
    return normalize_svg(svg)


def normalize_svg(svg: str) -> str:
    """Normalize matplotlib SVG for stable snapshot comparison.

    Removes/normalizes:
    - Timestamps in comments and metadata
    - Randomly generated path/clip-path IDs (order-preserving replacement)
    - Creation date metadata
    - Version-specific generator comments
    """
    # Remove matplotlib generator comments with version/date
    svg = re.sub(
        r"<!-- Created with matplotlib \(.*?\) -->", "<!-- matplotlib -->", svg
    )

    # Normalize date metadata with timestamp (e.g., 2025-12-24T17:12:55.033087)
    svg = re.sub(r"<dc:date>.*?</dc:date>", "<dc:date>NORMALIZED</dc:date>", svg)

    # Build a mapping of dynamic IDs to stable IDs in order of first appearance
    id_mapping = {}
    counters = {"clip": 0, "marker": 0, "font": 0, "path": 0}

    def get_stable_id(match):
        """Replace dynamic ID with stable ID based on first appearance order."""
        full_id = match.group(1)
        if full_id in id_mapping:
            return f'id="{id_mapping[full_id]}"'

        # Determine ID type and assign stable name
        if full_id.startswith("clip"):
            stable = f"clip_{counters['clip']}"
            counters["clip"] += 1
        elif full_id.startswith("m") and len(full_id) > 8:
            stable = f"marker_{counters['marker']}"
            counters["marker"] += 1
        elif full_id.startswith("DejaVu"):
            stable = f"font_{counters['font']}"
            counters["font"] += 1
        elif full_id.startswith("p") and len(full_id) > 8:
            stable = f"path_{counters['path']}"
            counters["path"] += 1
        else:
            stable = f"id_{len(id_mapping)}"

        id_mapping[full_id] = stable
        return f'id="{stable}"'

    # First pass: replace all id="..." definitions
    svg = re.sub(r'id="([a-zA-Z][a-zA-Z0-9_-]*)"', get_stable_id, svg)

    # Second pass: replace all references (#id and url(#id))
    for old_id, new_id in id_mapping.items():
        svg = svg.replace(f"#{old_id}", f"#{new_id}")
        svg = svg.replace(f'href="#{old_id}"', f'href="#{new_id}"')

    # Normalize other matplotlib-generated IDs (32-char hex)
    svg = re.sub(r'id="([a-f0-9]{32})"', 'id="normalized_id"', svg)

    # Normalize image IDs that may vary
    svg = re.sub(r'id="image[a-f0-9]+"', 'id="image_normalized"', svg)

    return svg


def letsplot_to_html(plot) -> str:
    """Convert lets_plot ggplot object to normalized HTML string."""
    if not _HAS_LETS_PLOT:
        pytest.skip("lets_plot not installed")

    import os
    import tempfile

    from lets_plot import ggsave

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        filepath = f.name
    try:
        ggsave(plot, filepath, iframe=False)
        with open(filepath) as f:
            html = f.read()
        return normalize_letsplot_html(html)
    finally:
        os.unlink(filepath)


def normalize_letsplot_html(html: str) -> str:
    """Normalize lets_plot HTML for stable snapshot comparison.

    Removes/normalizes:
    - Random element IDs
    - Timestamp-based identifiers
    - Version strings
    """
    # Normalize random IDs in HTML attributes (typically short random strings like "4VqpcV")
    html = re.sub(r'id="[a-zA-Z0-9_-]{4,32}"', 'id="lp_normalized"', html)

    # Normalize random IDs in JavaScript getElementById calls
    html = re.sub(
        r'getElementById\("[a-zA-Z0-9_-]{4,32}"\)',
        'getElementById("lp_normalized")',
        html,
    )

    # Normalize data attributes with random values
    html = re.sub(
        r'data-lets-plot-id="[^"]*"', 'data-lets-plot-id="normalized"', html
    )

    # Remove version-specific references
    html = re.sub(r"lets-plot-[0-9.]+", "lets-plot-VERSION", html)

    # Normalize any UUID-like strings
    html = re.sub(
        r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
        "UUID-NORMALIZED",
        html,
    )

    return html


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def plot_data():
    """Shared data for all plot tests."""
    data = get_data()
    data["f2"] = pd.Categorical(data["f2"])
    return data


@pytest.fixture(scope="module")
def iplot_model(plot_data):
    """Model with interaction variables for iplot."""
    return feols(fml="Y ~ i(f2, X1) | f1", data=plot_data, vcov="iid")


@pytest.fixture(scope="module")
def coefplot_model(plot_data):
    """Simple model for coefplot."""
    return feols(fml="Y ~ X1 + X2 | f1", data=plot_data, vcov="iid")


@pytest.fixture(scope="module")
def model_pair(plot_data):
    """Two models for multi-model plots."""
    fit1 = feols(fml="Y ~ i(f1)", data=plot_data)
    fit2 = feols(fml="Y ~ i(f1) + X2", data=plot_data)
    return [fit1, fit2]


# ============================================================================
# Coefplot Matplotlib Tests (10 tests)
# ============================================================================


@pytest.mark.snapshot
class TestCoefplotMatplotlib:
    """Snapshot tests for coefplot() with matplotlib backend."""

    def test_basic(self, coefplot_model, snapshot: SnapshotAssertion):
        """Basic coefplot with default settings."""
        fig = coefplot(coefplot_model, plot_backend="matplotlib")
        assert figure_to_svg(fig) == snapshot

    def test_with_title(self, coefplot_model, snapshot: SnapshotAssertion):
        """Coefplot with custom title."""
        fig = coefplot(coefplot_model, plot_backend="matplotlib", title="Custom Title")
        assert figure_to_svg(fig) == snapshot

    def test_coord_flip_false(self, coefplot_model, snapshot: SnapshotAssertion):
        """Coefplot without coordinate flip."""
        fig = coefplot(coefplot_model, plot_backend="matplotlib", coord_flip=False)
        assert figure_to_svg(fig) == snapshot

    def test_with_intercepts(self, coefplot_model, snapshot: SnapshotAssertion):
        """Coefplot with reference lines."""
        fig = coefplot(
            coefplot_model, plot_backend="matplotlib", yintercept=0, xintercept=2.0
        )
        assert figure_to_svg(fig) == snapshot

    def test_with_keep(self, coefplot_model, snapshot: SnapshotAssertion):
        """Coefplot filtering with keep."""
        fig = coefplot(coefplot_model, plot_backend="matplotlib", keep=["X1"])
        assert figure_to_svg(fig) == snapshot

    def test_exact_match(self, coefplot_model, snapshot: SnapshotAssertion):
        """Coefplot with exact_match for keep."""
        fig = coefplot(
            coefplot_model, plot_backend="matplotlib", keep=["X1"], exact_match=True
        )
        assert figure_to_svg(fig) == snapshot

    def test_rotate_xticks(self, coefplot_model, snapshot: SnapshotAssertion):
        """Coefplot with rotated labels."""
        fig = coefplot(
            coefplot_model,
            plot_backend="matplotlib",
            rotate_xticks=45,
            coord_flip=False,
        )
        assert figure_to_svg(fig) == snapshot

    def test_labels(self, coefplot_model, snapshot: SnapshotAssertion):
        """Coefplot with custom labels."""
        fig = coefplot(
            coefplot_model,
            plot_backend="matplotlib",
            labels={"X1": "Variable One", "X2": "Variable Two"},
        )
        assert figure_to_svg(fig) == snapshot

    def test_multi_model(self, model_pair, snapshot: SnapshotAssertion):
        """Coefplot with multiple models."""
        fig = coefplot(model_pair, plot_backend="matplotlib")
        assert figure_to_svg(fig) == snapshot

    def test_joint_both(self, iplot_model, snapshot: SnapshotAssertion):
        """Coefplot with joint confidence intervals."""
        fig = coefplot(iplot_model, plot_backend="matplotlib", joint="both", seed=42)
        assert figure_to_svg(fig) == snapshot


# ============================================================================
# Iplot Matplotlib Tests (5 tests)
# ============================================================================


@pytest.mark.snapshot
class TestIplotMatplotlib:
    """Snapshot tests for iplot() with matplotlib backend."""

    def test_basic(self, iplot_model, snapshot: SnapshotAssertion):
        """Basic iplot with default settings."""
        fig = iplot(iplot_model, plot_backend="matplotlib")
        assert figure_to_svg(fig) == snapshot

    def test_cat_template(self, iplot_model, snapshot: SnapshotAssertion):
        """Iplot with cat_template."""
        fig = iplot(iplot_model, plot_backend="matplotlib", cat_template="{value}")
        assert figure_to_svg(fig) == snapshot

    def test_alpha(self, iplot_model, snapshot: SnapshotAssertion):
        """Iplot with 90% confidence intervals."""
        fig = iplot(iplot_model, plot_backend="matplotlib", alpha=0.1)
        assert figure_to_svg(fig) == snapshot

    def test_drop(self, iplot_model, snapshot: SnapshotAssertion):
        """Iplot with dropped coefficient."""
        fig = iplot(iplot_model, plot_backend="matplotlib", drop="T.12")
        assert figure_to_svg(fig) == snapshot

    def test_multi_model(self, model_pair, snapshot: SnapshotAssertion):
        """Iplot with multiple models."""
        fig = iplot(model_pair, plot_backend="matplotlib")
        assert figure_to_svg(fig) == snapshot


# ============================================================================
# Coefplot lets_plot Tests (5 tests)
# ============================================================================


@pytest.mark.snapshot
class TestCoefplotLetsPlot:
    """Snapshot tests for coefplot() with lets_plot backend."""

    @pytest.fixture(autouse=True)
    def skip_if_no_letsplot(self):
        """Skip tests if lets_plot is not installed."""
        if not _HAS_LETS_PLOT:
            pytest.skip("lets_plot not installed")

    def test_basic(self, coefplot_model, snapshot: SnapshotAssertion):
        """Basic coefplot with lets_plot."""
        plot = coefplot(coefplot_model, plot_backend="lets_plot")
        assert letsplot_to_html(plot) == snapshot

    def test_with_title(self, coefplot_model, snapshot: SnapshotAssertion):
        """Coefplot with title using lets_plot."""
        plot = coefplot(coefplot_model, plot_backend="lets_plot", title="Custom Title")
        assert letsplot_to_html(plot) == snapshot

    def test_coord_flip_false(self, coefplot_model, snapshot: SnapshotAssertion):
        """Coefplot without coord_flip using lets_plot."""
        plot = coefplot(coefplot_model, plot_backend="lets_plot", coord_flip=False)
        assert letsplot_to_html(plot) == snapshot

    def test_multi_model(self, model_pair, snapshot: SnapshotAssertion):
        """Coefplot with multiple models using lets_plot."""
        plot = coefplot(model_pair, plot_backend="lets_plot")
        assert letsplot_to_html(plot) == snapshot

    def test_rotate_xticks(self, coefplot_model, snapshot: SnapshotAssertion):
        """Coefplot with rotated labels using lets_plot."""
        plot = coefplot(
            coefplot_model,
            plot_backend="lets_plot",
            rotate_xticks=45,
            coord_flip=False,
        )
        assert letsplot_to_html(plot) == snapshot


# ============================================================================
# Iplot lets_plot Tests (5 tests)
# ============================================================================


@pytest.mark.snapshot
class TestIplotLetsPlot:
    """Snapshot tests for iplot() with lets_plot backend."""

    @pytest.fixture(autouse=True)
    def skip_if_no_letsplot(self):
        """Skip tests if lets_plot is not installed."""
        if not _HAS_LETS_PLOT:
            pytest.skip("lets_plot not installed")

    def test_basic(self, iplot_model, snapshot: SnapshotAssertion):
        """Basic iplot with lets_plot."""
        plot = iplot(iplot_model, plot_backend="lets_plot")
        assert letsplot_to_html(plot) == snapshot

    def test_cat_template(self, iplot_model, snapshot: SnapshotAssertion):
        """Iplot with cat_template using lets_plot."""
        plot = iplot(
            iplot_model, plot_backend="lets_plot", cat_template="{variable}={value}"
        )
        assert letsplot_to_html(plot) == snapshot

    def test_labels(self, iplot_model, snapshot: SnapshotAssertion):
        """Iplot with custom labels using lets_plot."""
        plot = iplot(
            iplot_model, plot_backend="lets_plot", labels={"f2": "Factor 2"}
        )
        assert letsplot_to_html(plot) == snapshot

    def test_joint(self, iplot_model, snapshot: SnapshotAssertion):
        """Iplot with joint CIs using lets_plot."""
        plot = iplot(iplot_model, plot_backend="lets_plot", joint=True, seed=42)
        assert letsplot_to_html(plot) == snapshot

    def test_rename_models(self, model_pair, snapshot: SnapshotAssertion):
        """Iplot with renamed models using lets_plot."""
        plot = iplot(
            model_pair,
            plot_backend="lets_plot",
            rename_models={"Y~i(f1)": "Model A", "Y~i(f1)+X2": "Model B"},
        )
        assert letsplot_to_html(plot) == snapshot
