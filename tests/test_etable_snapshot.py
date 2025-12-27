"""Snapshot tests for etable() output formats.

These tests use syrupy to capture and verify the exact output of etable()
across different output formats (gt, tex, md, df, html).

Run with: pixi run -e snapshot snapshot-test
Update snapshots with: pixi run -e snapshot snapshot-update
"""

import io
import re
import sys

import pytest
from syrupy.assertion import SnapshotAssertion

import pyfixest as pf
from pyfixest.report.summarize import etable
from pyfixest.utils.dgps import gelbach_data
from pyfixest.utils.utils import get_data


def normalize_gt_html(html: str) -> str:
    """Normalize Great Tables HTML by replacing dynamic IDs with a constant.

    Great Tables generates random IDs like 'ypgifbnuug' for each table,
    which causes snapshot tests to fail. This function replaces them with
    a constant ID 'gt_table' for stable comparisons.
    """
    # Find the dynamic ID pattern (10 lowercase letters)
    pattern = r'id="([a-z]{10})"'
    match = re.search(pattern, html)
    if match:
        dynamic_id = match.group(1)
        # Replace all occurrences of the dynamic ID
        html = html.replace(dynamic_id, "gt_table")
    return html


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_models():
    """Create basic models for snapshot testing."""
    df = get_data()
    fit1 = pf.feols("Y ~ X1 + X2 | f1", data=df)
    fit2 = pf.feols("Y ~ X1 + X2 | f1 + f2", data=df)
    return [fit1, fit2]


@pytest.fixture
def single_model():
    """Create a single model for snapshot testing."""
    df = get_data()
    return pf.feols("Y ~ X1 + X2 | f1", data=df)


@pytest.fixture
def iv_model():
    """Create an IV model for snapshot testing."""
    df = get_data()
    return pf.feols("Y ~ X2 | f1 | X1 ~ Z1", data=df)


@pytest.fixture
def poisson_model():
    """Create a Poisson model for snapshot testing."""
    df = get_data(model="Fepois")
    return pf.fepois("Y ~ X1 + X2 + f2 | f1", data=df, vcov={"CRV1": "f1+f2"})


@pytest.fixture
def gelbach_decomposition():
    """Create Gelbach decomposition for snapshot testing."""
    data = gelbach_data(nobs=200)
    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)
    gb = fit.decompose(param="x1", seed=98765, reps=25, only_coef=True)
    return gb


@pytest.fixture
def fixest_multi():
    """Create a FixestMulti object for snapshot testing."""
    df = get_data()
    return pf.feols("Y + Y2 ~ X1 + X2 | f1", data=df)


@pytest.fixture
def model_with_categoricals():
    """Create a model with categorical variables for cat_template testing."""
    df = get_data()
    return pf.feols("Y ~ X1 + C(f1) + C(f2)", data=df)


# ============================================================================
# Helper for markdown output (prints to stdout)
# ============================================================================


def capture_md_output(models, **kwargs):
    """Capture markdown output from etable which prints to stdout."""
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        etable(models, type="md", **kwargs)
    finally:
        sys.stdout = old_stdout
    return captured.getvalue()


# ============================================================================
# Snapshot Tests: Basic etable() - GT format
# ============================================================================


@pytest.mark.snapshot
class TestEtableGt:
    """Snapshot tests for etable() with type='gt'."""

    def test_basic_gt(self, basic_models, snapshot: SnapshotAssertion):
        """Test basic GT table output."""
        result = etable(basic_models, type="gt")
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_single_model_gt(self, single_model, snapshot: SnapshotAssertion):
        """Test GT output with single model."""
        result = etable(single_model, type="gt")
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_gt_with_significance_codes(
        self, basic_models, snapshot: SnapshotAssertion
    ):
        """Test GT with custom significance codes."""
        result = etable(basic_models, type="gt", signif_code=[0.01, 0.05, 0.1])
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_gt_with_coef_fmt(self, basic_models, snapshot: SnapshotAssertion):
        """Test GT with custom coefficient format."""
        result = etable(basic_models, type="gt", coef_fmt="b (se)\nt [p]")
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_gt_with_keep(self, basic_models, snapshot: SnapshotAssertion):
        """Test GT with keep parameter."""
        result = etable(basic_models, type="gt", keep=["X1"])
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_gt_with_drop(self, basic_models, snapshot: SnapshotAssertion):
        """Test GT with drop parameter."""
        result = etable(basic_models, type="gt", drop=["X2"])
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_gt_with_labels(self, basic_models, snapshot: SnapshotAssertion):
        """Test GT with custom labels."""
        result = etable(
            basic_models,
            type="gt",
            labels={"X1": "Variable One", "X2": "Variable Two"},
            felabels={"f1": "Fixed Effect 1", "f2": "Fixed Effect 2"},
        )
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_gt_with_notes(self, basic_models, snapshot: SnapshotAssertion):
        """Test GT with custom notes."""
        result = etable(basic_models, type="gt", notes="Custom notes for this table.")
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_gt_with_model_heads(self, basic_models, snapshot: SnapshotAssertion):
        """Test GT with custom model headers."""
        result = etable(
            basic_models, type="gt", model_heads=["Model A", "Model B"], head_order="dh"
        )
        assert normalize_gt_html(result.as_raw_html()) == snapshot


# ============================================================================
# Snapshot Tests: etable() - LaTeX format
# ============================================================================


@pytest.mark.snapshot
class TestEtableTex:
    """Snapshot tests for etable() with type='tex'."""

    def test_basic_tex(self, basic_models, snapshot: SnapshotAssertion):
        """Test basic LaTeX table output."""
        result = etable(basic_models, type="tex")
        assert result == snapshot

    def test_tex_with_notes(self, basic_models, snapshot: SnapshotAssertion):
        """Test LaTeX with custom notes."""
        result = etable(basic_models, type="tex", notes="Custom notes here.")
        assert result == snapshot

    def test_tex_with_model_heads_dh(self, basic_models, snapshot: SnapshotAssertion):
        """Test LaTeX with custom model headers (dh order)."""
        result = etable(
            basic_models,
            type="tex",
            model_heads=["Model A", "Model B"],
            head_order="dh",
        )
        assert result == snapshot

    def test_tex_with_model_heads_hd(self, basic_models, snapshot: SnapshotAssertion):
        """Test LaTeX with custom model headers (hd order)."""
        result = etable(
            basic_models,
            type="tex",
            model_heads=["Model A", "Model B"],
            head_order="hd",
        )
        assert result == snapshot

    def test_tex_with_model_heads_h(self, basic_models, snapshot: SnapshotAssertion):
        """Test LaTeX with custom model headers (h only)."""
        result = etable(
            basic_models,
            type="tex",
            model_heads=["Model A", "Model B"],
            head_order="h",
        )
        assert result == snapshot

    def test_tex_with_model_heads_d(self, basic_models, snapshot: SnapshotAssertion):
        """Test LaTeX with custom model headers (d only)."""
        result = etable(
            basic_models,
            type="tex",
            model_heads=["Model A", "Model B"],
            head_order="d",
        )
        assert result == snapshot


# ============================================================================
# Snapshot Tests: etable() - Markdown format
# ============================================================================


@pytest.mark.snapshot
class TestEtableMd:
    """Snapshot tests for etable() with type='md'."""

    def test_basic_md(self, basic_models, snapshot: SnapshotAssertion):
        """Test basic markdown output."""
        result = capture_md_output(basic_models)
        assert result == snapshot

    def test_md_with_notes(self, basic_models, snapshot: SnapshotAssertion):
        """Test markdown with notes."""
        result = capture_md_output(basic_models, notes="Markdown notes.")
        assert result == snapshot


# ============================================================================
# Snapshot Tests: etable() - DataFrame format
# ============================================================================


@pytest.mark.snapshot
class TestEtableDf:
    """Snapshot tests for etable() with type='df'."""

    def test_basic_df(self, basic_models, snapshot: SnapshotAssertion):
        """Test basic DataFrame output."""
        result = etable(basic_models, type="df")
        # Convert to string for stable snapshot comparison
        assert result.to_string() == snapshot

    def test_df_with_custom_stats(self, basic_models, snapshot: SnapshotAssertion):
        """Test DataFrame with custom statistics."""
        fit1, fit2 = basic_models
        result = etable(
            models=basic_models,
            type="df",
            custom_stats={
                "conf_int_lb": [fit1._conf_int[0], fit2._conf_int[0]],
                "conf_int_ub": [fit1._conf_int[1], fit2._conf_int[1]],
            },
            coef_fmt="b [conf_int_lb, conf_int_ub]",
        )
        assert result.to_string() == snapshot


# ============================================================================
# Snapshot Tests: etable() - HTML format
# ============================================================================


@pytest.mark.snapshot
class TestEtableHtml:
    """Snapshot tests for etable() with type='html'."""

    def test_basic_html(self, basic_models, snapshot: SnapshotAssertion):
        """Test basic HTML output."""
        result = etable(basic_models, type="html")
        assert normalize_gt_html(result) == snapshot


# ============================================================================
# Snapshot Tests: Special Model Types
# ============================================================================


@pytest.mark.snapshot
class TestEtableSpecialModels:
    """Snapshot tests for etable() with special model types."""

    def test_iv_model(self, iv_model, single_model, snapshot: SnapshotAssertion):
        """Test etable with IV model."""
        result = etable([iv_model, single_model], type="gt")
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_poisson_model(self, poisson_model, snapshot: SnapshotAssertion):
        """Test etable with Poisson model."""
        result = etable(poisson_model, type="gt")
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_mixed_models(
        self, basic_models, poisson_model, snapshot: SnapshotAssertion
    ):
        """Test etable with mixed model types."""
        result = etable([*basic_models, poisson_model], type="gt")
        assert normalize_gt_html(result.as_raw_html()) == snapshot


# ============================================================================
# Snapshot Tests: Advanced etable() Parameters
# ============================================================================


@pytest.mark.snapshot
class TestEtableAdvancedParams:
    """Snapshot tests for etable() advanced parameters."""

    def test_custom_model_stats(self, basic_models, snapshot: SnapshotAssertion):
        """Test etable with custom_model_stats."""
        fit1, fit2 = basic_models
        result = etable(
            basic_models,
            type="gt",
            custom_model_stats={
                "Mean Y": [fit1._Y.mean(), fit2._Y.mean()],
            },
        )
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_exact_match(self, basic_models, snapshot: SnapshotAssertion):
        """Test etable with exact_match=True for keep parameter."""
        result = etable(basic_models, type="gt", keep=["X1"], exact_match=True)
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_cat_template(self, model_with_categoricals, snapshot: SnapshotAssertion):
        """Test etable with cat_template for categorical variable labels."""
        result = etable(
            model_with_categoricals,
            type="gt",
            cat_template="{variable}={value_int}",  # Custom format for categorical vars
        )
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_show_fe_false(self, basic_models, snapshot: SnapshotAssertion):
        """Test etable with show_fe=False to hide fixed effects panel."""
        result = etable(basic_models, type="gt", show_fe=False)
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_custom_fe_symbols(self, basic_models, snapshot: SnapshotAssertion):
        """Test etable with custom fixed effects symbols."""
        result = etable(basic_models, type="gt", fe_present="Yes", fe_absent="No")
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_thousands_sep(self, basic_models, snapshot: SnapshotAssertion):
        """Test etable with thousands separator enabled."""
        result = etable(basic_models, type="gt", thousands_sep=True)
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_scientific_notation(self, basic_models, snapshot: SnapshotAssertion):
        """Test etable with scientific notation disabled."""
        result = etable(basic_models, type="gt", scientific_notation=False)
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_digits(self, basic_models, snapshot: SnapshotAssertion):
        """Test etable with custom digits for rounding."""
        result = etable(basic_models, type="gt", digits=5)
        assert normalize_gt_html(result.as_raw_html()) == snapshot


# ============================================================================
# Snapshot Tests: Model Input Variations
# ============================================================================


@pytest.mark.snapshot
class TestEtableModelInputs:
    """Snapshot tests for different model input types to etable()."""

    def test_fixest_multi_direct(self, fixest_multi, snapshot: SnapshotAssertion):
        """Test etable with FixestMulti object passed directly."""
        result = etable(fixest_multi, type="gt")
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_fixest_multi_method(self, fixest_multi, snapshot: SnapshotAssertion):
        """Test FixestMulti.etable() method directly."""
        result = fixest_multi.etable(type="gt")
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_fixest_multi_to_list(self, fixest_multi, snapshot: SnapshotAssertion):
        """Test etable with FixestMulti.to_list() as input."""
        result = etable(fixest_multi.to_list(), type="gt")
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_explicit_model_list(self, snapshot: SnapshotAssertion):
        """Test etable with explicit list of different models."""
        df = get_data()
        fit1 = pf.feols("Y ~ X1", data=df)
        fit2 = pf.feols("Y ~ X1 + X2", data=df)
        fit3 = pf.feols("Y ~ X1 + X2 | f1", data=df)
        result = etable([fit1, fit2, fit3], type="gt")
        assert normalize_gt_html(result.as_raw_html()) == snapshot


# ============================================================================
# Snapshot Tests: GelbachDecomposition.etable()
# ============================================================================


@pytest.mark.snapshot
class TestGelbachEtable:
    """Snapshot tests for GelbachDecomposition.etable()."""

    def test_gelbach_basic_gt(self, gelbach_decomposition, snapshot: SnapshotAssertion):
        """Test basic Gelbach decomposition table."""
        result = gelbach_decomposition.etable(type="gt", digits=3)
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_gelbach_levels_panel(
        self, gelbach_decomposition, snapshot: SnapshotAssertion
    ):
        """Test Gelbach with levels panel only."""
        result = gelbach_decomposition.etable(panels="levels", type="gt", digits=3)
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_gelbach_share_full_panel(
        self, gelbach_decomposition, snapshot: SnapshotAssertion
    ):
        """Test Gelbach with share_full panel only."""
        result = gelbach_decomposition.etable(panels="share_full", type="gt", digits=3)
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_gelbach_share_explained_panel(
        self, gelbach_decomposition, snapshot: SnapshotAssertion
    ):
        """Test Gelbach with share_explained panel only."""
        result = gelbach_decomposition.etable(
            panels="share_explained", type="gt", digits=3
        )
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_gelbach_with_caption(
        self, gelbach_decomposition, snapshot: SnapshotAssertion
    ):
        """Test Gelbach with custom caption."""
        result = gelbach_decomposition.etable(
            caption="Gelbach Decomposition Results", type="gt", digits=3
        )
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_gelbach_with_column_heads(
        self, gelbach_decomposition, snapshot: SnapshotAssertion
    ):
        """Test Gelbach with custom column headers."""
        result = gelbach_decomposition.etable(
            column_heads=["Total", "Direct", "Mediated"], type="gt", digits=3
        )
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_gelbach_with_panel_heads(
        self, gelbach_decomposition, snapshot: SnapshotAssertion
    ):
        """Test Gelbach with custom panel headers."""
        result = gelbach_decomposition.etable(
            panels="all",
            panel_heads=["Absolute Values", "Share of Total", "Share of Explained"],
            type="gt",
            digits=3,
        )
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_gelbach_tex(self, gelbach_decomposition, snapshot: SnapshotAssertion):
        """Test Gelbach decomposition LaTeX output."""
        result = gelbach_decomposition.etable(type="tex", digits=3)
        assert result == snapshot

    def test_gelbach_df(self, gelbach_decomposition, snapshot: SnapshotAssertion):
        """Test Gelbach decomposition DataFrame output."""
        result = gelbach_decomposition.etable(type="df", digits=3)
        assert result.to_string() == snapshot


# ============================================================================
# Snapshot Tests: GelbachDecomposition Advanced Parameters
# ============================================================================


@pytest.mark.snapshot
class TestGelbachAdvancedParams:
    """Snapshot tests for GelbachDecomposition.etable() advanced parameters."""

    def test_rgroup_sep_tb(self, gelbach_decomposition, snapshot: SnapshotAssertion):
        """Test Gelbach with rgroup_sep='tb' (top and bottom separators)."""
        result = gelbach_decomposition.etable(type="gt", digits=3, rgroup_sep="tb")
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_rgroup_sep_t(self, gelbach_decomposition, snapshot: SnapshotAssertion):
        """Test Gelbach with rgroup_sep='t' (top separator only)."""
        result = gelbach_decomposition.etable(type="gt", digits=3, rgroup_sep="t")
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_rgroup_sep_b(self, gelbach_decomposition, snapshot: SnapshotAssertion):
        """Test Gelbach with rgroup_sep='b' (bottom separator only)."""
        result = gelbach_decomposition.etable(type="gt", digits=3, rgroup_sep="b")
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_rgroup_sep_none(self, gelbach_decomposition, snapshot: SnapshotAssertion):
        """Test Gelbach with rgroup_sep='' (no separators)."""
        result = gelbach_decomposition.etable(type="gt", digits=3, rgroup_sep="")
        assert normalize_gt_html(result.as_raw_html()) == snapshot

    def test_add_notes(self, gelbach_decomposition, snapshot: SnapshotAssertion):
        """Test Gelbach with add_notes parameter."""
        result = gelbach_decomposition.etable(
            type="gt", digits=3, add_notes="Additional custom note for the table."
        )
        assert normalize_gt_html(result.as_raw_html()) == snapshot
