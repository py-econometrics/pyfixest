"""Tests for the benchmark runner utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.bench import _prepare_demean_inputs
from benchmarks.plot import format_baseline_comparison_table
from benchmarks.run_benchmarks import _load_baseline, _save_baseline


@pytest.fixture
def sample_df():
    """Create a small DataFrame matching the DGP output schema."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "worker_id": rng.integers(0, 10, size=n),
        "firm_id": rng.integers(0, 5, size=n),
        "year": rng.integers(0, 3, size=n),
        "y": rng.standard_normal(n),
    })


class TestPrepareDemeanInputs:
    """Tests for _prepare_demean_inputs."""

    def test_single_feature(self, sample_df):
        y, flist, weights = _prepare_demean_inputs(sample_df, n_features=1)
        assert y.shape == (len(sample_df), 1)
        assert y.dtype == np.float64
        assert flist.shape == (len(sample_df), 3)
        assert flist.dtype == np.int64
        assert weights.shape == (len(sample_df),)
        assert weights.dtype == np.float64
        np.testing.assert_array_almost_equal(weights, 1.0)

    def test_multi_feature(self, sample_df):
        y, flist, weights = _prepare_demean_inputs(sample_df, n_features=5)
        assert y.shape == (len(sample_df), 5)
        assert y.dtype == np.float64
        # First column should be the actual y values
        np.testing.assert_array_equal(
            y[:, 0], sample_df["y"].values.astype(np.float64)
        )
        # Extra columns should be deterministic (same seed each call)
        y2, _, _ = _prepare_demean_inputs(sample_df, n_features=5)
        np.testing.assert_array_equal(y, y2)

    def test_custom_fe_columns_two_way(self, sample_df):
        fe_cols = ["worker_id", "firm_id"]
        y, flist, weights = _prepare_demean_inputs(
            sample_df, fe_columns=fe_cols,
        )
        assert flist.shape == (len(sample_df), 2)
        np.testing.assert_array_equal(
            flist[:, 0], sample_df["worker_id"].values.astype(np.int64)
        )
        np.testing.assert_array_equal(
            flist[:, 1], sample_df["firm_id"].values.astype(np.int64)
        )

    def test_default_fe_columns(self, sample_df):
        y, flist, weights = _prepare_demean_inputs(sample_df)
        assert flist.shape == (len(sample_df), 3)
        np.testing.assert_array_equal(
            flist[:, 0], sample_df["worker_id"].values.astype(np.int64)
        )
        np.testing.assert_array_equal(
            flist[:, 1], sample_df["firm_id"].values.astype(np.int64)
        )
        np.testing.assert_array_equal(
            flist[:, 2], sample_df["year"].values.astype(np.int64)
        )

    def test_multi_feature_with_custom_fe(self, sample_df):
        fe_cols = ["worker_id", "firm_id"]
        y, flist, weights = _prepare_demean_inputs(
            sample_df, n_features=3, fe_columns=fe_cols,
        )
        assert y.shape == (len(sample_df), 3)
        assert flist.shape == (len(sample_df), 2)


@pytest.fixture
def sample_summary():
    """Create a minimal summary DataFrame for baseline tests."""
    return pd.DataFrame({
        "scenario": ["easy", "medium"],
        "backend": ["numba", "numba"],
        "n_obs": [100_000, 600_000],
        "n_workers": [10_000, 50_000],
        "n_firms": [1_000, 5_000],
        "connected_set_fraction": [1.0, 0.95],
        "demean_time_median": [0.5, 2.0],
        "demean_time_min": [0.45, 1.8],
        "demean_time_max": [0.55, 2.2],
        "demean_converged": [True, True],
        "n_runs": [3, 3],
    })


class TestBaselineSaveLoad:
    """Tests for baseline save and load roundtrip."""

    def test_save_and_load_roundtrip(self, sample_summary, monkeypatch, tmp_path):
        import benchmarks.run_benchmarks as rb

        monkeypatch.setattr(rb, "BASELINES_DIR", tmp_path)
        _save_baseline(sample_summary, "test_v1")
        loaded = _load_baseline("test_v1")
        pd.testing.assert_frame_equal(
            loaded, sample_summary, check_dtype=False,
        )

    def test_load_missing_baseline_raises(self, monkeypatch, tmp_path):
        import benchmarks.run_benchmarks as rb

        monkeypatch.setattr(rb, "BASELINES_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            _load_baseline("nonexistent")


class TestBaselineComparisonTable:
    """Tests for format_baseline_comparison_table."""

    def test_identical_runs_show_1x_speedup(self, sample_summary):
        table = format_baseline_comparison_table(sample_summary, sample_summary)
        assert "1.00x" in table
        assert "+0.0%" in table

    def test_new_backend_labelled(self, sample_summary):
        current = pd.concat([
            sample_summary,
            pd.DataFrame({
                "scenario": ["easy"],
                "backend": ["rust"],
                "n_obs": [100_000],
                "n_workers": [10_000],
                "n_firms": [1_000],
                "connected_set_fraction": [1.0],
                "demean_time_median": [0.3],
                "demean_time_min": [0.25],
                "demean_time_max": [0.35],
                "demean_converged": [True],
                "n_runs": [3],
            }),
        ], ignore_index=True)
        table = format_baseline_comparison_table(current, sample_summary)
        assert "new" in table

    def test_faster_run_shows_positive_delta(self):
        baseline = pd.DataFrame({
            "scenario": ["easy"],
            "backend": ["numba"],
            "demean_time_median": [2.0],
        })
        current = pd.DataFrame({
            "scenario": ["easy"],
            "backend": ["numba"],
            "demean_time_median": [1.0],
        })
        table = format_baseline_comparison_table(current, baseline)
        assert "2.00x" in table
        assert "+50.0%" in table
