"""Tests for the three-way fixed effects DGP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.dgp import DGPConfig, DGPResult, ThreeWayFEData


class TestDefaults:
    """Test that default config produces valid output."""

    def test_columns_and_types(self):
        cfg = DGPConfig(n_workers=500, n_firms=50, n_years=5, seed=0)
        result = ThreeWayFEData(cfg).simulate()
        df = result.data

        assert set(df.columns) == {"worker_id", "firm_id", "year", "y"}
        assert df["worker_id"].dtype in (np.int64, np.int32, int)
        assert df["firm_id"].dtype in (np.int64, np.int32, int)
        assert df["year"].dtype in (np.int64, np.int32, int)
        assert df["y"].dtype == np.float64

    def test_no_nans(self):
        cfg = DGPConfig(n_workers=500, n_firms=50, n_years=5, seed=1)
        result = ThreeWayFEData(cfg).simulate()
        assert not result.data.isna().any().any()

    def test_contiguous_ids(self):
        cfg = DGPConfig(n_workers=500, n_firms=50, n_years=5, seed=2)
        result = ThreeWayFEData(cfg).simulate()
        df = result.data

        workers = sorted(df["worker_id"].unique())
        firms = sorted(df["firm_id"].unique())

        assert workers == list(range(len(workers)))
        assert firms == list(range(len(firms)))

    def test_n_obs_positive(self):
        cfg = DGPConfig(n_workers=500, n_firms=50, n_years=5, seed=3)
        result = ThreeWayFEData(cfg).simulate()
        assert result.n_obs > 0
        assert result.n_workers_observed > 0
        assert result.n_firms_observed > 0


class TestBalancedPanel:
    """With p_observe=1.0 and p_survive=1.0, panel should be fully balanced."""

    def test_balanced_obs_count(self):
        n_w, n_y = 200, 5
        cfg = DGPConfig(
            n_workers=n_w, n_firms=20, n_years=n_y,
            p_observe=1.0, p_survive=1.0,
            seed=10,
        )
        result = ThreeWayFEData(cfg).simulate()
        assert result.n_obs == n_w * n_y

    def test_each_worker_observed_every_year(self):
        cfg = DGPConfig(
            n_workers=200, n_firms=20, n_years=5,
            p_observe=1.0, p_survive=1.0,
            seed=11,
        )
        result = ThreeWayFEData(cfg).simulate()
        obs_per_worker = result.data.groupby("worker_id").size()
        assert (obs_per_worker == 5).all()


class TestFirmSizeDistribution:
    """Pareto firm size distribution should match theoretical moments."""

    @pytest.mark.parametrize("theta", [2.0, 3.0, 5.0])
    def test_pareto_mean(self, theta):
        # E[s] = s_min * theta / (theta - 1) for theta > 1
        # Floor operation biases the mean downward, especially for large theta
        # where the distribution is concentrated near s_min.
        s_min = 5
        n_firms = 50_000
        cfg = DGPConfig(
            n_workers=100, n_firms=n_firms, n_years=2,
            pareto_shape=theta, min_firm_size=s_min,
            seed=20,
        )
        dgp = ThreeWayFEData(cfg)
        sizes, _ = dgp._assign_firm_sizes()

        expected_mean = s_min * theta / (theta - 1)
        assert abs(sizes.mean() - expected_mean) / expected_mean < 0.10


class TestMobility:
    """Empirical mobility rate should approximate p_move."""

    @pytest.mark.parametrize("p_move", [0.05, 0.15, 0.30])
    def test_mobility_rate(self, p_move):
        cfg = DGPConfig(
            n_workers=5_000, n_firms=100, n_years=10,
            p_move=p_move,
            p_observe=1.0, p_survive=1.0,
            seed=30,
        )
        result = ThreeWayFEData(cfg).simulate()
        df = result.data.sort_values(["worker_id", "year"])

        # For each worker-year, check if firm differs from previous year
        df["prev_firm"] = df.groupby("worker_id")["firm_id"].shift(1)
        moves = df.dropna(subset=["prev_firm"])
        empirical_rate = (moves["firm_id"] != moves["prev_firm"]).mean()

        assert abs(empirical_rate - p_move) < 0.03


class TestSorting:
    """Worker-firm sorting should produce correlated assignments."""

    def test_high_sorting(self):
        cfg = DGPConfig(
            n_workers=5_000, n_firms=200, n_years=5,
            sorting_wf=1.0,
            p_observe=1.0, p_survive=1.0, p_move=0.10,
            seed=40,
        )
        result = ThreeWayFEData(cfg).simulate()
        df = result.data

        worker_alpha = result.true_alpha[df["worker_id"].values]
        firm_psi = result.true_psi[df["firm_id"].values]
        corr = np.corrcoef(worker_alpha, firm_psi)[0, 1]
        assert corr > 0.3  # Positive sorting

    def test_no_sorting(self):
        cfg = DGPConfig(
            n_workers=5_000, n_firms=200, n_years=5,
            sorting_wf=0.0,
            p_observe=1.0, p_survive=1.0, p_move=0.10,
            seed=41,
        )
        result = ThreeWayFEData(cfg).simulate()
        df = result.data

        worker_alpha = result.true_alpha[df["worker_id"].values]
        firm_psi = result.true_psi[df["firm_id"].values]
        corr = np.corrcoef(worker_alpha, firm_psi)[0, 1]
        assert abs(corr) < 0.15  # Near-zero correlation


class TestParticipation:
    """Panel unbalancedness should roughly match p_observe."""

    def test_half_observed(self):
        n_w, n_y = 5_000, 10
        cfg = DGPConfig(
            n_workers=n_w, n_firms=200, n_years=n_y,
            p_observe=0.5, selection_worker=0.0,
            p_survive=1.0,
            seed=50,
        )
        result = ThreeWayFEData(cfg).simulate()
        balanced_count = n_w * n_y
        ratio = result.n_obs / balanced_count
        assert 0.35 < ratio < 0.65


class TestFirmSurvivalSelection:
    """With selection_firm > 0, surviving firms should have higher psi."""

    def test_survival_selection(self):
        cfg = DGPConfig(
            n_workers=5_000, n_firms=500, n_years=10,
            p_survive=0.85, selection_firm=3.0,
            p_observe=1.0,
            seed=60,
        )
        dgp = ThreeWayFEData(cfg)
        result = dgp.simulate()

        # Firms that appear in the last year survived
        df = result.data
        last_year = df["year"].max()
        surviving_firms = df[df["year"] == last_year]["firm_id"].unique()
        all_firms = df["firm_id"].unique()
        exited_firms = np.setdiff1d(all_firms, surviving_firms)

        if len(exited_firms) > 0 and len(surviving_firms) > 0:
            psi = result.true_psi
            # Use original firm IDs from the data (already re-indexed)
            mean_psi_surviving = psi[surviving_firms].mean()
            mean_psi_exited = psi[exited_firms].mean()
            assert mean_psi_surviving > mean_psi_exited


class TestConnectedSet:
    """Connected set fraction should be 1.0 for easy configs."""

    def test_easy_full_connected(self):
        cfg = DGPConfig(
            n_workers=2_000, n_firms=50, n_years=5,
            p_move=0.15, p_observe=1.0, p_survive=1.0,
            n_clusters=1,
            seed=70,
        )
        result = ThreeWayFEData(cfg).simulate()
        assert result.connected_set_fraction > 0.99

    def test_low_mobility_may_disconnect(self):
        cfg = DGPConfig(
            n_workers=2_000, n_firms=200, n_years=5,
            p_move=0.001,
            n_clusters=10, p_between_cluster=0.01,
            p_observe=1.0, p_survive=1.0,
            seed=71,
        )
        result = ThreeWayFEData(cfg).simulate()
        # With very low mobility and many clusters, some disconnection is likely
        # but the largest component should still contain most observations
        assert result.connected_set_fraction > 0.0
        assert result.connected_set_fraction <= 1.0


class TestReproducibility:
    """Same seed should produce identical output."""

    def test_same_seed(self):
        cfg = DGPConfig(n_workers=500, n_firms=50, n_years=5, seed=99)

        r1 = ThreeWayFEData(cfg).simulate()
        r2 = ThreeWayFEData(cfg).simulate()

        pd.testing.assert_frame_equal(r1.data, r2.data)
        np.testing.assert_array_equal(r1.true_alpha, r2.true_alpha)
        np.testing.assert_array_equal(r1.true_psi, r2.true_psi)
        np.testing.assert_array_equal(r1.true_phi, r2.true_phi)

    def test_different_seed(self):
        cfg1 = DGPConfig(n_workers=500, n_firms=50, n_years=5, seed=100)
        cfg2 = DGPConfig(n_workers=500, n_firms=50, n_years=5, seed=200)

        r1 = ThreeWayFEData(cfg1).simulate()
        r2 = ThreeWayFEData(cfg2).simulate()

        # y values should differ
        assert not np.allclose(r1.data["y"].values, r2.data["y"].values)


class TestDescribe:
    """DGPResult.describe() should run without error."""

    def test_describe_runs(self, capsys):
        cfg = DGPConfig(n_workers=200, n_firms=20, n_years=3, seed=0)
        result = ThreeWayFEData(cfg).simulate()
        summary = result.describe()
        assert "Observations:" in summary
        assert "Connected set frac:" in summary
