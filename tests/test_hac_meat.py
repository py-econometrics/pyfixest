"""Unit tests for the HAC meat matrix functions in vcov_utils.

These tests exercise the numba implementations directly, providing a
ground-truth contract before porting to Rust. Each test is deterministic
and verifies the mathematical properties of the estimators.
"""

import numpy as np
import pytest

from pyfixest.estimation.internals.vcov_utils import (
    _dk_meat_panel,
    _get_bartlett_weights,
    _get_panel_idx,
    _hac_meat_loop,
    _nw_meat_panel,
    _nw_meat_time,
)

# ---------------------------------------------------------------------------
# _get_bartlett_weights
# ---------------------------------------------------------------------------


class TestBartlettWeights:
    def test_lag_zero(self):
        w = _get_bartlett_weights(lag=0)
        assert len(w) == 1
        assert w[0] == pytest.approx(0.5)

    def test_lag_four(self):
        w = _get_bartlett_weights(lag=4)
        assert len(w) == 5
        assert w[0] == pytest.approx(0.5)
        # weights[j] = 1 - j/(lag+1) = 1 - j/5
        assert w[1] == pytest.approx(0.8)
        assert w[2] == pytest.approx(0.6)
        assert w[3] == pytest.approx(0.4)
        assert w[4] == pytest.approx(0.2)

    def test_monotone_decreasing(self):
        w = _get_bartlett_weights(lag=10)
        # After index 0 (which is halved), weights should be strictly decreasing
        for j in range(1, len(w) - 1):
            assert w[j] > w[j + 1]


# ---------------------------------------------------------------------------
# _hac_meat_loop
# ---------------------------------------------------------------------------


class TestHacMeatLoop:
    def test_lag_zero_equals_xtx(self):
        """With lag=0, meat = 0.5*(S'S + S'S) = S'S (since weight[0]=0.5)."""
        scores = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        weights = _get_bartlett_weights(lag=0)
        meat = _hac_meat_loop(
            scores=scores, weights=weights, time_periods=3, k=2, lag=0
        )
        expected = scores.T @ scores
        np.testing.assert_allclose(meat, expected)

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        scores = rng.standard_normal((20, 3))
        weights = _get_bartlett_weights(lag=5)
        meat = _hac_meat_loop(
            scores=scores, weights=weights, time_periods=20, k=3, lag=5
        )
        np.testing.assert_allclose(meat, meat.T, atol=1e-12)

    def test_single_obs(self):
        """Single observation: meat = weight[0] * (s's + s's) = s's."""
        scores = np.array([[3.0, 7.0]])
        weights = _get_bartlett_weights(lag=0)
        meat = _hac_meat_loop(
            scores=scores, weights=weights, time_periods=1, k=2, lag=0
        )
        expected = scores.T @ scores
        np.testing.assert_allclose(meat, expected)

    def test_manual_lag1(self):
        """Manual verification with lag=1, 3 observations, 1 regressor."""
        scores = np.array([[1.0], [2.0], [3.0]])
        lag = 1
        weights = _get_bartlett_weights(lag=lag)
        # weights = [0.5, 0.5]
        # gamma_0 = scores[0:]'scores[0:] = [[1,2,3]] @ [[1],[2],[3]] = 14
        # lag=0: weight[0]*(gamma_0 + gamma_0) = 0.5*(14+14) = 14
        # gamma_1 = scores[1:]'scores[0:-1] = [[2,3]] @ [[1],[2]] = 8
        # lag=1: weight[1]*(gamma_1 + gamma_1') = 0.5*(8+8) = 8
        # total = 14 + 8 = 22
        meat = _hac_meat_loop(
            scores=scores, weights=weights, time_periods=3, k=1, lag=lag
        )
        np.testing.assert_allclose(meat, np.array([[22.0]]))


# ---------------------------------------------------------------------------
# _nw_meat_time
# ---------------------------------------------------------------------------


class TestNwMeatTime:
    def test_already_sorted(self):
        """When time_arr is already sorted, result should match direct hac_meat_loop."""
        scores = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        time_arr = np.array([1.0, 2.0, 3.0])
        lag = 1
        meat = _nw_meat_time(scores=scores, time_arr=time_arr, lag=lag)
        # Compare with direct call using sorted scores
        weights = _get_bartlett_weights(lag=lag)
        expected = _hac_meat_loop(
            scores=scores, weights=weights, time_periods=3, k=2, lag=lag
        )
        np.testing.assert_allclose(meat, expected)

    def test_unsorted_time(self):
        """When time_arr is unsorted, _nw_meat_time should sort by time first."""
        scores = np.array([[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]])
        time_arr = np.array([3.0, 1.0, 2.0])
        lag = 1

        meat = _nw_meat_time(scores=scores, time_arr=time_arr, lag=lag)

        # Manually sort
        order = np.argsort(time_arr)
        sorted_scores = scores[order]
        weights = _get_bartlett_weights(lag=lag)
        expected = _hac_meat_loop(
            scores=sorted_scores, weights=weights, time_periods=3, k=2, lag=lag
        )
        np.testing.assert_allclose(meat, expected)

    def test_symmetry(self):
        rng = np.random.default_rng(99)
        scores = rng.standard_normal((30, 4))
        time_arr = np.arange(30, dtype=float)
        meat = _nw_meat_time(scores=scores, time_arr=time_arr, lag=3)
        np.testing.assert_allclose(meat, meat.T, atol=1e-12)


# ---------------------------------------------------------------------------
# _nw_meat_panel
# ---------------------------------------------------------------------------


class TestNwMeatPanel:
    @staticmethod
    def _make_balanced_panel(n_units, n_periods, k, rng):
        """Create a balanced panel dataset sorted by (unit, time)."""
        n = n_units * n_periods
        scores = rng.standard_normal((n, k))
        panel_arr = np.repeat(np.arange(n_units), n_periods).astype(float)
        time_arr = np.tile(np.arange(n_periods), n_units).astype(float)
        order, _, starts, counts, panel_sorted, time_sorted = _get_panel_idx(
            panel_arr=panel_arr, time_arr=time_arr
        )
        return scores[order], time_sorted, panel_sorted, starts, counts

    def test_single_panel_matches_time(self):
        """With one panel unit, panel NW should equal time-series NW."""
        rng = np.random.default_rng(42)
        scores = rng.standard_normal((10, 2))
        time_arr = np.arange(10, dtype=float)
        panel_arr = np.zeros(10, dtype=float)
        starts = np.array([0])
        counts = np.array([10])
        lag = 2

        meat_panel = _nw_meat_panel(
            scores=scores,
            time_arr=time_arr,
            panel_arr=panel_arr,
            starts=starts,
            counts=counts,
            lag=lag,
        )
        meat_time = _nw_meat_time(scores=scores, time_arr=time_arr, lag=lag)
        np.testing.assert_allclose(meat_panel, meat_time, atol=1e-10)

    def test_lag_zero_equals_xtx(self):
        """With lag=0, NW panel meat = sum of per-panel S_i'S_i."""
        rng = np.random.default_rng(7)
        scores, time_arr, panel_arr, starts, counts = self._make_balanced_panel(
            n_units=5, n_periods=8, k=3, rng=rng
        )
        meat = _nw_meat_panel(
            scores=scores,
            time_arr=time_arr,
            panel_arr=panel_arr,
            starts=starts,
            counts=counts,
            lag=0,
        )
        # With lag=0, each panel contributes gamma0 = S_i'S_i
        expected = scores.T @ scores
        np.testing.assert_allclose(meat, expected, atol=1e-10)

    def test_symmetry(self):
        rng = np.random.default_rng(123)
        scores, time_arr, panel_arr, starts, counts = self._make_balanced_panel(
            n_units=10, n_periods=20, k=4, rng=rng
        )
        meat = _nw_meat_panel(
            scores=scores,
            time_arr=time_arr,
            panel_arr=panel_arr,
            starts=starts,
            counts=counts,
            lag=3,
        )
        np.testing.assert_allclose(meat, meat.T, atol=1e-12)

    def test_default_lag(self):
        """When lag=None, should default to floor(n_unique_times ** 0.25)."""
        rng = np.random.default_rng(55)
        n_units, n_periods, k = 5, 16, 2
        scores, time_arr, panel_arr, starts, counts = self._make_balanced_panel(
            n_units=n_units, n_periods=n_periods, k=k, rng=rng
        )
        # Default lag = floor(16 ** 0.25) = floor(2.0) = 2
        meat_default = _nw_meat_panel(
            scores=scores,
            time_arr=time_arr,
            panel_arr=panel_arr,
            starts=starts,
            counts=counts,
            lag=None,
        )
        meat_explicit = _nw_meat_panel(
            scores=scores,
            time_arr=time_arr,
            panel_arr=panel_arr,
            starts=starts,
            counts=counts,
            lag=2,
        )
        np.testing.assert_allclose(meat_default, meat_explicit, atol=1e-12)

    def test_unbalanced_panel(self):
        """Test with panels of different lengths."""
        # Panel 0: 3 obs, Panel 1: 2 obs
        scores = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        time_arr = np.array([0.0, 1.0, 2.0, 0.0, 1.0])
        panel_arr = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        starts = np.array([0, 3])
        counts = np.array([3, 2])
        lag = 1

        meat = _nw_meat_panel(
            scores=scores,
            time_arr=time_arr,
            panel_arr=panel_arr,
            starts=starts,
            counts=counts,
            lag=lag,
        )
        assert meat.shape == (2, 2)
        np.testing.assert_allclose(meat, meat.T, atol=1e-12)

        # Verify manually: compute per-panel contribution
        weights = _get_bartlett_weights(lag=1)
        # Panel 0: scores[0:3]
        s0 = scores[0:3]
        g0_0 = s0.T @ s0
        g0_1 = s0[1:].T @ s0[:2]
        panel0_contrib = g0_0 + weights[1] * (g0_1 + g0_1.T)

        # Panel 1: scores[3:5]
        s1 = scores[3:5]
        g1_0 = s1.T @ s1
        g1_1 = s1[1:].T @ s1[:1]
        panel1_contrib = g1_0 + weights[1] * (g1_1 + g1_1.T)

        expected = panel0_contrib + panel1_contrib
        np.testing.assert_allclose(meat, expected, atol=1e-12)

    def test_lag_exceeds_panel_length(self):
        """When lag > panel length, should clamp to count-1 per panel."""
        scores = np.array([[1.0], [2.0], [3.0]])
        time_arr = np.array([0.0, 1.0, 2.0])
        panel_arr = np.array([0.0, 0.0, 0.0])
        starts = np.array([0])
        counts = np.array([3])
        lag = 100  # way bigger than panel length

        meat = _nw_meat_panel(
            scores=scores,
            time_arr=time_arr,
            panel_arr=panel_arr,
            starts=starts,
            counts=counts,
            lag=lag,
        )
        # Should clamp to lag=2 (count-1), and still produce a valid result
        assert meat.shape == (1, 1)
        assert np.isfinite(meat).all()


# ---------------------------------------------------------------------------
# _dk_meat_panel
# ---------------------------------------------------------------------------


class TestDkMeatPanel:
    def test_single_obs_per_time(self):
        """When there's one obs per time period, DK aggregated scores = raw scores."""
        rng = np.random.default_rng(42)
        T = 20
        k = 3
        scores = rng.standard_normal((T, k))
        time_arr = np.arange(T, dtype=float)
        idx = np.arange(T)  # one obs per time
        lag = 2

        meat_dk = _dk_meat_panel(scores=scores, time_arr=time_arr, idx=idx, lag=lag)
        # With 1 obs per time, aggregated scores = raw scores → should match hac_meat_loop
        weights = _get_bartlett_weights(lag=lag)
        expected = _hac_meat_loop(
            scores=scores, weights=weights, time_periods=T, k=k, lag=lag
        )
        np.testing.assert_allclose(meat_dk, expected, atol=1e-10)

    def test_symmetry(self):
        rng = np.random.default_rng(77)
        # 5 units, 10 time periods
        n_units, n_periods = 5, 10
        n = n_units * n_periods
        scores = rng.standard_normal((n, 3))
        # Sort by (time, unit) as DK expects
        time_arr = np.repeat(np.arange(n_periods), n_units).astype(float)
        idx = np.arange(0, n, n_units)  # start of each time period
        lag = 2

        meat = _dk_meat_panel(scores=scores, time_arr=time_arr, idx=idx, lag=lag)
        np.testing.assert_allclose(meat, meat.T, atol=1e-12)

    def test_default_lag(self):
        """When lag=None, should default to floor(n_unique_times ** 0.25)."""
        rng = np.random.default_rng(11)
        n_units, n_periods = 3, 16
        n = n_units * n_periods
        scores = rng.standard_normal((n, 2))
        time_arr = np.repeat(np.arange(n_periods), n_units).astype(float)
        idx = np.arange(0, n, n_units)

        meat_default = _dk_meat_panel(
            scores=scores, time_arr=time_arr, idx=idx, lag=None
        )
        # Default: floor(16 ** 0.25) = 2
        meat_explicit = _dk_meat_panel(scores=scores, time_arr=time_arr, idx=idx, lag=2)
        np.testing.assert_allclose(meat_default, meat_explicit, atol=1e-12)

    def test_aggregation_manual(self):
        """Verify the time-aggregation step manually."""
        # 2 units, 3 time periods, sorted by time
        # t=0: obs 0,1  t=1: obs 2,3  t=2: obs 4,5
        scores = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        )
        time_arr = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        idx = np.array([0, 2, 4])
        lag = 0

        meat = _dk_meat_panel(scores=scores, time_arr=time_arr, idx=idx, lag=lag)

        # Aggregated scores by time:
        # t=0: [1+3, 2+4] = [4, 6]
        # t=1: [5+7, 6+8] = [12, 14]
        # t=2: [9+11, 10+12] = [20, 22]
        scores_agg = np.array([[4.0, 6.0], [12.0, 14.0], [20.0, 22.0]])
        expected = scores_agg.T @ scores_agg
        np.testing.assert_allclose(meat, expected, atol=1e-10)
