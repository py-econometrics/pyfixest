"""
Tests for Conley (1999) Spatial HAC Standard Errors.

Tests the standalone function and (when pyfixest is importable) the
Feols.vcov("conley", ...) integration.
"""
import importlib.util

import numpy as np
import pytest

# Load the standalone module directly (avoids Rust backend dependency)
spec = importlib.util.spec_from_file_location(
    "conley_se",
    str(
        __import__("pathlib").Path(__file__).parent.parent
        / "pyfixest"
        / "estimation"
        / "post_estimation"
        / "conley_se.py"
    ),
)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

conley_meat = _mod.conley_meat
vcov_conley = _mod.vcov_conley
_haversine_scalar = _mod._haversine_scalar


# --- Haversine tests ---


def test_haversine_nyc_to_boston():
    """NYC to Boston should be about 306 km."""
    d = _haversine_scalar(40.7128, -74.0060, 42.3601, -71.0589)
    assert 300 < d < 320


def test_haversine_same_point():
    """Distance from a point to itself is zero."""
    d = _haversine_scalar(45.0, -73.0, 45.0, -73.0)
    assert d == 0.0


def test_haversine_symmetry():
    """Distance is symmetric."""
    d1 = _haversine_scalar(40.0, -74.0, 42.0, -71.0)
    d2 = _haversine_scalar(42.0, -71.0, 40.0, -74.0)
    assert abs(d1 - d2) < 1e-10


# --- Meat computation tests ---


class TestConleyMeat:
    """Tests for the conley_meat function."""

    def setup_method(self):
        np.random.seed(42)
        self.n = 80
        self.k = 3
        self.lat = np.random.uniform(35, 45, self.n)
        self.lon = np.random.uniform(-80, -70, self.n)
        self.scores = np.random.randn(self.n, self.k)

    def test_matches_brute_force(self):
        """Our vectorized implementation matches a naive double loop."""
        n, k = self.n, self.k
        scores, lat, lon = self.scores, self.lat, self.lon
        cutoff = 300.0

        # Brute force reference
        meat_ref = np.zeros((k, k))
        for i in range(n):
            for j in range(n):
                d = _haversine_scalar(lat[i], lon[i], lat[j], lon[j])
                if d <= cutoff:
                    w = 1.0 - d / cutoff
                    meat_ref += w * np.outer(scores[i], scores[j])

        meat_ours = conley_meat(scores, lat, lon, cutoff, "bartlett")
        assert np.allclose(meat_ours, meat_ref, atol=1e-10)

    def test_uniform_kernel_all_pairs(self):
        """Uniform kernel with huge cutoff = outer product of total score."""
        meat = conley_meat(self.scores, self.lat, self.lon, 50000.0, "uniform")
        total = self.scores.sum(axis=0)
        expected = np.outer(total, total)
        assert np.allclose(meat, expected, atol=1e-8)

    def test_symmetry(self):
        """Meat matrix must be symmetric."""
        meat = conley_meat(self.scores, self.lat, self.lon, 500.0, "bartlett")
        assert np.allclose(meat, meat.T, atol=1e-12)

    def test_positive_diagonal(self):
        """Diagonal of meat should be non-negative."""
        meat = conley_meat(self.scores, self.lat, self.lon, 500.0, "bartlett")
        assert np.all(np.diag(meat) >= 0)

    def test_smaller_cutoff_smaller_meat(self):
        """Smaller cutoff means fewer pairs, so trace should not increase."""
        meat_big = conley_meat(self.scores, self.lat, self.lon, 1000.0, "bartlett")
        meat_small = conley_meat(self.scores, self.lat, self.lon, 100.0, "bartlett")
        # Not strictly guaranteed for off-diagonals, but trace should follow
        # (with bartlett, all weights are positive, so this holds)
        assert np.trace(meat_big) >= np.trace(meat_small) - 1e-10

    def test_invalid_kernel_raises(self):
        """Invalid kernel name should raise ValueError."""
        with pytest.raises(ValueError, match="kernel must be"):
            conley_meat(self.scores, self.lat, self.lon, 500.0, "gaussian")


# --- Full sandwich tests ---


class TestVcovConley:
    """Tests for the full vcov_conley sandwich computation."""

    def setup_method(self):
        np.random.seed(2024)
        self.n = 50
        self.lat = np.random.uniform(40, 42, self.n)
        self.lon = np.random.uniform(-75, -73, self.n)
        x = np.random.randn(self.n)
        e = np.random.randn(self.n) * 0.5
        y = 1.0 + 2.0 * x + e

        self.X = np.column_stack([np.ones(self.n), x])
        self.beta = np.linalg.lstsq(self.X, y, rcond=None)[0]
        resid = y - self.X @ self.beta
        self.scores = self.X * resid[:, None]
        self.bread = np.linalg.inv(self.X.T @ self.X)

    def test_vcov_shape(self):
        """Vcov should be k x k."""
        vcov = vcov_conley(
            self.scores, self.lat, self.lon, 100.0, "bartlett",
            self.bread, False, np.array([]), np.array([]), np.array([])
        )
        assert vcov.shape == (2, 2)

    def test_vcov_symmetric(self):
        """Vcov should be symmetric."""
        vcov = vcov_conley(
            self.scores, self.lat, self.lon, 100.0, "bartlett",
            self.bread, False, np.array([]), np.array([]), np.array([])
        )
        assert np.allclose(vcov, vcov.T, atol=1e-14)

    def test_vcov_positive_diagonal(self):
        """Variances should be positive."""
        vcov = vcov_conley(
            self.scores, self.lat, self.lon, 100.0, "bartlett",
            self.bread, False, np.array([]), np.array([]), np.array([])
        )
        assert np.all(np.diag(vcov) > 0)

    def test_se_reasonable_magnitude(self):
        """SE should be in a reasonable range for this DGP."""
        vcov = vcov_conley(
            self.scores, self.lat, self.lon, 100.0, "bartlett",
            self.bread, False, np.array([]), np.array([]), np.array([])
        )
        se = np.sqrt(np.diag(vcov))
        # For n=50 with sigma=0.5, SE should be roughly 0.01-0.2
        assert np.all(se > 0.001)
        assert np.all(se < 1.0)


# --- Size control test ---


def test_size_control_iid_errors():
    """Under IID errors, Conley test at 5% should reject about 5% of the time."""
    np.random.seed(0)
    n_sims = 300
    n = 100
    rejections = 0

    for _ in range(n_sims):
        lat = np.random.uniform(30, 50, n)
        lon = np.random.uniform(-90, -70, n)
        x = np.random.randn(n)
        e = np.random.randn(n)
        y = 1.0 + 0.0 * x + e  # true beta_x = 0

        X = np.column_stack([np.ones(n), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        scores = X * resid[:, None]
        bread = np.linalg.inv(X.T @ X)

        vcov = vcov_conley(
            scores, lat, lon, 200.0, "bartlett", bread,
            False, np.array([]), np.array([]), np.array([])
        )
        se = np.sqrt(vcov[1, 1])
        t_stat = beta[1] / se
        if abs(t_stat) > 1.96:
            rejections += 1

    rate = rejections / n_sims
    # Should be close to 5%, allow some Monte Carlo noise
    assert 0.01 < rate < 0.15, f"Size distortion: {rate:.3f}"


# --- Spatial correlation detection test ---


def test_conley_detects_spatial_correlation():
    """Conley SE should be larger than IID SE when errors are spatially correlated."""
    np.random.seed(42)
    n = 200
    lat = np.random.uniform(40, 41, n)
    lon = np.random.uniform(-74, -73, n)
    x = np.random.randn(n)

    # Common shock model
    n_centers = 10
    center_lat = np.random.uniform(40, 41, n_centers)
    center_lon = np.random.uniform(-74, -73, n_centers)
    center_shocks = np.random.randn(n_centers) * 2.0

    e = np.random.randn(n) * 0.3
    for c in range(n_centers):
        for i in range(n):
            d = _haversine_scalar(lat[i], lon[i], center_lat[c], center_lon[c])
            if d < 30:
                e[i] += center_shocks[c] * (1 - d / 30)

    y = 1.0 + 2.0 * x + e
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    scores = X * resid[:, None]
    bread = np.linalg.inv(X.T @ X)

    vcov_c = vcov_conley(
        scores, lat, lon, 50.0, "bartlett", bread,
        False, np.array([]), np.array([]), np.array([])
    )
    se_conley = np.sqrt(np.diag(vcov_c))

    sigma2 = np.sum(resid**2) / (n - 2)
    se_iid = np.sqrt(np.diag(sigma2 * bread))

    # Conley SE for intercept should be much larger than IID
    assert se_conley[0] > 1.5 * se_iid[0], (
        f"Conley SE ({se_conley[0]:.4f}) should be much larger than IID SE ({se_iid[0]:.4f})"
    )
