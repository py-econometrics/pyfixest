"""Unit tests for Conley spatial HAC meat matrix functions."""

import numpy as np
import pytest

import pyfixest as pf
from pyfixest.estimation.internals.vcov_utils import _conley_meat


@pytest.mark.hac
class TestConleyMeat:
    def test_symmetry(self):
        rng = np.random.default_rng(123)
        scores = rng.standard_normal((50, 4))
        lat = rng.uniform(-30, 30, 50)
        lon = rng.uniform(-80, -40, 50)

        meat = _conley_meat(
            scores=scores, lon_arr=lon, lat_arr=lat, cutoff=500, distance="triangular"
        )

        np.testing.assert_allclose(meat, meat.T, atol=1e-12)

    def test_sorted_and_unsorted_inputs_match(self):
        rng = np.random.default_rng(456)
        scores = rng.standard_normal((40, 3))
        lat = rng.uniform(-20, 20, 40)
        lon = rng.uniform(10, 60, 40)

        order = np.argsort(lat)
        meat_unsorted = _conley_meat(
            scores=scores, lon_arr=lon, lat_arr=lat, cutoff=300, distance="triangular"
        )
        meat_sorted = _conley_meat(
            scores=scores[order],
            lon_arr=lon[order],
            lat_arr=lat[order],
            cutoff=300,
            distance="triangular",
        )

        np.testing.assert_allclose(meat_unsorted, meat_sorted, atol=1e-12)

    def test_longitude_wraparound(self):
        scores = np.array([[1.0], [2.0]])
        lat = np.array([0.0, 0.0])
        lon = np.array([179.9, -179.9])

        meat = _conley_meat(
            scores=scores, lon_arr=lon, lat_arr=lat, cutoff=50, distance="triangular"
        )

        np.testing.assert_allclose(meat, np.array([[9.0]]), atol=1e-12)

    def test_spherical_keeps_high_latitude_neighbors(self):
        scores = np.array([[1.0], [2.0]])
        lat = np.array([80.0, 80.0])
        lon = np.array([0.0, 120.0])

        meat = _conley_meat(
            scores=scores, lon_arr=lon, lat_arr=lat, cutoff=2000, distance="spherical"
        )

        np.testing.assert_allclose(meat, np.array([[9.0]]), atol=1e-12)

    def test_latitude_0_180_encoding_matches_normalized_coordinates(self):
        scores = np.array([[1.0], [2.0]])
        lon = np.array([0.0, 1.0])

        meat_0_180 = _conley_meat(
            scores=scores,
            lon_arr=lon,
            lat_arr=np.array([100.0, 100.0]),
            cutoff=50,
            distance="triangular",
        )
        meat_normalized = _conley_meat(
            scores=scores,
            lon_arr=lon,
            lat_arr=np.array([10.0, 10.0]),
            cutoff=50,
            distance="triangular",
        )

        np.testing.assert_allclose(meat_0_180, meat_normalized, atol=1e-12)

    def test_cutoff_excludes_far_observations(self):
        scores = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        lat = np.array([0.0, 20.0, 40.0])
        lon = np.array([0.0, 20.0, 40.0])

        meat = _conley_meat(
            scores=scores, lon_arr=lon, lat_arr=lat, cutoff=1, distance="triangular"
        )

        np.testing.assert_allclose(meat, scores.T @ scores, atol=1e-12)

    def test_zero_cutoff_unique_coordinates_equals_hc0_meat(self):
        scores = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        lat = np.array([0.0, 20.0, 40.0])
        lon = np.array([0.0, 20.0, 40.0])

        meat = _conley_meat(
            scores=scores, lon_arr=lon, lat_arr=lat, cutoff=0, distance="triangular"
        )

        np.testing.assert_allclose(meat, scores.T @ scores, atol=1e-12)

    def test_zero_cutoff_groups_identical_coordinates(self):
        scores = np.array([[1.0], [2.0], [3.0]])
        lat = np.array([0.0, 0.0, 1.0])
        lon = np.array([0.0, 0.0, 1.0])

        meat = _conley_meat(
            scores=scores, lon_arr=lon, lat_arr=lat, cutoff=0, distance="triangular"
        )

        np.testing.assert_allclose(meat, np.array([[18.0]]), atol=1e-12)

    def test_huge_cutoff_includes_all_observations(self):
        scores = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        lat = np.array([0.0, 20.0, 40.0])
        lon = np.array([0.0, 20.0, 40.0])

        meat = _conley_meat(
            scores=scores,
            lon_arr=lon,
            lat_arr=lat,
            cutoff=20_000,
            distance="triangular",
        )
        score_sum = scores.sum(axis=0)

        np.testing.assert_allclose(meat, np.outer(score_sum, score_sum), atol=1e-12)

    @pytest.mark.parametrize("distance", ["triangular", "spherical"])
    def test_distance_modes(self, distance):
        scores = np.array([[1.0, 0.5], [2.0, -1.0], [0.5, 1.5]])
        lat = np.array([0.0, 0.1, 1.0])
        lon = np.array([0.0, 0.1, 1.0])

        meat = _conley_meat(
            scores=scores, lon_arr=lon, lat_arr=lat, cutoff=100, distance=distance
        )

        assert meat.shape == (2, 2)
        assert np.isfinite(meat).all()

    def test_grouped_coordinates_match_ungrouped_meat(self):
        rng = np.random.default_rng(321)
        scores = rng.standard_normal((12, 3))
        lat = np.repeat(np.array([0.0, 1.0, 2.0, 3.0]), 3)
        lon = np.repeat(np.array([10.0, 11.0, 12.0, 13.0]), 3)

        meat_grouped = _conley_meat(
            scores=scores,
            lon_arr=lon,
            lat_arr=lat,
            cutoff=200,
            distance="triangular",
        )
        meat_ungrouped = _conley_meat(
            scores=scores,
            lon_arr=lon,
            lat_arr=lat,
            cutoff=200,
            distance="triangular",
            aggregate=False,
        )

        np.testing.assert_allclose(meat_grouped, meat_ungrouped, atol=1e-12)


@pytest.mark.hac
def test_vcov_updating_conley():
    data = pf.get_data()
    data["lat"] = np.linspace(-20, 20, data.shape[0])
    data["lon"] = np.linspace(-60, -20, data.shape[0])
    vcov_kwargs = {
        "lat": "lat",
        "lon": "lon",
        "cutoff": 250,
        "distance": "triangular",
    }

    fit_hetero = pf.feols("Y ~ X1", data=data, vcov="hetero")
    fit_conley = pf.feols("Y ~ X1", data=data, vcov="conley", vcov_kwargs=vcov_kwargs)

    fit_hetero.vcov(vcov="conley", vcov_kwargs=vcov_kwargs)

    assert fit_hetero._vcov_type == "conley"
    assert fit_hetero._vcov_type_detail == "conley"
    np.testing.assert_allclose(fit_hetero._vcov, fit_conley._vcov, atol=1e-12)


@pytest.mark.hac
def test_vcov_updating_conley_validates_kwargs():
    data = pf.get_data()
    data["lat"] = np.linspace(-20, 20, data.shape[0])
    data["lon"] = np.linspace(-60, -20, data.shape[0])
    fit = pf.feols("Y ~ X1", data=data, vcov="hetero")

    with pytest.raises(ValueError, match="must contain 'lat', 'lon', and 'cutoff'"):
        fit.vcov(vcov="conley", vcov_kwargs={"lat": "lat"})

    with pytest.raises(ValueError, match="non-negative finite value for 'cutoff'"):
        fit.vcov(
            vcov="conley",
            vcov_kwargs={"lat": "lat", "lon": "lon", "cutoff": -1},
        )

    with pytest.raises(ValueError, match="either 'triangular' or 'spherical'"):
        fit.vcov(
            vcov="conley",
            vcov_kwargs={
                "lat": "lat",
                "lon": "lon",
                "cutoff": 250,
                "distance": "euclidean",
            },
        )


@pytest.mark.hac
class TestConleyValidationErrors:
    def test_missing_vcov_kwargs(self):
        data = pf.get_data()
        with pytest.raises(ValueError, match="Missing required vcov_kwargs for Conley"):
            pf.feols("Y ~ X1", data=data, vcov="conley")

    def test_missing_required_keys(self):
        data = pf.get_data()
        with pytest.raises(ValueError, match="must contain 'lat', 'lon', and 'cutoff'"):
            pf.feols("Y ~ X1", data=data, vcov="conley", vcov_kwargs={"lat": "lat", "lon": "lon"})

    def test_invalid_key_type(self):
        data = pf.get_data()
        with pytest.raises(TypeError, match="must be a dictionary with string values for 'lat'"):
            pf.feols("Y ~ X1", data=data, vcov="conley", vcov_kwargs={"lat": 123, "lon": "lon", "cutoff": 100})

    def test_variable_not_in_data(self):
        data = pf.get_data()
        with pytest.raises(ValueError, match="is not in the data"):
            pf.feols("Y ~ X1", data=data, vcov="conley", vcov_kwargs={"lat": "nonexistent_lat", "lon": "lon", "cutoff": 100})

    def test_invalid_cutoff(self):
        data = pf.get_data()
        data["lat"] = np.linspace(-20, 20, data.shape[0])
        data["lon"] = np.linspace(-60, -20, data.shape[0])
        with pytest.raises(TypeError, match="must be a dictionary with a numeric value for 'cutoff'"):
            pf.feols("Y ~ X1", data=data, vcov="conley", vcov_kwargs={"lat": "lat", "lon": "lon", "cutoff": "invalid"})
        with pytest.raises(ValueError, match="must contain a non-negative finite value for 'cutoff'"):
            pf.feols("Y ~ X1", data=data, vcov="conley", vcov_kwargs={"lat": "lat", "lon": "lon", "cutoff": -50})

    def test_invalid_distance(self):
        data = pf.get_data()
        data["lat"] = np.linspace(-20, 20, data.shape[0])
        data["lon"] = np.linspace(-60, -20, data.shape[0])
        with pytest.raises(ValueError, match="The Conley distance must be either 'triangular' or 'spherical'"):
            pf.feols("Y ~ X1", data=data, vcov="conley", vcov_kwargs={"lat": "lat", "lon": "lon", "cutoff": 100, "distance": "euclidean"})

    def test_non_numeric_coordinates(self):
        data = pf.get_data()
        data["lat"] = "not_numeric"
        data["lon"] = np.linspace(-60, -20, data.shape[0])
        with pytest.raises(ValueError, match="The latitude variable must be numeric"):
            pf.feols("Y ~ X1", data=data, vcov="conley", vcov_kwargs={"lat": "lat", "lon": "lon", "cutoff": 100})

    def test_nan_in_coordinates(self):
        data = pf.get_data()
        data["lat"] = np.linspace(-20, 20, data.shape[0])
        data["lon"] = np.linspace(-60, -20, data.shape[0])
        idx = data[["Y", "X1"]].dropna().index[0]
        data.loc[idx, "lat"] = np.nan
        with pytest.raises(ValueError, match="Conley inference is not supported with missing values"):
            pf.feols("Y ~ X1", data=data, vcov="conley", vcov_kwargs={"lat": "lat", "lon": "lon", "cutoff": 100})

    def test_nan_in_dropped_model_row_is_ignored(self):
        data = pf.get_data()
        data["lat"] = np.linspace(-20, 20, data.shape[0])
        data["lon"] = np.linspace(-60, -20, data.shape[0])
        data.loc[0, "Y"] = np.nan
        data.loc[0, "lat"] = np.nan

        fit = pf.feols(
            "Y ~ X1",
            data=data,
            vcov="conley",
            vcov_kwargs={"lat": "lat", "lon": "lon", "cutoff": 100},
        )

        assert np.isfinite(fit._vcov).all()

    def test_nonfinite_coordinates_error_after_sample_trimming(self):
        data = pf.get_data()
        data["lat"] = np.linspace(-20, 20, data.shape[0])
        data["lon"] = np.linspace(-60, -20, data.shape[0])
        idx = data[["Y", "X1"]].dropna().index[0]
        data.loc[idx, "lat"] = np.inf

        with pytest.raises(ValueError, match="non-finite coordinate values"):
            pf.feols(
                "Y ~ X1",
                data=data,
                vcov="conley",
                vcov_kwargs={"lat": "lat", "lon": "lon", "cutoff": 100},
            )

    def test_fepois_not_implemented(self):
        data = pf.get_data().copy()
        data["Y"] = np.abs(data["Y"])
        data["lat"] = np.linspace(-20, 20, data.shape[0])
        data["lon"] = np.linspace(-60, -20, data.shape[0])
        with pytest.raises(NotImplementedError, match="Conley inference is currently only supported for feols models"):
            pf.fepois("Y ~ X1", data=data, vcov="conley", vcov_kwargs={"lat": "lat", "lon": "lon", "cutoff": 100})

    def test_feiv_not_implemented(self):
        data = pf.get_data()
        data["lat"] = np.linspace(-20, 20, data.shape[0])
        data["lon"] = np.linspace(-60, -20, data.shape[0])
        with pytest.raises(NotImplementedError, match="Conley inference is currently only supported for feols models"):
            pf.feols("Y ~ X1 | X2 ~ Z1", data=data, vcov="conley", vcov_kwargs={"lat": "lat", "lon": "lon", "cutoff": 100})


@pytest.mark.hac
def test_conley_meat_2d_coordinate_squeezing():
    rng = np.random.default_rng(789)
    scores = rng.standard_normal((30, 2))
    lat = rng.uniform(-10, 10, 30)
    lon = rng.uniform(-30, 30, 30)

    # 1D input
    meat_1d = _conley_meat(
        scores=scores, lon_arr=lon, lat_arr=lat, cutoff=500, distance="triangular"
    )

    # 2D input with shape (N, 1)
    meat_2d = _conley_meat(
        scores=scores, lon_arr=lon[:, None], lat_arr=lat[:, None], cutoff=500, distance="triangular"
    )

    np.testing.assert_allclose(meat_1d, meat_2d, atol=1e-12)
