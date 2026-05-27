"""Tests for Conley standard errors against R fixest."""

import sys

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf

if " " in sys.prefix:
    pytest.skip(
        "rpy2/R tests are skipped when the Python environment path contains spaces.",
        allow_module_level=True,
    )

rpy2 = pytest.importorskip("rpy2")
import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects.packages import importr  # noqa: E402

fixest = importr("fixest")
stats = importr("stats")


@pytest.fixture(scope="module")
def data_conley():
    rng = np.random.default_rng(9791)
    n_units = 80
    n_periods = 4
    n = n_units * n_periods
    unit = np.repeat(np.arange(n_units), n_periods)
    year = np.tile(np.arange(n_periods), n_units)
    lat_unit = rng.uniform(-25, 25, n_units)
    lon_unit = rng.uniform(-80, -30, n_units)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    alpha = rng.normal(scale=0.5, size=n_units)
    gamma = rng.normal(scale=0.25, size=n_periods)
    y = 1 + x1 + x2 + alpha[unit] + gamma[year] + rng.normal(size=n)

    return pd.DataFrame(
        {
            "Y": y,
            "X1": x1,
            "X2": x2,
            "unit": unit,
            "year": year,
            "lat": lat_unit[unit],
            "lon": lon_unit[unit],
        }
    )


@pytest.mark.hac
@pytest.mark.parametrize("distance", ["triangular", "spherical"])
@pytest.mark.parametrize("cutoff", [0, 250, 750])
@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ X1",
        "Y ~ X1 + X2",
        "Y ~ X1 | unit",
        "Y ~ X1 + X2 | unit",
        "Y ~ X1 | unit + year",
        "Y ~ X1 + X2 | unit + year",
    ],
)
def test_feols_conley_vs_fixest(data_conley, fml, cutoff, distance):
    k_adj = True
    G_adj = True
    k_fixef = "none"
    vcov_kwargs = {
        "lat": "lat",
        "lon": "lon",
        "cutoff": cutoff,
        "distance": distance,
    }

    r_fit = fixest.feols(
        ro.Formula(fml),
        data=data_conley,
        vcov=fixest.vcov_conley(
            lat="lat",
            lon="lon",
            cutoff=cutoff,
            distance=distance,
            vcov_fix=False,
        ),
        ssc=fixest.ssc(k_adj, k_fixef, False, G_adj, "min", "min"),
    )

    py_fit = pf.feols(
        fml=fml,
        data=data_conley,
        vcov="conley",
        vcov_kwargs=vcov_kwargs,
        ssc=pf.ssc(k_adj=k_adj, k_fixef=k_fixef, G_adj=G_adj),
    )

    ro.globalenv["r_fit"] = r_fit
    py_vcov = py_fit._vcov
    r_vcov = np.asarray(stats.vcov(r_fit))

    np.testing.assert_allclose(py_vcov, r_vcov, rtol=1e-5, atol=1e-5)


@pytest.mark.hac
@pytest.mark.parametrize("distance", ["triangular", "spherical"])
def test_feols_conley_latitude_0_180_vs_fixest(data_conley, distance):
    data_0_180 = data_conley.copy()
    data_0_180["lat_0_180"] = data_0_180["lat"] + 90
    fml = "Y ~ X1 + X2 | unit + year"
    cutoff = 250
    k_adj = True
    G_adj = True
    k_fixef = "none"

    r_fit = fixest.feols(
        ro.Formula(fml),
        data=data_0_180,
        vcov=fixest.vcov_conley(
            lat="lat_0_180",
            lon="lon",
            cutoff=cutoff,
            distance=distance,
            vcov_fix=False,
        ),
        ssc=fixest.ssc(k_adj, k_fixef, False, G_adj, "min", "min"),
    )

    py_fit = pf.feols(
        fml=fml,
        data=data_0_180,
        vcov="conley",
        vcov_kwargs={
            "lat": "lat_0_180",
            "lon": "lon",
            "cutoff": cutoff,
            "distance": distance,
        },
        ssc=pf.ssc(k_adj=k_adj, k_fixef=k_fixef, G_adj=G_adj),
    )

    r_vcov = np.asarray(stats.vcov(r_fit))
    np.testing.assert_allclose(py_fit._vcov, r_vcov, rtol=1e-5, atol=1e-5)
