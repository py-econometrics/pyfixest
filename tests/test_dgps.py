import numpy as np
import pandas as pd

from pyfixest.utils.dgps import (
    gelbach_data,
    get_blw,
    get_panel_dgp_stagg,
    get_sharkfin,
)


def test_get_blw():
    """Test Baker, Larcker, and Wang (2022) DGP."""
    # Get data
    df = get_blw()

    # Test basic properties
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 30_000  # 30 years * 1000 ids
    assert set(df.columns) == {
        "n",
        "id",
        "year",
        "state",
        "group",
        "treat_date",
        "time_til",
        "treat",
        "firms",
        "e",
        "te",
        "y",
        "y2",
    }

    # Test value ranges
    assert df["n"].min() == 1 and df["n"].max() == 30
    assert df["id"].min() == 1 and df["id"].max() == 1000
    assert df["year"].min() == 1980 and df["year"].max() == 2009
    assert df["state"].min() == 1 and df["state"].max() == 40
    assert df["group"].min() == 1 and df["group"].max() == 4

    # Test treatment assignment
    assert df["treat"].dtype == bool
    assert (df["treat"] == (df["time_til"] >= 0)).all()


def test_get_sharkfin():
    """Test sharkfin DGP with various parameter combinations."""
    # Test default parameters - num_periods=30, treatment_start=15
    base_effect = np.zeros(15)  # 30 - 15 periods after treatment
    effect_periods = min(8, 15)
    base_effect[:effect_periods] = 0.2 * np.log(2 * np.arange(1, effect_periods + 1))
    df = get_sharkfin(base_treatment_effect=base_effect)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1000 * 30  # default num_units * num_periods
    assert set(df.columns) == {"unit", "year", "treat", "Y", "ever_treated"}

    # Test with heterogeneous effects and custom dimensions
    num_periods = 20
    treatment_start = 10
    post_treatment_periods = num_periods - treatment_start
    # Create treatment effect vector matching the exact shape needed
    base_effect_het = np.zeros(post_treatment_periods)
    effect_periods = min(8, post_treatment_periods)
    base_effect_het[:effect_periods] = 0.2 * np.log(
        2 * np.arange(1, effect_periods + 1)
    )

    df_het = get_sharkfin(
        hetfx=True,
        num_units=500,
        num_periods=num_periods,
        treatment_start=treatment_start,
        base_treatment_effect=base_effect_het,
    )
    assert len(df_het) == 500 * num_periods
    assert df_het["treat"].sum() > 0

    # Test return type when return_dataframe=False
    post_treat = 15  # 30 - 15 periods after treatment
    base_effect_dict = np.zeros(post_treat)
    effect_periods = min(8, post_treat)
    base_effect_dict[:effect_periods] = 0.2 * np.log(
        2 * np.arange(1, effect_periods + 1)
    )

    result_dict = get_sharkfin(
        num_periods=30,
        treatment_start=15,
        base_treatment_effect=base_effect_dict,
        return_dataframe=False,
    )
    assert isinstance(result_dict, dict)
    assert set(result_dict.keys()) == {
        "Y1",
        "Y0",
        "W",
        "unit_intercepts",
        "time_intercepts",
    }

    # Test treatment timing
    post_treat_timing = 20  # 30 - 10 periods after treatment
    base_effect_timing = np.zeros(post_treat_timing)
    effect_periods = min(8, post_treat_timing)
    base_effect_timing[:effect_periods] = 0.2 * np.log(
        2 * np.arange(1, effect_periods + 1)
    )

    df = get_sharkfin(
        num_periods=30, treatment_start=10, base_treatment_effect=base_effect_timing
    )
    assert df[df["year"] < 10]["treat"].sum() == 0
    assert df[df["year"] >= 10]["treat"].sum() > 0


def test_get_panel_dgp_stagg():
    """Test staggered treatment DGP."""
    # Test default parameters
    result = get_panel_dgp_stagg()
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {
        "unit_id",
        "time_id",
        "W_it",
        "Y_it",
        "unit_intercept",
        "time_intercept",
    }

    # Test with custom treatment cohorts
    custom_starts = [5, 10, 15]
    custom_treated = [100, 200, 300]
    result = get_panel_dgp_stagg(
        treatment_start_cohorts=custom_starts,
        num_treated=custom_treated,
        num_units=1000,
        num_periods=20,
    )
    df = result["dataframe"]
    assert len(df) == 1000 * 20

    # Test return dictionary format
    assert set(result.keys()) == {
        "Y1",
        "Y0",
        "W",
        "unit_intercepts",
        "time_intercepts",
        "dataframe",
    }

    # Test with heterogeneous effects
    result_het = get_panel_dgp_stagg(hetfx=True)
    assert isinstance(result_het["dataframe"], pd.DataFrame)


def test_gelbach_data():
    """Test Gelbach decomposition data generation."""
    # Test with different sample sizes
    for n in [100, 1000]:
        df = gelbach_data(n)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == n
        assert set(df.columns) == {"x1", "x21", "x22", "x23", "y"}

        # Test basic statistical properties
        assert abs(df["x1"].mean()) < 0.5  # Should be roughly centered at 0
        assert 0.9 < df["x1"].std() < 1.1  # Should be roughly standard normal

        # Test correlations exist between variables
        assert df["x1"].corr(df["x21"]) > 0
        assert df["x21"].corr(df["x22"]) > 0
        assert df["y"].corr(df["x1"]) != 0
