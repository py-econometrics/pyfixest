from dataclasses import asdict
from pathlib import Path

from benchmarks.modular.akm_dgp import (
    AKMConfig,
    simulate_akm_panel,
    summarize_akm_panel,
)
from benchmarks.modular.dgps import (
    get_akm_occupation_scenarios,
    get_akm_sweep_scenarios,
)


def test_get_akm_sweep_scenarios():
    names = [
        "akm_baseline",
        "akm_scale_2",
        "akm_sorting_2",
        "akm_mobility_3",
        "akm_freeze_4",
    ]
    scenarios = get_akm_sweep_scenarios(Path("unused"), names)

    assert [scenario.dgp_name for scenario in scenarios] == names


def test_get_akm_occupation_scenarios():
    names = [
        "akm_baseline",
        "akm_occlambda_4",
        "akm_occsize_5",
    ]
    scenarios = get_akm_occupation_scenarios(Path("unused"), names)

    assert [scenario.dgp_name for scenario in scenarios] == names


def test_non_pathological_occupation_scenarios_vary_one_parameter():
    baseline = get_akm_occupation_scenarios(Path("unused"), ["akm_baseline"])[0]
    baseline_config = asdict(baseline._build_config())

    expectations = {
        "akm_occlambda_4": {"occ_lambda"},
        "akm_occsize_5": {"n_occupations"},
    }
    scenarios = get_akm_occupation_scenarios(Path("unused"), list(expectations))

    for scenario in scenarios:
        config = asdict(scenario._build_config())
        changed = {
            key for key, value in config.items() if value != baseline_config[key]
        }
        assert changed == expectations[scenario.dgp_name]


def test_simulate_akm_panel_returns_required_columns():
    df = simulate_akm_panel(
        AKMConfig(n_workers=50, n_firms=10, n_time=4),
        seed=7,
    )

    assert list(df.columns) == ["indiv_id", "firm_id", "year", "x1", "y"]
    assert len(df) == 200


def test_simulate_akm_panel_with_occupation_returns_occ_column():
    df = simulate_akm_panel(
        AKMConfig(
            n_workers=50,
            n_firms=10,
            n_time=4,
            n_occupations=12,
            occ_menu_size=3,
        ),
        seed=7,
    )

    assert list(df.columns) == ["indiv_id", "firm_id", "year", "x1", "y", "occ_id"]
    assert len(df) == 200


def test_high_sorting_increases_worker_firm_fe_alignment():
    common = {
        "n_workers": 400,
        "n_firms": 40,
        "n_time": 8,
        "n_industries": 3,
        "delta": 0.2,
        "lambda_": 0.8,
    }
    low_sort_df = simulate_akm_panel(
        AKMConfig(**common, rho=0.0),
        seed=11,
        include_latent=True,
    )
    high_sort_df = simulate_akm_panel(
        AKMConfig(**common, rho=10.0),
        seed=11,
        include_latent=True,
    )

    low_corr = low_sort_df["worker_fe"].corr(low_sort_df["firm_fe"])
    high_corr = high_sort_df["worker_fe"].corr(high_sort_df["firm_fe"])

    assert high_corr > low_corr


def test_lower_delta_reduces_mover_share():
    common = {
        "n_workers": 500,
        "n_firms": 60,
        "n_time": 10,
        "n_industries": 4,
        "rho": 1.0,
        "lambda_": 0.8,
    }
    high_move_df = simulate_akm_panel(AKMConfig(**common, delta=0.5), seed=13)
    low_move_df = simulate_akm_panel(AKMConfig(**common, delta=0.05), seed=13)

    high_move_share = summarize_akm_panel(high_move_df)["mover_share"]
    low_move_share = summarize_akm_panel(low_move_df)["mover_share"]

    assert high_move_share > low_move_share


def test_lower_gamma_increases_firm_size_concentration():
    common = {
        "n_workers": 600,
        "n_firms": 50,
        "n_time": 8,
        "n_industries": 3,
        "delta": 0.2,
        "rho": 1.0,
        "lambda_": 0.8,
    }
    near_uniform_df = simulate_akm_panel(AKMConfig(**common, gamma=100.0), seed=17)
    concentrated_df = simulate_akm_panel(AKMConfig(**common, gamma=0.5), seed=17)

    near_uniform_max = near_uniform_df.groupby("firm_id").size().max()
    concentrated_max = concentrated_df.groupby("firm_id").size().max()

    assert concentrated_max > near_uniform_max


def test_higher_lambda_reduces_cross_industry_histories():
    common = {
        "n_workers": 500,
        "n_firms": 50,
        "n_time": 10,
        "n_industries": 5,
        "delta": 0.2,
        "rho": 1.0,
    }
    weak_lock_df = simulate_akm_panel(
        AKMConfig(**common, lambda_=0.4),
        seed=19,
        include_latent=True,
    )
    strong_lock_df = simulate_akm_panel(
        AKMConfig(**common, lambda_=0.95),
        seed=19,
        include_latent=True,
    )

    weak_cross = summarize_akm_panel(weak_lock_df)["cross_industry_share"]
    strong_cross = summarize_akm_panel(strong_lock_df)["cross_industry_share"]

    assert strong_cross < weak_cross


def test_unbalanced_panels_have_fewer_rows_and_keep_at_least_two_periods():
    config = AKMConfig(
        n_workers=300,
        n_firms=60,
        n_time=10,
        n_industries=4,
        delta=0.2,
        rho=1.0,
        lambda_=0.8,
        entry_exit_share=0.75,
        entry_exit_n_periods=2,
    )

    df = simulate_akm_panel(config, seed=29)
    obs_per_worker = df.groupby("indiv_id").size()

    assert len(df) < config.n_workers * config.n_time
    assert obs_per_worker.min() == 2
    assert obs_per_worker.max() == config.n_time


def test_higher_entry_exit_share_reduces_average_observed_periods():
    common = {
        "n_workers": 400,
        "n_firms": 60,
        "n_time": 10,
        "n_industries": 4,
        "delta": 0.2,
        "rho": 1.0,
        "lambda_": 0.8,
        "entry_exit_n_periods": 2,
    }
    mild_unbalance_df = simulate_akm_panel(
        AKMConfig(**common, entry_exit_share=0.10),
        seed=31,
    )
    heavy_unbalance_df = simulate_akm_panel(
        AKMConfig(**common, entry_exit_share=0.75),
        seed=31,
    )

    mild_summary = summarize_akm_panel(mild_unbalance_df)
    heavy_summary = summarize_akm_panel(heavy_unbalance_df)

    assert (
        heavy_summary["mean_observed_periods"] < mild_summary["mean_observed_periods"]
    )
    assert (
        heavy_summary["two_period_worker_share"]
        > mild_summary["two_period_worker_share"]
    )


def test_incompatible_moves_force_occupation_change():
    df = simulate_akm_panel(
        AKMConfig(
            n_workers=400,
            n_firms=20,
            n_time=6,
            n_industries=1,
            lambda_=1.0,
            delta=1.0,
            n_occupations=2,
            occ_menu_size=1,
            occ_lambda=1.0,
            occ_delta=1.0,
        ),
        seed=41,
        include_latent=True,
    )

    forced = df["occ_forced_change"] & df["firm_changed"]

    assert forced.any()
    assert df.loc[forced, "occ_changed"].all()


def test_higher_occ_delta_increases_occupation_persistence_on_compatible_moves():
    common = {
        "n_workers": 500,
        "n_firms": 60,
        "n_time": 8,
        "n_industries": 3,
        "delta": 0.5,
        "rho": 1.0,
        "lambda_": 0.8,
        "n_occupations": 30,
        "occ_menu_size": 8,
        "occ_lambda": 0.5,
    }
    low_delta_df = simulate_akm_panel(
        AKMConfig(**common, occ_delta=0.1),
        seed=47,
        include_latent=True,
    )
    high_delta_df = simulate_akm_panel(
        AKMConfig(**common, occ_delta=0.9),
        seed=47,
        include_latent=True,
    )

    low_mask = low_delta_df["firm_changed"] & low_delta_df["occ_move_compatible"]
    high_mask = high_delta_df["firm_changed"] & high_delta_df["occ_move_compatible"]

    low_change_share = low_delta_df.loc[low_mask, "occ_changed"].mean()
    high_change_share = high_delta_df.loc[high_mask, "occ_changed"].mean()

    assert high_change_share < low_change_share


def test_higher_occ_lambda_increases_within_firm_occ_concentration():
    common = {
        "n_workers": 500,
        "n_firms": 60,
        "n_time": 8,
        "n_industries": 3,
        "delta": 0.2,
        "rho": 1.0,
        "lambda_": 0.8,
        "n_occupations": 50,
        "occ_menu_size": 5,
        "occ_delta": 0.3,
    }
    low_lambda_df = simulate_akm_panel(
        AKMConfig(**common, occ_lambda=0.1),
        seed=53,
    )
    high_lambda_df = simulate_akm_panel(
        AKMConfig(**common, occ_lambda=0.9),
        seed=53,
    )

    low_concentration = summarize_akm_panel(low_lambda_df)[
        "within_firm_occ_concentration"
    ]
    high_concentration = summarize_akm_panel(high_lambda_df)[
        "within_firm_occ_concentration"
    ]

    assert high_concentration > low_concentration
