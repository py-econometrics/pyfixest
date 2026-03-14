from pathlib import Path

from benchmarks.modular.dgp_functions import BipartiteConfig, simulate_bipartite
from benchmarks.modular.dgps import get_bipartite_scenarios


def test_get_bipartite_scenarios():
    scenarios = get_bipartite_scenarios(
        Path("unused"), ["akm_low_mobility", "akm_two_industry_bridge"]
    )

    assert [scenario.dgp_name for scenario in scenarios] == [
        "akm_low_mobility",
        "akm_two_industry_bridge",
    ]


def test_pareto_firm_sizes_create_more_concentrated_firm_ids():
    common = {
        "n_workers": 500,
        "n_time": 8,
        "firm_size": 5,
        "n_firm_types": 6,
        "n_worker_types": 6,
        "p_move": 0.10,
        "c_sort": 0.0,
    }
    equal_df = simulate_bipartite(
        BipartiteConfig(**common, firm_size_dist="equal"), seed=7
    )
    pareto_df = simulate_bipartite(
        BipartiteConfig(
            **common,
            firm_size_dist="pareto",
            firm_size_pareto_shape=1.5,
        ),
        seed=7,
    )

    equal_max = equal_df.groupby("firm_id").size().max()
    pareto_max = pareto_df.groupby("firm_id").size().max()

    assert pareto_max > equal_max
    assert {"worker_cluster", "firm_cluster"} <= set(pareto_df.columns)


def test_cluster_penalty_reduces_cross_cluster_worker_histories():
    common = {
        "n_workers": 400,
        "n_time": 10,
        "firm_size": 5,
        "n_firm_types": 8,
        "n_worker_types": 8,
        "p_move": 0.15,
        "c_sort": 0.0,
        "n_clusters": 2,
        "firm_size_dist": "lognormal",
    }
    unrestricted_df = simulate_bipartite(
        BipartiteConfig(**common, cross_cluster_scale=1.0), seed=13
    )
    bridge_df = simulate_bipartite(
        BipartiteConfig(**common, cross_cluster_scale=0.02), seed=13
    )

    unrestricted_share = (
        unrestricted_df.groupby("indiv_id")["firm_cluster"].nunique().gt(1).mean()
    )
    bridge_share = bridge_df.groupby("indiv_id")["firm_cluster"].nunique().gt(1).mean()

    assert bridge_share < unrestricted_share


def test_high_sorting_increases_worker_firm_type_alignment():
    common = {
        "n_workers": 400,
        "n_time": 8,
        "firm_size": 5,
        "n_firm_types": 6,
        "n_worker_types": 6,
        "p_move": 0.10,
        "firm_size_dist": "lognormal",
    }
    low_sort_df = simulate_bipartite(BipartiteConfig(**common, c_sort=0.0), seed=23)
    high_sort_df = simulate_bipartite(BipartiteConfig(**common, c_sort=3.0), seed=23)

    low_corr = low_sort_df["worker_type"].corr(low_sort_df["firm_type"])
    high_corr = high_sort_df["worker_type"].corr(high_sort_df["firm_type"])

    assert high_corr > low_corr
