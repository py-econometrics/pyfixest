from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _run_wildboottest(
    Y: np.ndarray,
    X: np.ndarray,
    xnames: list[str],
    cluster_name: str | None,
    cluster_array: np.ndarray | None,
    reps: int,
    param: str | None = None,
    weights_type: str | None = "rademacher",
    impose_null: bool | None = True,
    bootstrap_type: str | None = "11",
    seed: int | None = None,
    k_adj: bool | None = True,
    G_adj: bool | None = True,
    parallel: bool | None = False,
    return_bootstrapped_t_stats=False,
):
    """Compute wild bootstrap inference from a prepared design and clusters."""
    try:
        from wildboottest.wildboottest import WildboottestCL, WildboottestHC
    except ImportError:
        print(
            "Module 'wildboottest' not found. Please install 'wildboottest', e.g. via `PyPi`."
        )

    # later: allow r <> 0 and custom R
    R = np.zeros(len(xnames))
    if param is not None:
        R[xnames.index(param)] = 1
    r = 0

    if cluster_array is None:
        inference = "HC"

        boot = WildboottestHC(X=X, Y=Y, R=R, r=r, B=reps, seed=seed)
        boot.get_adjustments(bootstrap_type=bootstrap_type)
        boot.get_uhat(impose_null=impose_null)
        boot.get_tboot(weights_type=weights_type)
        boot.get_tstat()
        boot.get_pvalue(pval_type="two-tailed")
        full_enumeration_warn = False

    else:
        inference = f"CRV({cluster_name})"

        boot = WildboottestCL(
            X=X,
            Y=Y,
            cluster=cluster_array,
            R=R,
            B=reps,
            seed=seed,
            parallel=parallel,
        )
        boot.get_scores(
            bootstrap_type=bootstrap_type,
            impose_null=impose_null,
            adj=k_adj,
            cluster_adj=G_adj,
        )
        _, _, full_enumeration_warn = boot.get_weights(weights_type=weights_type)
        boot.get_numer()
        boot.get_denom()
        boot.get_tboot()
        boot.get_vcov()
        boot.get_tstat()
        boot.get_pvalue(pval_type="two-tailed")

        if full_enumeration_warn:
            warnings.warn(
                "2^G < the number of boot iterations, setting full_enumeration to True."
            )

    if np.isscalar(boot.t_stat):
        boot.t_stat = np.asarray(boot.t_stat)
    else:
        boot.t_stat = boot.t_stat[0]

    res = {
        "param": param,
        "t value": boot.t_stat.astype(np.float64),
        "Pr(>|t|)": np.asarray(boot.pvalue).astype(np.float64),
        "bootstrap_type": bootstrap_type,
        "inference": inference,
        "impose_null": impose_null,
        "ssc": boot.small_sample_correction if cluster_array is None else boot.ssc,
    }

    res_df = pd.Series(res)

    if return_bootstrapped_t_stats:
        return res_df, boot.t_boot
    else:
        return res_df
