"""Wild (cluster) bootstrap inference via the `wildboottest` package."""

import warnings

import numpy as np
import pandas as pd

from pyfixest.estimation.post_estimation.decomposition import _model_matrix_one_hot


def _wildboottest_impl(
    model,
    reps: int,
    cluster: str | None = None,
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
    "Implementation of Feols.wildboottest; see the method docstring for details."
    if param is not None and param not in model._coefnames:
        raise ValueError(f"Parameter {param} not found in the model's coefficients.")

    if not model._supports_wildboottest:
        if model._is_iv:
            raise NotImplementedError(
                "Wild cluster bootstrap is not supported for IV estimation."
            )
        if model._has_weights:
            raise NotImplementedError(
                "Wild cluster bootstrap is not supported for WLS estimation."
            )

    cluster_list = []

    if cluster is not None and isinstance(cluster, str):
        cluster_list = [cluster]
    if cluster is not None and isinstance(cluster, list):
        cluster_list = cluster

    if cluster is None and model._clustervar is not None:
        if isinstance(model._clustervar, str):
            cluster_list = [model._clustervar]
        else:
            cluster_list = model._clustervar

    run_heteroskedastic = not cluster_list

    if not run_heteroskedastic and not len(cluster_list) == 1:
        raise NotImplementedError(
            "Multiway clustering is currently not supported with the wild cluster bootstrap."
        )

    if not run_heteroskedastic and cluster_list[0] not in model._data.columns:
        raise ValueError(f"Cluster variable {cluster_list[0]} not found in the data.")

    try:
        from wildboottest.wildboottest import WildboottestCL, WildboottestHC
    except ImportError:
        print(
            "Module 'wildboottest' not found. Please install 'wildboottest', e.g. via `PyPi`."
        )

    if model._is_iv:
        raise NotImplementedError(
            "Wild cluster bootstrap is not supported with IV estimation."
        )

    if model._method == "fepois":
        raise NotImplementedError(
            "Wild cluster bootstrap is not supported for Poisson regression."
        )

    _Y, _X, _xnames = _model_matrix_one_hot(model)

    # later: allow r <> 0 and custom R
    R = np.zeros(len(_xnames))
    if param is not None:
        R[_xnames.index(param)] = 1
    r = 0

    if run_heteroskedastic:
        inference = "HC"

        boot = WildboottestHC(X=_X, Y=_Y, R=R, r=r, B=reps, seed=seed)
        boot.get_adjustments(bootstrap_type=bootstrap_type)
        boot.get_uhat(impose_null=impose_null)
        boot.get_tboot(weights_type=weights_type)
        boot.get_tstat()
        boot.get_pvalue(pval_type="two-tailed")
        full_enumeration_warn = False

    else:
        inference = f"CRV({cluster_list[0]})"

        cluster_array = model._data[cluster_list[0]].to_numpy().flatten()

        boot = WildboottestCL(
            X=_X,
            Y=_Y,
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
        "ssc": boot.small_sample_correction if run_heteroskedastic else boot.ssc,
    }

    res_df = pd.Series(res)

    if return_bootstrapped_t_stats:
        return res_df, boot.t_boot
    else:
        return res_df
