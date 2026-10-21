import warnings

import numpy as np
import pandas as pd
from scipy.stats import t

from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.models.feiv_ import Feiv
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.models.fepois_ import Fepois
from pyfixest.report.utils import _post_processing_input_checks
from pyfixest.utils.dev_utils import _find_stack_level

ModelInputType = FixestMulti | list[Feols | Fepois | Feiv]


def bonferroni(models: ModelInputType, param: str) -> pd.DataFrame:
    """
    Compute Bonferroni adjusted p-values for multiple hypothesis testing.

    Call as `pf.bonferroni(models, param=...)` after `import pyfixest as pf`.
    For each model, it is assumed that tests to adjust are of the form
    `param = 0`.

    Parameters
    ----------
    models : Feols or FixestMulti or list
        A fitted result with ordinary coefficient p-values, a list of such
        results, or a `FixestMulti` container. Includes supported OLS, IV,
        Poisson, other GLM and quantile fits. No resampling is performed.

    param : str
        Coefficient name present in every model. Each hypothesis tests that
        coefficient against zero; the family contains one test per model.

    Returns
    -------
    pd.DataFrame
        Statistics as rows and models as columns `est0`, `est1`, ... in input
        order. Includes the ordinary coefficient-table rows and an additional
        `Bonferroni Pr(>|t|)` row containing Bonferroni adjusted p-values.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    data = pf.get_data().dropna()
    fit1 = pf.feols("Y ~ X1", data=data)
    fit2 = pf.feols("Y ~ X1 + X2", data=data)
    pf.bonferroni([fit1, fit2], param="X1")
    ```

    Notes
    -----
    Adjusted p-values equal `min(1, number_of_models * pvalue)`. This method
    controls the family-wise error rate under arbitrary dependence between
    tests, provided each model's underlying p-value is valid.

    See Also
    --------
    Compare [pf.rwolf()](/reference/estimation.post_estimation.multcomp.rwolf.qmd) and
    [pf.wyoung()](/reference/estimation.post_estimation.multcomp.wyoung.qmd).
    [fit.wildboottest()](/reference/estimation.models.feols_.Feols.wildboottest.qmd)
    and [fit.ritest()](/reference/estimation.models.feols_.Feols.ritest.qmd) document the resampling methods.
    """
    models = _post_processing_input_checks(models)
    all_model_stats = pd.DataFrame()
    S = len(models)
    pvalues = np.zeros(S)
    for i, model in enumerate(models):
        if param not in model._coefnames:
            raise ValueError(
                f"Parameter '{param}' not found in the model {model.model.formula}."
            )
        pvalues[i] = model.pvalue().xs(param)
        all_model_stats = pd.concat([all_model_stats, model.tidy().xs(param)], axis=1)

    adjusted_pvalues = np.minimum(1, pvalues * S)

    all_model_stats.loc["Bonferroni Pr(>|t|)"] = adjusted_pvalues
    all_model_stats.columns = pd.Index([f"est{i}" for i, _ in enumerate(models)])

    return all_model_stats


def rwolf(
    models: ModelInputType,
    param: str,
    reps: int,
    seed: int,
    sampling_method: str = "wild-bootstrap",
) -> pd.DataFrame:
    """
    Compute Romano-Wolf adjusted p-values for multiple hypothesis testing.

    Call as `pf.rwolf(models, param=...)` after `import pyfixest as pf`.
    For each model, it is assumed that tests to adjust are of the form
    `param = 0`. The resampled statistics come from the `wildboottest()`
    method (`sampling_method="wild-bootstrap"`, the default) or from the
    `ritest()` method (`sampling_method="ri"`), so each model must support
    the chosen method:

    - `"wild-bootstrap"`: unweighted `Feols` fits only. `Feiv`, `Fepois`,
      `Feglm` and weighted `Feols` fits raise a `NotImplementedError`.
    - `"ri"`: unweighted `Feols` fits without instruments and `Fepois` fits.
      `Feiv`, `Feglm` and weighted fits raise a `NotImplementedError`.

    Parameters
    ----------
    models : Feols, Fepois, list or FixestMulti
        A fitted model, a list of fitted models, or a `FixestMulti` object.
    param : str
        Coefficient name present in every model. Each hypothesis tests that
        coefficient against zero; the family contains one test per model.

    reps : int
        The number of bootstrap replications (`"wild-bootstrap"`) or random
        permutations (`"ri"`). Use a common one-way cluster structure across
        clustered models. If `reps` exceeds `2**G` for `G` clusters, wild
        bootstrap uses all `2**G` sign vectors and warns. With
        `sampling_method="ri"`, this case currently raises an error, so keep
        `reps` at or below `2**G`.
    seed : int
        The seed for the random number generator.
    sampling_method : str, optional
        Sampling method for computing resampled statistics: `"wild-bootstrap"`
        (default) for a wild (cluster) bootstrap or `"ri"` for randomization
        inference.

    Returns
    -------
    pd.DataFrame
        Statistics as rows and models as columns `est0`, `est1`, ... in input
        order. Includes the ordinary coefficient-table rows and an additional
        `RW Pr(>|t|)` row containing Romano-Wolf adjusted p-values.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    data = pf.get_data().dropna()
    fit1 = pf.feols("Y ~ X1", data=data)
    fit2 = pf.feols("Y ~ X1 + X2", data=data)
    pf.rwolf([fit1, fit2], param="X1", reps=999, seed=123)
    ```

    Notes
    -----
    Resampling uses the same seed across models to preserve dependence
    between tests. Use compatible observations and row order across models;
    the function does not align samples for you. Wild bootstrap requires
    the optional `wildboottest` package and supports at most one cluster
    dimension. With randomization inference, the supplied parameter is the
    variable permuted by `fit.ritest()`; that assignment mechanism must be
    appropriate for the application.

    Methods may need retained model data; see the
    [storage guide](/reference-guide/model-storage.qmd). RI writes each
    fitted result's `ritest_statistics` state. These routines do not adjust
    coefficients or their confidence intervals.

    See Also
    --------
    Compare [pf.bonferroni()](/reference/estimation.post_estimation.multcomp.bonferroni.qmd) and
    [pf.wyoung()](/reference/estimation.post_estimation.multcomp.wyoung.qmd).
    [fit.wildboottest()](/reference/estimation.models.feols_.Feols.wildboottest.qmd)
    and [fit.ritest()](/reference/estimation.models.feols_.Feols.ritest.qmd) document the resampling methods.
    """
    return _multcomp_resample("rwolf", models, param, reps, seed, sampling_method)


def _get_rwolf_pval(t_stats, boot_t_stats):
    """
    Compute Romano-Wolf adjusted p-values based on bootstrapped(or "ri") t-statistics.

    Parameters
    ----------
    t_stats (np.ndarray): A vector of length S - where S is the number of
                        tested hypotheses - containing the original,
                        non-bootstrappe t-statisics.
    boot_t_stats (np.ndarray): A (B x S) matrix containing the
                            bootstrapped(or "ri") t-statistics.

    Returns
    -------
    np.ndarray: A vector of Romano-Wolf corrected p-values.
    """
    t_stats = np.abs(t_stats)
    boot_t_stats = np.abs(boot_t_stats)

    S = boot_t_stats.shape[1]
    B = boot_t_stats.shape[0]

    pinit = corr_padj = pval = np.zeros(S)
    stepdown_index = np.argsort(t_stats)[::-1]
    ro = np.argsort(stepdown_index)

    for s in range(S):
        if s == 0:
            max_stat = np.max(boot_t_stats, axis=1)
            pinit[s] = min(
                1,
                (np.sum(max_stat >= np.abs(t_stats[stepdown_index[s]])) + 1) / (B + 1),
            )
        else:
            boot_t_stat_udp = np.delete(boot_t_stats, stepdown_index[:s], axis=1)
            max_stat = np.max(boot_t_stat_udp, axis=1)
            pinit[s] = min(
                1,
                (np.sum(max_stat >= np.abs(t_stats[stepdown_index[s]])) + 1) / (B + 1),
            )

    for j in range(S):
        if j == 0:
            corr_padj[j] = pinit[j]
        else:
            corr_padj[j] = max(pinit[j], corr_padj[j - 1])

    # Collect the results
    pval = corr_padj[ro]

    return pval


def wyoung(
    models: ModelInputType,
    param: str,
    reps: int,
    seed: int,
    sampling_method: str = "wild-bootstrap",
) -> pd.DataFrame:
    """
    Compute the Westfall-Young adjusted p-values for multiple hypothesis testing.

    Call as `pf.wyoung(models, param=...)` after `import pyfixest as pf`.
    For each model, it is assumed that tests to adjust are of the form
    `param = 0`. The resampled statistics come from the `wildboottest()`
    method (`sampling_method="wild-bootstrap"`, the default) or from the
    `ritest()` method (`sampling_method="ri"`), so each model must support
    the chosen method:

    - `"wild-bootstrap"`: unweighted `Feols` fits only. `Feiv`, `Fepois`,
      `Feglm` and weighted `Feols` fits raise a `NotImplementedError`.
    - `"ri"`: unweighted `Feols` fits without instruments and `Fepois` fits.
      `Feiv`, `Feglm` and weighted fits raise a `NotImplementedError`.

    Parameters
    ----------
    models : Feols, Fepois, list or FixestMulti
        A fitted model, a list of fitted models, or a `FixestMulti` object.
    param : str
        Coefficient name present in every model. Each hypothesis tests that
        coefficient against zero; the family contains one test per model.

    reps : int
        The number of bootstrap replications (`"wild-bootstrap"`) or random
        permutations (`"ri"`). Use a common one-way cluster structure across
        clustered models. If `reps` exceeds `2**G` for `G` clusters, wild
        bootstrap uses all `2**G` sign vectors and warns. With
        `sampling_method="ri"`, this case currently raises an error, so keep
        `reps` at or below `2**G`.
    seed : int
        The seed for the random number generator.
    sampling_method : str, optional
        Sampling method for computing resampled statistics: `"wild-bootstrap"`
        (default) for a wild (cluster) bootstrap or `"ri"` for randomization
        inference.

    Returns
    -------
    pd.DataFrame
        Statistics as rows and models as columns `est0`, `est1`, ... in input
        order. Includes the ordinary coefficient-table rows and an additional
        `WY Pr(>|t|)` row containing Westfall-Young adjusted p-values.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    data = pf.get_data().dropna()
    fit1 = pf.feols("Y ~ X1", data=data)
    fit2 = pf.feols("Y ~ X1 + X2", data=data)
    pf.wyoung([fit1, fit2], param="X1", reps=999, seed=123)
    ```

    Notes
    -----
    Resampling uses the same seed across models to preserve dependence
    between tests. Use compatible observations and row order across models;
    the function does not align samples for you. Wild bootstrap requires
    the optional `wildboottest` package and supports at most one cluster
    dimension. With randomization inference, the supplied parameter is the
    variable permuted by `fit.ritest()`; that assignment mechanism must be
    appropriate for the application.

    Methods may need retained model data; see the
    [storage guide](/reference-guide/model-storage.qmd). RI writes each
    fitted result's `ritest_statistics` state. These routines do not adjust
    coefficients or their confidence intervals.

    See Also
    --------
    Compare [pf.bonferroni()](/reference/estimation.post_estimation.multcomp.bonferroni.qmd) and
    [pf.rwolf()](/reference/estimation.post_estimation.multcomp.rwolf.qmd).
    [fit.wildboottest()](/reference/estimation.models.feols_.Feols.wildboottest.qmd)
    and [fit.ritest()](/reference/estimation.models.feols_.Feols.ritest.qmd) document the resampling methods.
    """
    return _multcomp_resample("wyoung", models, param, reps, seed, sampling_method)


def _get_wyoung_pval(p_vals, boot_p_vals):
    """
    Compute Westfall-Young adjusted p-values based on bootstrapped t-statistics.

    Parameters
    ----------
    t_stats (np.ndarray): A vector of length S - where S is the number of
                        tested hypotheses - containing the original,
                        non-bootstrappe t-statisics.
    boot_t_stats (np.ndarray): A (B x S) matrix containing the
                            bootstrapped(or "ri") t-statistics.

    Returns
    -------
    np.ndarray: A vector of Westfall-Young corrected p-values.
    """
    S = boot_p_vals.shape[1]
    B = boot_p_vals.shape[0]

    pinit = corr_padj = pval = np.zeros(S)
    stepdown_index = np.argsort(p_vals)
    ro = np.argsort(stepdown_index)

    # Step 3 (p.28 -- Westfall and Young Free step-down resampling method)
    Qs = np.minimum.accumulate(boot_p_vals[:, stepdown_index[::-1]], axis=1)[:, ::-1]

    # Step 4 and 5
    p_against_null = np.less_equal(Qs, p_vals[stepdown_index])

    pinit = p_against_null.sum(axis=0) / B

    # Step 6: Enforce monotonicity of adjusted p-values
    corr_padj = np.maximum.accumulate(pinit)

    pval = corr_padj[ro]

    return pval


def _multcomp_resample(
    type: str,
    models: ModelInputType,
    param: str,
    reps: int,
    seed: int,
    sampling_method: str = "wild-bootstrap",
) -> pd.DataFrame:
    if isinstance(models, FixestMulti):
        models = models.to_list()

    models = _post_processing_input_checks(models)
    if type not in ["rwolf", "wyoung"]:
        raise ValueError("Type should be one of 'rwolf' and 'wyoung'")
    if sampling_method not in ["wild-bootstrap", "ri"]:
        raise ValueError(
            "Adjustment procedure should be one of 'wild-bootstrap' and 'ri'"
        )

    all_model_stats = pd.DataFrame()
    full_enumeration = False

    S = 0
    for model in models:
        if param not in model._coefnames:
            raise ValueError(
                f"Parameter '{param}' not found in the model {model.model.formula}."
            )

        if model.variance_covariance.spec.is_clustered:
            # covariance.G has one entry per cluster dimension
            G = min(model.variance_covariance.G)
            if reps > 2**G:
                warnings.warn(
                    f"""
                              2^(the number of clusters) < the number of boot iterations for at least one model,
                              setting full_enumeration to True and reps = {2**G}.
                              """,
                    UserWarning,
                    stacklevel=_find_stack_level(),
                )
                full_enumeration = True

        model_tidy = model.tidy().xs(param)
        all_model_stats = pd.concat([all_model_stats, model_tidy], axis=1)
        S += 1

    t_stats = np.zeros(S)
    boot_t_stats = np.zeros((2**G, S)) if full_enumeration else np.zeros((reps, S))

    _df = np.zeros_like(t_stats)
    p_vals = np.zeros_like(t_stats)
    boot_p_vals = np.zeros_like(boot_t_stats)

    for i in range(S):
        model = models[i]

        if sampling_method == "wild-bootstrap":
            wildboot_res_df, bootstrapped_t_stats = model.wildboottest(
                param=param,
                reps=reps,
                return_bootstrapped_t_stats=True,
                seed=seed,  # all S iterations require the same bootstrap samples, hence seed needs to be reset
            )
            t_stats[i] = wildboot_res_df["t value"]
            boot_t_stats[:, i] = bootstrapped_t_stats

        elif sampling_method == "ri":
            rng = np.random.default_rng(seed)
            model.ritest(
                resampvar=param,
                rng=rng,
                reps=reps,
                type="randomization-t",
                store_ritest_statistics=True,
            )

            stored = model.ritest_statistics
            t_stats[i] = stored.sample_stat
            boot_t_stats[:, i] = stored.statistics

        if type == "wyoung":
            _df[i] = (
                model.sample_info.n_obs - model._k
                if model.variance_covariance.spec.vcov_type in ["iid", "hetero"]
                else min(model.variance_covariance.G) - 1
            )
            p_vals[i] = 2 * (1 - t.cdf(np.abs(t_stats[i]), _df[i]))
            boot_p_vals[:, i] = 2 * (1 - t.cdf(np.abs(boot_t_stats[:, i]), _df[i]))

    if type == "rwolf":
        pval = _get_rwolf_pval(t_stats, boot_t_stats)
        all_model_stats.loc["RW Pr(>|t|)"] = pval
    else:
        pval = _get_wyoung_pval(p_vals, boot_p_vals)
        all_model_stats.loc["WY Pr(>|t|)"] = pval

    all_model_stats.columns = pd.Index([f"est{i}" for i, _ in enumerate(models)])
    return all_model_stats
