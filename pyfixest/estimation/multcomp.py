import warnings
from typing import Union

import numpy as np
import pandas as pd

from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.fepois_ import Fepois
from pyfixest.report.summarize import _post_processing_input_checks


def bonferroni(models: list[Union[Feols, Fepois]], param: str) -> pd.DataFrame:
    """
    Compute Bonferroni adjusted p-values for multiple hypothesis testing.

    For each model, it is assumed that tests to adjust are of the form
    "param = 0".

    Parameters
    ----------
    models : list[Feols, Fepois], Feols or Fepois
        A list of models for which the p-values should be adjusted, or a Feols or
        Fepois object.
    param : str
        The parameter for which the p-values should be adjusted.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing estimation statistics, including the Bonferroni
        adjusted p-values.

    Examples
    --------
    ```python
    from pyfixest.estimation import feols
    from pyfixest.utils import get_data
    from pyfixest.multcomp import bonferroni

    data = get_data().dropna()
    fit1 = feols("Y ~ X1", data=data)
    fit2 = feols("Y ~ X1 + X2", data=data)
    bonf_df = bonferroni([fit1, fit2], param="X1")
    bonf_df
    ```
    """
    models = _post_processing_input_checks(models)
    all_model_stats = pd.DataFrame()
    S = len(models)
    pvalues = np.zeros(S)
    for i, model in enumerate(models):
        if param not in model._coefnames:
            raise ValueError(
                f"Parameter '{param}' not found in the model {model._fml}."
            )
        pvalues[i] = model.pvalue().xs(param)
        all_model_stats = pd.concat([all_model_stats, model.tidy().xs(param)], axis=1)

    adjusted_pvalues = np.minimum(1, pvalues * S)

    all_model_stats.loc["Bonferroni Pr(>|t|)"] = adjusted_pvalues
    all_model_stats.columns = pd.Index([f"est{i}" for i, _ in enumerate(models)])

    return all_model_stats


def rwolf(
    models: list[Union[Feols, Fepois]],
    param: str,
    reps: int,
    seed: int,
    sampling_method: str = "wild-bootstrap",
) -> pd.DataFrame:
    """
    Compute Romano-Wolf adjusted p-values for multiple hypothesis testing.

    For each model, it is assumed that tests to adjust are of the form
    "param = 0". This function uses the `wildboottest()` method for running the
    bootstrap, hence models of type `Feiv` or `Fepois` are not supported.

    Parameters
    ----------
    models : list[Feols] or FixestMulti
        A list of models for which the p-values should be computed, or a
        FixestMulti object.
        Models of type `Feiv` or `Fepois` are not supported.
    param : str
        The parameter for which the p-values should be computed.
    reps : int
        The number of bootstrap replications.
    seed : int
        The seed for the random number generator.
    sampling_method : str
        Sampling method for computing resampled statistics.
        Users can choose either bootstrap('wild-bootstrap')
        or randomization inference('ri')

    Returns
    -------
    pd.DataFrame
        A DataFrame containing estimation statistics, including the Romano-Wolf
        adjusted p-values.

    Examples
    --------
    ```python
    from pyfixest.estimation import feols
    from pyfixest.utils import get_data
    from pyfixest.multcomp import rwolf

    data = get_data().dropna()
    fit = feols("Y ~ Y2 + X1 + X2", data=data)
    rwolf(fit.to_list(), "X1", reps=9999, seed=123)

    fit1 = feols("Y ~ X1", data=data)
    fit2 = feols("Y ~ X1 + X2", data=data)
    rwolf_df = rwolf([fit1, fit2], "X1", reps=9999, seed=123)
    rwolf_df
    ```
    """
    models = _post_processing_input_checks(models)
    all_model_stats = pd.DataFrame()
    full_enumeration = False

    S = 0
    for model in models:
        if param not in model._coefnames:
            raise ValueError(
                f"Parameter '{param}' not found in the model {model._fml}."
            )

        if model._is_clustered:
            # model._G a list of length 3
            # for oneway clusering: repeated three times
            G = min(model._G)
            if reps > 2**G:
                warnings.warn(
                    f"""
                              2^(the number of clusters) < the number of boot iterations for at least one model,
                              setting full_enumeration to True and reps = {2**G}.
                              """
                )
                full_enumeration = True

        model_tidy = model.tidy().xs(param)
        all_model_stats = pd.concat([all_model_stats, model_tidy], axis=1)
        S += 1

    t_stats = np.zeros(S)
    boot_t_stats = np.zeros((2**G, S)) if full_enumeration else np.zeros((reps, S))

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

            t_stats[i] = model._ritest_sample_stat
            boot_t_stats[:, i] = model._ritest_statistics
        else:
            raise ValueError("Invalid sampling method specified")

    pval = _get_rwolf_pval(t_stats, boot_t_stats)

    all_model_stats.loc["RW Pr(>|t|)"] = pval
    all_model_stats.columns = pd.Index([f"est{i}" for i, _ in enumerate(models)])
    return all_model_stats


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
    models: list[Union[Feols, Fepois]],
    param: str,
    reps: int,
    seed: int,
    sampling_method: str = "wild-bootstrap",
) -> pd.DataFrame:
    """
    Compute the Westfall-Young adjusted p-values for multiple hypothesis testing.

    For each model, it is assumed that tests to adjust are of the form
    "param = 0". This function uses the `wildboottest()` method for running the
    bootstrap, hence models of type `Feiv` or `Fepois` are not supported.

    Parameters
    ----------
    models : list[Feols] or FixestMulti
        A list of models for which the p-values should be computed, or a
        FixestMulti object.
        Models of type `Feiv` or `Fepois` are not supported.
    param : str
        The parameter for which the p-values should be computed.
    reps : int
        The number of bootstrap replications.
    seed : int
        The seed for the random number generator.
    sampling_method : str
        Sampling method for computing resampled statistics.
        Users can choose either bootstrap('wild-bootstrap')
        or randomization inference('ri')

    Returns
    -------
    pd.DataFrame
        A DataFrame containing estimation statistics, including the Romano-Wolf
        adjusted p-values.

    Examples
    --------
    ```python
    from pyfixest.estimation import feols
    from pyfixest.utils import get_data
    from pyfixest.multcomp import rwolf

    data = get_data().dropna()
    fit = feols("Y ~ Y2 + X1 + X2", data=data)
    wyoung(fit, "X1", reps=9999, seed=123)

    fit1 = feols("Y ~ X1", data=data)
    fit2 = feols("Y ~ X1 + X2", data=data)
    rwolf_df = wyoung([fit1, fit2], "X1", reps=9999, seed=123)
    rwolf_df
    ```
    """
    models = _post_processing_input_checks(models)
    all_model_stats = pd.DataFrame()
    full_enumeration = False

    S = 0
    for model in models:
        if param not in model._coefnames:
            raise ValueError(
                f"Parameter '{param}' not found in the model {model._fml}."
            )

        if model._is_clustered:
            # model._G a list of length 3
            # for oneway clusering: repeated three times
            G = min(model._G)
            if reps > 2**G:
                warnings.warn(
                    f"""
                              2^(the number of clusters) < the number of boot iterations for at least one model,
                              setting full_enumeration to True and reps = {2**G}.
                              """
                )
                full_enumeration = True

        model_tidy = model.tidy().xs(param)
        all_model_stats = pd.concat([all_model_stats, model_tidy], axis=1)
        S += 1

    t_stats = np.zeros(S)
    boot_t_stats = np.zeros((2**G, S)) if full_enumeration else np.zeros((reps, S))

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

            t_stats[i] = model._ritest_sample_stat
            boot_t_stats[:, i] = model._ritest_statistics
        else:
            raise ValueError("Invalid sampling method specified")

    pval = _get_wyoung_pval(t_stats, boot_t_stats)

    all_model_stats.loc["RW Pr(>|t|)"] = pval
    all_model_stats.columns = pd.Index([f"est{i}" for i, _ in enumerate(models)])
    return all_model_stats


def _get_wyoung_pval(t_stats, boot_t_stats):
    """
    Compute Westfall-Young adjusted p-values based on bootstrapped(or "ri") t-statistics.

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
    t_stats = np.abs(t_stats)
    boot_t_stats = np.abs(boot_t_stats)

    S = boot_t_stats.shape[1]
    B = boot_t_stats.shape[0]

    pinit = corr_padj = pval = np.zeros(S)
    stepdown_index = np.argsort(t_stats)[::-1]
    ro = np.argsort(stepdown_index)

    # Step 3 (p.28 -- Westfall and Young Free step-down resampling method)
    # We sample t-stats instead of p-values
    Qs = np.maximum.accumulate(boot_t_stats[:, stepdown_index[::-1]], axis=1)[::-1]

    # Step 4 and 5
    t_against_null = np.greater_equal(Qs, t_stats[stepdown_index])

    pinit = t_against_null.sum(axis=0) / B

    # Step 6: Enforce monotonicity of adjusted p-values
    corr_padj = np.maximum.accumulate(pinit)

    pval = corr_padj[ro]

    return pval
