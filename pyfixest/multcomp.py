from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from pyfixest.feols import Feols
from pyfixest.summarize import _post_processing_input_checks


def rwolf(
    models: Union[list[Feols], Feols], param: str, B: int, seed: int
) -> pd.DataFrame:
    """
    Compute Romano-Wolf adjusted p-values for multiple hypothesis testing.

    For each model, it is assumed that the adjustment is for the family of hypotheses is
    "param = 0". This function uses the `wildboottest()` method for running the bootstrap,
    hence models of type `Feiv` or `Fepois` are not supported.

    Parameters
    ----------
    models : list[Feols] or FixestMulti
        A list of models for which the p-values should be computed, or a FixestMulti object.
        Models of type `Feiv` or `Fepois` are not supported.
    param : str
        The parameter for which the p-values should be computed.
    B : int
        The number of bootstrap replications.
    seed : int
        The seed for the random number generator.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing estimation statistics, including the Romano-Wolf adjusted p-values.

    Examples
    --------
    ```python
    from pyfixest.estimation import feols
    from pyfixest.utils import get_data
    from pyfixest.multcomp import rwolf

    data = get_data().dropna()
    fit = feols("Y ~ Y2 + X1 + X2", data=data)
    rwolf(fit.to_list(), "X1", B=9999, seed=123)

    fit1 = feols("Y ~ X1", data=data)
    fit2 = feols("Y ~ X1 + X2", data=data)
    rwolf([fit1, fit2], "X1", B=9999, seed=123)
    ```
    """

    models = _post_processing_input_checks(models)
    all_model_stats = pd.DataFrame()

    S = 0
    for model in models:

        if param not in model._coefnames:
            raise ValueError(
                f"Parameter '{param}' not found in the model {model._fml}."
            )

        model_tidy = model.tidy().xs(param)
        all_model_stats = pd.concat([all_model_stats, model_tidy], axis=1)
        S += 1

    t_stats = all_model_stats.xs("t value").values
    t_stats = np.zeros(S)
    boot_t_stats = np.zeros((B, S))

    for i in tqdm(range(S)):

        model = models[i]

        wildboot_res_df, bootstrapped_t_stats = model.wildboottest(
            param=param,
            B=B,
            return_bootstrapped_t_stats=True,
            seed=seed,  # all S iterations require the same bootstrap samples, hence seed needs to be reset
        )
        t_stats[i] = wildboot_res_df["t value"]
        boot_t_stats[:, i] = bootstrapped_t_stats

    pval = _get_rwolf_pval(t_stats, boot_t_stats)

    all_model_stats.loc["RW Pr(>|t|)"] = pval
    all_model_stats.columns = [f"est{i}" for i, _ in enumerate(models)]

    return all_model_stats


def _get_rwolf_pval(t_stats, boot_t_stats):
    """
    Compute Romano-Wolf adjusted p-values based on bootstrapped t-statistics.

    Parameters:
    t_stats (np.ndarray): A vector of length S - where S is the number of
                          tested hypotheses - containing the original,
                          non-bootstrappe t-statisics.
    boot_t_stats (np.ndarray): A (B x S) matrix containing the
                               bootstrapped t-statistics.

    Returns:
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
