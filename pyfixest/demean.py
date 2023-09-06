import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
import pyhdfe


def demean_model(
    Y: pd.DataFrame,
    X: pd.DataFrame,
    fe: Optional[pd.DataFrame],
    weights: Optional[np.ndarray],
    lookup_demeaned_data: Dict[str, Any],
    na_index_str: str,
    drop_singletons: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Demeans a single regression model.

    If the model has fixed effects, the fixed effects are demeaned using the PyHDFE package.
    Prior to demeaning, the function checks if some of the variables have already been demeaned and uses values
    from the cache `lookup_demeaned_data` if possible. If the model has no fixed effects, the function does not demean the data.

    Args:
        Y (pd.DataFrame): A DataFrame of the dependent variable.
        X (pd.DataFrame): A DataFrame of the covariates.
        fe (pd.DataFrame or None): A DataFrame of the fixed effects. None if no fixed effects specified.
        weights (np.ndarray or None): A numpy array of weights. None if no weights.
        lookup_demeaned_data (Dict[str, Any]): A dictionary with keys for each fixed effects combination and
            potentially values of demeaned data frames. The function checks this dictionary to see if some of
            the variables have already been demeaned.
        na_index_str (str): A string with indices of dropped columns. Used for caching of demeaned variables.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]: A tuple of the following elements:
            - Yd (pd.DataFrame): A DataFrame of the demeaned dependent variable.
            - Xd (pd.DataFrame): A DataFrame of the demeaned covariates.
            - Id (pd.DataFrame or None): A DataFrame of the demeaned Instruments. None if no IV.
    """

    YX = pd.concat([Y, X], axis=1)

    yx_names = YX.columns
    YX = YX.to_numpy()

    if fe is not None:
        # check if looked dict has data for na_index
        if lookup_demeaned_data.get(na_index_str) is not None:
            # get data out of lookup table: list of [algo, data]
            algorithm, YX_demeaned_old = lookup_demeaned_data.get(na_index_str)

            # get not yet demeaned covariates
            var_diff_names = list(set(yx_names) - set(YX_demeaned_old.columns))

            # if some variables still need to be demeaned
            if var_diff_names:
                # var_diff_names = var_diff_names

                yx_names_list = list(yx_names)
                var_diff_index = [yx_names_list.index(item) for item in var_diff_names]
                # var_diff_index = list(yx_names).index(var_diff_names)
                var_diff = YX[:, var_diff_index]
                if var_diff.ndim == 1:
                    var_diff = var_diff.reshape(len(var_diff), 1)

                YX_demean_new = algorithm.residualize(var_diff)
                YX_demeaned = np.concatenate([YX_demeaned_old, YX_demean_new], axis=1)
                YX_demeaned = pd.DataFrame(YX_demeaned)

                # check if var_diff_names is a list
                if isinstance(var_diff_names, str):
                    var_diff_names = [var_diff_names]

                YX_demeaned.columns = list(YX_demeaned_old.columns) + var_diff_names

            else:
                # all variables already demeaned
                YX_demeaned = YX_demeaned_old[yx_names]

        else:
            # not data demeaned yet for NA combination
            algorithm = pyhdfe.create(
                ids=fe,
                residualize_method="map",
                drop_singletons=drop_singletons,
                # weights=weights
            )

            if (
                drop_singletons == True
                and algorithm.singletons != 0
                and algorithm.singletons is not None
            ):
                print(
                    algorithm.singletons,
                    "columns are dropped due to singleton fixed effects.",
                )
                dropped_singleton_indices = np.where(algorithm._singleton_indices)[
                    0
                ].tolist()
                na_index += dropped_singleton_indices

            YX_demeaned = algorithm.residualize(YX)
            YX_demeaned = pd.DataFrame(YX_demeaned)

            YX_demeaned.columns = yx_names

        lookup_demeaned_data[na_index_str] = [algorithm, YX_demeaned]

    else:
        # nothing to demean here
        pass

        YX_demeaned = pd.DataFrame(YX)
        YX_demeaned.columns = yx_names

    # get demeaned Y, X (if no fixef, equal to Y, X, I)
    Yd = YX_demeaned[Y.columns]
    Xd = YX_demeaned[X.columns]

    return Yd, Xd
