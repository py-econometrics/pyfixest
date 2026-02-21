from importlib import import_module
from typing import Optional, Union

import numpy as np
import pandas as pd

from pyfixest.errors import VcovTypeNotSupportedError
from pyfixest.estimation.internals.vcov_utils import (
    _check_cluster_df,
    _check_vcov_input,
    _compute_bread,
    _count_G_for_ssc_correction,
    _deparse_vcov_input,
    _dk_meat_panel,
    _get_cluster_df,
    _get_panel_idx,
    _nw_meat_panel,
    _nw_meat_time,
    _prepare_twoway_clustering,
)
from pyfixest.utils.dev_utils import DataFrameType, _narwhals_to_pandas
from pyfixest.utils.utils import get_ssc


def compute_vcov(
    model,
    vcov: Union[str, dict[str, str]],
    vcov_kwargs: Optional[dict[str, Union[str, int]]] = None,
    data: Optional[DataFrameType] = None,
) -> None:
    data_to_check = data if data is not None else model._data
    try:
        data_to_check = _narwhals_to_pandas(data_to_check)
    except TypeError as e:
        raise TypeError(
            f"The data set must be a DataFrame type. Received: {type(data)}"
        ) from e

    _check_vcov_input(vcov=vcov, vcov_kwargs=vcov_kwargs, data=data_to_check)

    (
        model._vcov_type,
        model._vcov_type_detail,
        model._is_clustered,
        model._clustervar,
    ) = _deparse_vcov_input(vcov, model._has_fixef, model._is_iv)

    model._bread = _compute_bread(
        model._is_iv, model._tXZ, model._tZZinv, model._tZX, model._hessian
    )

    # HAC attributes
    model._lag = vcov_kwargs.get("lag", None) if vcov_kwargs is not None else None
    model._time_id = (
        vcov_kwargs.get("time_id", None) if vcov_kwargs is not None else None
    )
    model._panel_id = (
        vcov_kwargs.get("panel_id", None) if vcov_kwargs is not None else None
    )
    model._is_sorted = (
        vcov_kwargs.get("is_sorted", None) if vcov_kwargs is not None else None
    )

    ssc_kwargs = {
        "ssc_dict": model._ssc_dict,
        "N": model._N,
        "k": model._k,
        "k_fe": model._k_fe.sum() if model._has_fixef else 0,
        "n_fe": model._n_fe,
    }

    if model._vcov_type == "iid":
        ssc_kwargs_iid = {
            "k_fe_nested": 0,
            "n_fe_fully_nested": 0,
            "vcov_sign": 1,
            "vcov_type": "iid",
            "G": 1,
        }

        all_kwargs = {**ssc_kwargs, **ssc_kwargs_iid}
        model._ssc, model._df_k, model._df_t = get_ssc(**all_kwargs)
        model._vcov = model._ssc * model._vcov_iid()

    elif model._vcov_type == "hetero":
        ssc_kwargs_hetero = {
            "k_fe_nested": 0,
            "n_fe_fully_nested": 0,
            "vcov_sign": 1,
            "vcov_type": "hetero",
            "G": model._N,
        }

        all_kwargs = {**ssc_kwargs, **ssc_kwargs_hetero}
        model._ssc, model._df_k, model._df_t = get_ssc(**all_kwargs)
        model._vcov = model._ssc * model._vcov_hetero()

    elif model._vcov_type == "HAC":
        ssc_kwargs_hac = {
            "k_fe_nested": 0,
            "n_fe_fully_nested": 0,
            "vcov_sign": 1,
            "vcov_type": "HAC",
            "G": np.unique(model._data[model._time_id]).shape[0],
        }

        all_kwargs = {**ssc_kwargs, **ssc_kwargs_hac}
        model._ssc, model._df_k, model._df_t = get_ssc(**all_kwargs)
        model._vcov = model._ssc * model._vcov_hac()

    elif model._vcov_type == "nid":
        ssc_kwargs_hetero = {
            "k_fe_nested": 0,
            "n_fe_fully_nested": 0,
            "vcov_sign": 1,
            "vcov_type": "hetero",
            "G": model._N,
        }

        all_kwargs = {**ssc_kwargs, **ssc_kwargs_hetero}
        model._ssc, model._df_k, model._df_t = get_ssc(**all_kwargs)
        model._vcov = model._ssc * model._vcov_nid()

    elif model._vcov_type == "CRV":
        if data is not None:
            model._cluster_df = _get_cluster_df(
                data=data,
                clustervar=model._clustervar,
            )
            _check_cluster_df(cluster_df=model._cluster_df, data=data)
        else:
            model._cluster_df = _get_cluster_df(
                data=model._data, clustervar=model._clustervar
            )
            _check_cluster_df(cluster_df=model._cluster_df, data=model._data)

        if model._cluster_df.shape[1] > 1:
            model._cluster_df = _prepare_twoway_clustering(
                clustervar=model._clustervar, cluster_df=model._cluster_df
            )

        model._G = _count_G_for_ssc_correction(
            cluster_df=model._cluster_df, ssc_dict=model._ssc_dict
        )

        vcov_sign_list = [1, 1, -1]
        df_t_full = np.zeros(model._cluster_df.shape[1])

        cluster_arr_int = np.column_stack(
            [pd.factorize(model._cluster_df[col])[0] for col in model._cluster_df.columns]
        )

        k_fe_nested = 0
        n_fe_fully_nested = 0
        if model._fixef is not None and model._ssc_dict["k_fixef"] == "nonnested":
            k_fe_nested_flag, n_fe_fully_nested = model._count_nested_fixef_func(
                all_fixef_array=np.array(model._fixef.replace("^", "_").split("+"), dtype=str),
                cluster_colnames=np.array(model._cluster_df.columns, dtype=str),
                cluster_data=cluster_arr_int.astype(np.uintp),
                fe_data=model._fe.to_numpy().astype(np.uintp)
                if isinstance(model._fe, pd.DataFrame)
                else model._fe.astype(np.uintp),
            )
            k_fe_nested = (
                np.sum(model._k_fe[k_fe_nested_flag]) if n_fe_fully_nested > 0 else 0
            )

        model._vcov = np.zeros((model._k, model._k))

        for x, _ in enumerate(model._cluster_df.columns):
            cluster_col = cluster_arr_int[:, x]
            clustid = np.unique(cluster_col)

            ssc_kwargs_crv = {
                "k_fe_nested": k_fe_nested,
                "n_fe_fully_nested": n_fe_fully_nested,
                "G": model._G[x],
                "vcov_sign": vcov_sign_list[x],
                "vcov_type": "CRV",
            }

            all_kwargs = {**ssc_kwargs, **ssc_kwargs_crv}
            ssc, df_k, df_t = get_ssc(**all_kwargs)

            model._ssc = np.array([ssc]) if x == 0 else np.append(model._ssc, ssc)
            model._df_k = df_k
            df_t_full[x] = df_t

            if model._vcov_type_detail == "CRV1":
                model._vcov += model._ssc[x] * model._vcov_crv1(
                    clustid=clustid, cluster_col=cluster_col
                )
            elif model._vcov_type_detail == "CRV3":
                if not model._support_crv3_inference:
                    raise VcovTypeNotSupportedError(
                        f"CRV3 inference is not for models of type '{model._method}'."
                    )

                if (
                    (model._has_fixef is False)
                    and (model._method == "feols")
                    and (model._is_iv is False)
                ):
                    model._vcov += model._ssc[x] * model._vcov_crv3_fast(
                        clustid=clustid, cluster_col=cluster_col
                    )
                else:
                    model._vcov += model._ssc[x] * model._vcov_crv3_slow(
                        clustid=clustid, cluster_col=cluster_col
                    )

        model._df_t = np.min(df_t_full)


def vcov_iid(model):
    sigma2 = np.sum(model._u_hat.flatten() ** 2) / (model._N - 1)
    return model._bread * sigma2


def vcov_hetero(model):
    if model._vcov_type_detail in ["hetero", "HC1"]:
        transformed_scores = model._scores
    elif model._vcov_type_detail in ["HC2", "HC3"]:
        leverage = np.sum(model._X * (model._X @ np.linalg.inv(model._tZX)), axis=1)
        if model._weights_type == "fweights":
            leverage = leverage / model._weights.flatten()
        transformed_scores = (
            model._scores / np.sqrt(1 - leverage)[:, None]
            if model._vcov_type_detail == "HC2"
            else model._scores / (1 - leverage)[:, None]
        )
    else:
        transformed_scores = model._scores

    if model._weights_type == "fweights":
        transformed_scores = transformed_scores / np.sqrt(model._weights)

    omega = transformed_scores.T @ transformed_scores
    meat = (
        model._tXZ @ model._tZZinv @ omega @ model._tZZinv @ model._tZX
        if model._is_iv
        else omega
    )
    return model._bread @ meat @ model._bread


def vcov_hac(model):
    if not model._support_hac_inference:
        raise NotImplementedError("HAC inference is not supported for this model type.")

    if not np.issubdtype(model._data[model._time_id], np.number) and not np.issubdtype(
        model._data[model._time_id], np.datetime64
    ):
        raise ValueError(
            "The time variable must be numeric or date, else we cannot sort by time."
        )

    time_arr = model._data[model._time_id].to_numpy()
    panel_arr = (
        model._data[model._panel_id].to_numpy() if model._panel_id is not None else None
    )

    if model._vcov_type_detail == "NW":
        if model._panel_id is None:
            if model._lag is None:
                raise ValueError(
                    "We have not yet implemented the default Newey-West HAC lag. Please provide a lag value via the `vcov_kwargs`."
                )
            if len(np.unique(time_arr)) != len(time_arr):
                raise ValueError(
                    "There are duplicate time periods in the data. This is not supported for HAC SEs."
                )
            hac_meat = _nw_meat_time(scores=model._scores, time_arr=time_arr, lag=model._lag)
        else:
            order, _, starts, counts, panel_arr_sorted, time_arr_sorted = _get_panel_idx(
                panel_arr=panel_arr, time_arr=time_arr
            )
            hac_meat = _nw_meat_panel(
                scores=model._scores[order],
                time_arr=time_arr_sorted,
                panel_arr=panel_arr_sorted,
                starts=starts,
                counts=counts,
                lag=model._lag,
            )
    elif model._vcov_type_detail == "DK":
        order, _, starts, _, time_arr_sorted, _ = _get_panel_idx(
            panel_arr=time_arr,
            time_arr=panel_arr,
        )
        scores_sorted = model._scores[order]
        hac_meat = _dk_meat_panel(
            scores=scores_sorted, time_arr=time_arr_sorted, idx=starts, lag=model._lag
        )
    else:
        raise ValueError(f"Unsupported HAC vcov detail: {model._vcov_type_detail}")

    meat = (
        model._tXZ @ model._tZZinv @ hac_meat @ model._tZZinv @ model._tZX
        if model._is_iv
        else hac_meat
    )
    return model._bread @ meat @ model._bread


def vcov_nid(model):
    raise NotImplementedError(
        "Only models of type Quantreg support a variance-covariance matrix of type 'nid'."
    )


def vcov_crv1(model, clustid: np.ndarray, cluster_col: np.ndarray):
    meat = model._crv1_meat_func(
        scores=model._scores.astype(np.float64),
        clustid=clustid.astype(np.uintp),
        cluster_col=cluster_col.astype(np.uintp),
    )
    meat = (
        model._tXZ @ model._tZZinv @ meat @ model._tZZinv @ model._tZX
        if model._is_iv
        else meat
    )
    return model._bread @ meat @ model._bread


def vcov_crv3_fast(model, clustid, cluster_col):
    beta_jack = np.zeros((len(clustid), model._k))

    txx = np.transpose(model._X) @ model._X
    txy = np.transpose(model._X) @ model._Y

    for ixg, g in enumerate(clustid):
        xg = model._X[np.equal(g, cluster_col)]
        yg = model._Y[np.equal(g, cluster_col)]
        tXgXg = np.transpose(xg) @ xg
        beta_jack[ixg, :] = (
            np.linalg.pinv(txx - tXgXg) @ (txy - np.transpose(xg) @ yg)
        ).flatten()

    beta_center = model._beta_hat
    vcov_mat = np.zeros((model._k, model._k))
    for ixg, _ in enumerate(clustid):
        beta_centered = beta_jack[ixg, :] - beta_center
        vcov_mat += np.outer(beta_centered, beta_centered)
    return vcov_mat


def vcov_crv3_slow(model, clustid, cluster_col):
    beta_jack = np.zeros((len(clustid), model._k))

    fixest_module = import_module("pyfixest.estimation")
    fit_ = fixest_module.feols if model._method == "feols" else fixest_module.fepois

    for ixg, g in enumerate(clustid):
        data = model._data[~np.equal(g, cluster_col)]
        fit = fit_(
            fml=model._fml,
            data=data,
            vcov="iid",
            weights=model._weights_name,
            weights_type=model._weights_type,
        )
        beta_jack[ixg, :] = fit.coef().to_numpy()

    beta_center = model._beta_hat
    vcov_mat = np.zeros((model._k, model._k))
    for ixg, _ in enumerate(clustid):
        beta_centered = beta_jack[ixg, :] - beta_center
        vcov_mat += np.outer(beta_centered, beta_centered)
    return vcov_mat
