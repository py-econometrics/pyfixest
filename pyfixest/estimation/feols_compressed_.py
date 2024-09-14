from typing import Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.FormulaParser import FixestFormula


class FeolsCompressed(Feols):

    """
    Non-user-facing class for compressed regression with fixed effects.

    See the paper "You only compress once" by Wong et al (https://arxiv.org/abs/2102.11297) for
    details on regression compression.

    Parameters
    ----------
    FixestFormula : FixestFormula
        The formula object.
    data : pd.DataFrame
        The data.
    ssc_dict : dict[str, Union[str, bool]]
        The ssc dictionary.
    drop_singletons : bool
        Whether to drop columns with singleton fixed effects.
    drop_intercept : bool
        Whether to include an intercept.
    weights : Optional[str]
        The column name of the weights. None if no weights are used. For this method,
        weights needs to be None.
    weights_type : Optional[str]
        The type of weights. For this method, weights_type needs to be 'fweights'.
    collin_tol : float
        The tolerance level for collinearity.
    fixef_tol : float
        The tolerance level for the fixed effects.
    lookup_demeaned_data : dict[str, pd.DataFrame]
        The lookup table for demeaned data.
    solver : str
        The solver to use.
    store_data : bool
        Whether to store the data.
    copy_data : bool
        Whether to copy the data.
    lean : bool
        Whether to keep memory-heavy objects as attributes or not.
    reps : int
        The number of bootstrap repetitions. Default is 100. Only used for CRV1 inference, where
        a wild cluster bootstrap is used.
    seed : Optional[int]
        The seed for the random number generator. Only relevant for CRV1 inference, where a wild
        cluster bootstrap is used.
    """


    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        ssc_dict: dict[str, Union[str, bool]],
        drop_singletons: bool,
        drop_intercept: bool,
        weights: Optional[str],
        weights_type: Optional[str],
        collin_tol: float,
        fixef_tol: float,
        lookup_demeaned_data: dict[str, pd.DataFrame],
        solver: str = "np.linalg.solve",
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        reps=100,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            FixestFormula,
            data,
            ssc_dict,
            drop_singletons,
            drop_intercept,
            weights,
            weights_type,
            collin_tol,
            fixef_tol,
            lookup_demeaned_data,
            solver,
            store_data,
            copy_data,
            lean,
        )

        self._is_iv = False
        self._support_crv3_inference = False
        self._support_iid_inference = True
        self._supports_cluster_causal_variance = False
        if weights is not None:
            raise ValueError(
                "weights argument needs to be None. WLS not supported for compressed regression."
            )
        # if weights_type is not "fweights":
        #    raise ValueError("weights_type argument needs to be 'fweights'. WLS not supported for compressed regression.")
        self._has_weights = True
        self._reps = reps
        self._seed = seed

    def prepare_model_matrix(self):
        "Prepare model inputs for estimation."
        super().prepare_model_matrix()

        if self._is_iv:
            raise NotImplementedError(
                "Compression is not supported with IV regression."
            )

        # now run compression algos
        depvars = self._Y.columns.tolist()
        covars = self._X.columns.tolist()

        Y_polars = pl.DataFrame(pd.DataFrame(self._Y))
        X_polars = pl.DataFrame(pd.DataFrame(self._X))

        fevars = []
        if self._fe is not None:
            fevars = self._fe.columns.tolist()
            fe_polars = pl.DataFrame(pd.DataFrame(self._fe))
            data_long = pl.concat([Y_polars, X_polars, fe_polars], how="horizontal")
        else:
            data_long = pl.concat([Y_polars, X_polars], how="horizontal")

        if self._fe is not None:
            self._use_mundlak = True

        if self._use_mundlak:
            if len(fevars) > 2:
                raise ValueError(
                    "The Mundlak transform is only supported for models with up to two fixed effects."
                )
            data_long, covars_updated = _mundlak_transform(
                covars=covars,
                fevars=fevars,
                data_long=data_long,
            )
            data_long = data_long.with_columns(pl.lit(1).alias("Intercept"))
            data_long = data_long.select(
                ["Intercept"] + [col for col in data_long.columns if col != "Intercept"]
            )
            # no fixed effects in estimation after mundlak transformation
            covars = ["Intercept"] + covars_updated
            self._coefnames = covars
            self._fe = None
        else:
            data_long = pl.concat([Y_polars, X_polars], how="horizontal")

        compressed_dict = _regression_compression(
            depvars=depvars,
            covars=covars,
            fevars=fevars,
            data_long=data_long,
        )

        depvar_string = depvars[0]
        self._depvar = depvar_string
        self._fml = self._fml.replace(self._depvar, f"mean_{self._depvar}")

        # overwrite Y, X, _data
        self._data_long = data_long
        self._Yd = compressed_dict.get("Y").to_pandas()
        self._Xd = compressed_dict.get("X").to_pandas()
        self._fe = compressed_dict.get("fe").to_pandas()
        # covars = X.columns
        self._compression_count = compressed_dict.get("compression_count").to_pandas()
        self._weights = self._compression_count.to_numpy()
        self._Yprime = compressed_dict.get("Yprime").to_pandas()
        self._Yprimeprime = compressed_dict.get("Yprimeprime").to_pandas()
        self._data = compressed_dict.get("df_compressed").to_pandas()

    def _vcov_iid(self):
        _N = self._N
        _bread = self._bread

        weights = self._compression_count.to_numpy()
        Yprime = self._Yprime.to_numpy()
        Yprimeprime = self._Yprimeprime.to_numpy()
        X = self._X / np.sqrt(weights)
        beta_hat = self._beta_hat
        yhat = (X @ beta_hat).reshape(-1, 1)
        rss_g = (yhat**2) * weights - 2 * yhat * Yprime + Yprimeprime
        sigma2 = np.sum(rss_g) / (_N - 1)

        _vcov = _bread * sigma2

        return _vcov

    def _vcov_hetero(self):
        _vcov_type_detail = self._vcov_type_detail
        _bread = self._bread

        if _vcov_type_detail in ["HC2", "HC3"]:
            raise NotImplementedError(
                f"Only HC1 robust inference is supported, but {_vcov_type_detail} was specified."
            )

        yprime = self._Yprime.to_numpy()
        yprimeprime = self._Yprimeprime.to_numpy()
        weights = self._compression_count.to_numpy()
        X = self._X / np.sqrt(weights)
        beta_hat = self._beta_hat
        yhat = (X @ beta_hat).reshape(-1, 1)
        rss_g = (yhat**2) * weights - 2 * yhat * yprime + yprimeprime

        _meat = (X * rss_g).T @ X

        return _bread @ _meat @ _bread

    def _vcov_crv1(self, clustid: np.ndarray, cluster_col: np.ndarray):
        _data_long_pl = self._data_long
        if "Intercept" in self._coefnames and "Intercept" not in _data_long_pl.columns:
            _data_long_pl = _data_long_pl.with_columns([pl.lit(1).alias("Intercept")])
            _data_long_pl = _data_long_pl.select(
                ["Intercept"] + _data_long_pl.columns[:-1]
            )

        X_long = _data_long_pl.select(self._coefnames).to_numpy()
        Y_long = _data_long_pl.select(self._depvar).to_numpy()

        yhat = X_long @ self._beta_hat
        uhat = Y_long.flatten() - yhat

        _data_long_pl = _data_long_pl.with_columns(
            [
                pl.lit(yhat).alias("yhat"),
                pl.lit(uhat).alias("uhat"),
                pl.lit(yhat + uhat).alias("yhat_g_boot_pos"),  # rademacher weights = 1
                pl.lit(yhat - uhat).alias("yhat_g_boot_neg"),  # rademacher weights = -1
            ]
        )

        boot_iter = self._reps
        rng = np.random.default_rng(self._seed)
        beta_boot = np.zeros((boot_iter, self._k))

        clustervar = self._clustervar
        cluster = _data_long_pl[clustervar]
        cluster_ids = np.sort(np.unique(cluster).astype(np.int32))
        _data_long_pl = _data_long_pl.with_columns(pl.col(clustervar[0]).cast(pl.Int32))
        # _data_long_pl.sort(f"({clustervar[0]})")

        for b in tqdm(range(boot_iter)):
            boot_df = pl.DataFrame(
                {
                    "coin_flip": rng.integers(0, 2, size=len(cluster_ids)),
                    f"{clustervar[0]}": cluster_ids,
                }
            )

            df_boot = _data_long_pl.join(boot_df, on=f"{clustervar[0]}", how="left")
            df_boot = df_boot.with_columns(
                [
                    pl.when(pl.col("coin_flip") == 1)
                    .then(pl.col("yhat_g_boot_pos"))
                    .otherwise(pl.col("yhat_g_boot_neg"))
                    .alias("yhat_boot")
                ]
            )

            # model_matrix needs to be called here

            comp_dict = _regression_compression(
                depvars=["yhat_boot"],
                covars=self._coefnames,
                fevars=self._fe.columns.tolist(),
                data_long=df_boot,
            )
            Y = comp_dict.get("Y").to_numpy()
            X = comp_dict.get("X").to_numpy()
            compression_count = comp_dict.get("compression_count").to_numpy()
            Yw = Y * np.sqrt(compression_count)
            Xw = X * np.sqrt(compression_count)

            beta_boot[b, :] = np.linalg.lstsq(Xw.T @ Xw, Xw.T @ Yw, rcond=None)[
                0
            ].flatten()

        return np.cov(beta_boot.T)


def _regression_compression(
    depvars: list[str],
    covars: list[str],
    fevars: Optional[list[str]],
    data_long: pl.DataFrame,
    short: bool = False,
) -> dict:
    "Compress data for regression based on sufficient statistics."
    covars_updated = (
        covars.copy() + fevars.copy() if fevars is not None else covars.copy()
    )

    agg_expressions = []

    data_long = data_long.lazy()

    agg_expressions.append(pl.count(depvars[0]).alias("count"))

    if not short:
        for var in depvars:
            agg_expressions.append(pl.sum(var).alias(f"sum_{var}"))
            agg_expressions.append(pl.col(var).pow(2).sum().alias(f"sum_{var}_sq"))

    df_compressed = data_long.group_by(covars_updated).agg(agg_expressions)

    mean_expressions = []
    for var in depvars:
        mean_expressions.append(
            (pl.col(f"sum_{var}") / pl.col("count")).alias(f"mean_{var}")
        )
    df_compressed = df_compressed.with_columns(mean_expressions)

    df_compressed = df_compressed.collect()
    compressed_dict = {
        "Y": df_compressed.select(f"mean_{depvars[0]}"),
        "X": df_compressed.select(covars),
        "fe": df_compressed.select(fevars) if fevars is not None else None,
        "compression_count": df_compressed.select("count"),
        "Yprime": df_compressed.select(f"sum_{depvars[0]}") if not short else None,
        "Yprimeprime": df_compressed.select(f"sum_{depvars[0]}_sq")
        if not short
        else None,
        "df_compressed": df_compressed,
    }

    return compressed_dict


def _mundlak_transform(
    covars: list[str], fevars: list[str], data_long: pl.DataFrame
) -> Union[pl.DataFrame, list[str]]:
    "Compute the Mundlak transformation of the data."
    covars_updated = covars.copy()
    # Factorize and prepare group-wise mean calculations in one go
    for fevar in fevars:
        for var in covars:
            mean_var_name = f"mean_{var}_by_{fevar}"
            covars_updated.append(mean_var_name)
            # Add mean calculation for this var by factorized fevar
            data_long = data_long.with_columns(
                pl.col(var).mean().over(fevar).alias(mean_var_name)
            )

    return data_long, covars_updated
