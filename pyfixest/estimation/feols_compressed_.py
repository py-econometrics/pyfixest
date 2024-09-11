from typing import Optional, Union
import pandas as pd
from pyfixest.estimation.FormulaParser import FixestFormula
from tqdm import tqdm
from pyfixest.estimation.feols_ import Feols
import polars as pl
import numpy as np

class FeolsCompressed(Feols):

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
        use_mundlak = False,
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
        self._use_mundlak = use_mundlak

    def prepare_model_matrix(self):

        "Prepare model inputs for estimation."
        super().prepare_model_matrix()

        if self._is_iv:
            raise NotImplementedError("Compression is not supported with IV regression.")

        # now run compression algos
        depvars = self._Y.columns.tolist()
        covars = self._X.columns.tolist()
        Y_polars = pl.DataFrame(pd.DataFrame(self._Y))
        X_polars = pl.DataFrame(pd.DataFrame(self._X))
        if self._fe is not None:
            fe_polars = pl.DataFrame(pd.DataFrame(self._fe))
            data_long = pl.concat(
                 [Y_polars, X_polars, fe_polars], how="horizontal"
            )

        if self._use_mundlak:
            if len(self._fval.split("+")) > 2:
                raise ValueError(
                    "The Mundlak transform is only supported for models with up to two fixed effects."
                )
            data_long, covars_updated = _mundlak_transform(
                covars=covars,
                fevars=[f"factorize({x})" for x in fval.split("+")],
                data_long=self._data,
            )
            data_long = data_long.with_columns(
                pl.lit(1).alias("Intercept")
            )

            # no fixed effects in estimation after mundlak transformation
            covars = covars_updated + ["Intercept"]
        else:
            data_long = pl.concat([Y_polars, X_polars], how="horizontal")

        compressed_dict = _regression_compression(
            depvars=depvars,
            covars=covars,
            data_long=data_long,
        )

        # overwrite Y, X
        self._Y = compressed_dict.get("Y").to_pandas()
        self._X = compressed_dict.get("X").to_pandas()
        #covars = X.columns
        self._compression_count = compressed_dict.get(
            "compression_count"
        ).to_pandas()
        self._Yprime = compressed_dict.get("Yprime").to_pandas()
        self._Yprimeprime = compressed_dict.get("Yprimeprime").to_pandas()
        self._df_compressed = compressed_dict.get("df_compressed").to_pandas()


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


    def _vcov_crv1(self):

        df_long = self._data_mundlak if self._use_mundlak else self._data_long

        X_long = self._data_mundlak.select(self._coefnames).to_numpy()
        Y_long = self._data_mundlak.select(self._depvar).to_numpy()

        yhat = X_long @ self._beta_hat
        uhat = Y_long.flatten() - yhat

        df_long = df_long.with_columns(
            [
                pl.lit(yhat).alias("yhat"),
                pl.lit(uhat).alias("uhat"),
                pl.lit(yhat + uhat).alias(
                    "yhat_g_boot_pos"
                ),  # rademacher weights = 1
                    pl.lit(yhat - uhat).alias(
                    "yhat_g_boot_neg"
                ),  # rademacher weights = -1
            ]
        )

        boot_iter = 100
        beta_boot = np.zeros((boot_iter, self._k))

        clustervar = self._clustervar
        cluster = self._data_long[clustervar]
        cluster_ids = np.sort(np.unique(cluster).astype(np.int32))
        df_long.sort(f"factorize({clustervar[0]})")

        for b in tqdm(range(boot_iter)):
            boot_df = pl.DataFrame(
                {
                    "coin_flip": np.random.randint(0, 2, size=len(cluster_ids)),
                    f"factorize({clustervar[0]})": cluster_ids,
                }
            )
            df_boot = df_long.join(
                boot_df, on=f"factorize({clustervar[0]})", how="left"
            )
            df_boot = df_boot.with_columns(
                [
                    pl.when(pl.col("coin_flip") == 1)
                    .then(pl.col("yhat_g_boot_pos"))
                    .otherwise(pl.col("yhat_g_boot_neg"))
                    .alias("yhat_boot")
                ]
            )
            comp_dict = _regression_compression(
                depvars=["yhat_boot"],
                covars=self._coefnames,
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
    data_long: pl.DataFrame,
    short: bool = False,
) -> dict:
    "Compress data for regression based on sufficient statistics."
    covars_updated = covars.copy()

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
        "X": df_compressed.select(covars_updated),
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