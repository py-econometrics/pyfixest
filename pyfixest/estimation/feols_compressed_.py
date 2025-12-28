import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Union

import narwhals as nw
import numpy as np
import pandas as pd
from tqdm import tqdm

from pyfixest.estimation.feols_ import Feols, PredictionErrorOptions, PredictionType
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.literals import (
    DemeanerBackendOptions,
    SolverOptions,
)
from pyfixest.utils.dev_utils import DataFrameType

logging.basicConfig(level=logging.INFO)

try:
    import polars as pl

    polars_installed = True
except ImportError:
    polars_installed = False


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
    fixef_maxiter : int
         The maximum iterations for the demeaning algorithm.
    lookup_demeaned_data : dict[str, pd.DataFrame]
        The lookup table for demeaned data.
    solver : SolverOptions
        The solver to use.
    store_data : bool
        Whether to store the data.
    copy_data : bool
        Whether to copy the data.
    lean : bool
        Whether to keep memory-heavy objects as attributes or not.
    context : int or Mapping[str, Any]
        A dictionary containing additional context variables to be used by
        formulaic during the creation of the model matrix. This can include
        custom factorization functions, transformations, or any other
        variables that need to be available in the formula environment.
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
        fixef_maxiter: int,
        lookup_demeaned_data: dict[str, pd.DataFrame],
        solver: SolverOptions = "np.linalg.solve",
        demeaner_backend: DemeanerBackendOptions = "numba",
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        context: Union[int, Mapping[str, Any]] = 0,
        sample_split_var: Optional[str] = None,
        sample_split_value: Optional[Union[str, int]] = None,
        reps: Optional[int] = None,
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
            fixef_maxiter,
            lookup_demeaned_data,
            solver,
            demeaner_backend,
            store_data,
            copy_data,
            lean,
            context,
            sample_split_var,
            sample_split_value,
        )

        if FixestFormula.fml_first_stage is not None:
            raise NotImplementedError(
                "Compression is not supported with IV regression."
            )

        self._is_iv = False
        self._support_crv3_inference = False
        self._support_iid_inference = True
        self._supports_cluster_causal_variance = False
        self._support_decomposition = False
        self._support_hac_inference = False

        if weights is not None:
            raise ValueError(
                "weights argument needs to be None. WLS not supported for compressed regression."
            )
        # if weights_type is not "fweights":
        #    raise ValueError("weights_type argument needs to be 'fweights'. WLS not supported for compressed regression.")
        self._has_weights = True
        self._reps = reps
        self._seed = seed

        # if FixestFormula._fval != "0":
        #    raise NotImplementedError(
        #        "Compression is not supported with fixed effects syntax. Please use C(var) syntax to one-hot encode fixed effects instead."
        #    )

    def prepare_model_matrix(self):
        "Prepare model inputs for estimation."
        super().prepare_model_matrix()

        # now run compression algos
        depvars = self._Y.columns.tolist()
        covars = self._X.columns.tolist()

        if polars_installed:
            Y_nw = nw.from_native(pl.from_pandas(self._Y))
            X_nw = nw.from_native(pl.from_pandas(self._X))
        else:
            logging.info(
                "Polars is not installed. Falling back to pandas. You can likely speed up the compression drastically by installing polars."
            )
            Y_nw = nw.from_native(self._Y)
            X_nw = nw.from_native(self._X)

        fevars = []
        if self._has_fixef:
            self._use_mundlak = True
            self._has_fixef = False

            fevars = self._fe.columns.tolist()
            if polars_installed:
                fe_nw = nw.from_native(pl.from_pandas(self._fe))
            else:
                fe_nw = nw.from_native(self._fe)

            data_long = nw.concat([Y_nw, X_nw, fe_nw], how="horizontal")

            if self._use_mundlak:
                if len(fevars) > 2:
                    raise NotImplementedError(
                        "The Mundlak transform is only supported for models with up to two fixed effects."
                    )

                data_long_mundlak, covars_updated = _mundlak_transform(
                    covars=covars,
                    fevars=fevars,
                    data_long=data_long,
                )

                # add intercept
                data_long_mundlak = data_long_mundlak.with_columns(
                    nw.lit(1).alias("Intercept")
                )
                data_long_mundlak = data_long_mundlak.select(
                    ["Intercept"]
                    + [col for col in data_long_mundlak.columns if col != "Intercept"]
                )

                self._coefnames = ["Intercept", *covars_updated]
                self._fe = None

        else:
            data_long = nw.concat([Y_nw, X_nw], how="horizontal")

        compressed_dict = _regression_compression(
            depvars=depvars,
            covars=self._coefnames,
            fevars=fevars,
            data_long=data_long_mundlak if self._use_mundlak else data_long,
        )

        depvar_string = depvars[0]
        self._depvar = depvar_string
        self._fml = self._fml.replace(self._depvar, f"mean_{self._depvar}")

        # overwrite Y, X, _data
        self._data_long = data_long_mundlak if self._use_mundlak else data_long
        self._Yd = compressed_dict.Y.to_pandas()
        # store compressed dependent variable before demeaning
        self._Y_untransformed = self._Yd.copy()
        self._Xd = compressed_dict.X.to_pandas()
        self._fe = compressed_dict.fe.to_pandas()
        # covars = X.columns
        self._compression_count = compressed_dict.compression_count.to_pandas()
        self._weights = self._compression_count.to_numpy()
        self._Yprime = compressed_dict.Yprime.to_pandas()
        self._Yprimeprime = compressed_dict.Yprimeprime.to_pandas()
        self._data = compressed_dict.df_compressed.to_pandas()

    def vcov(
        self,
        vcov: Union[str, dict[str, str]],
        vcov_kwargs: Optional[dict[str, Any]] = None,
        data: Optional[DataFrameType] = None,
    ):
        "Compute the variance-covariance matrix for the compressed regression."
        if self._use_mundlak and vcov in ["iid", "hetero", "HC1", "HC2", "HC3"]:
            raise NotImplementedError(
                "Only CRV1 inference is supported with the Mundlak transform."
            )

        if isinstance(vcov, dict):
            if "CRV1" in vcov:
                if vcov.get("CRV1") not in self._data_long.columns:
                    raise NotImplementedError(
                        f"The cluster variable {vcov.get('CRV1')} is not part of the model features."
                        f"To use compressed regression with clustered errors, please include the cluster variable in the model features."
                    )
            else:
                raise NotImplementedError(
                    f"The only supported clustered vcov type for compressed regression is CRV1, but {vcov} was specified."
                )

        super().vcov(vcov=vcov, vcov_kwargs=vcov_kwargs, data=data)

    def _vcov_iid(self):
        weights = self._compression_count.to_numpy()
        yprime = self._Yprime.to_numpy()
        yprimeprime = self._Yprimeprime.to_numpy()
        X = self._X / np.sqrt(weights)
        yhat = (X @ self._beta_hat).reshape(-1, 1)
        rss_g = (yhat**2) * weights - 2 * yhat * yprime + yprimeprime
        sigma2 = np.sum(rss_g) / (self._N - 1)

        return self._bread * sigma2

    def _vcov_hetero(self):
        if self._vcov_type_detail in ["HC2", "HC3"]:
            raise NotImplementedError(
                f"Only HC1 robust inference is supported, but {self._vcov_type_detail} was specified."
            )

        yprime = self._Yprime.to_numpy()
        yprimeprime = self._Yprimeprime.to_numpy()
        weights = self._compression_count.to_numpy()
        X = self._X / np.sqrt(weights)
        yhat = (X @ self._beta_hat).reshape(-1, 1)
        rss_g = (yhat**2) * weights - 2 * yhat * yprime + yprimeprime

        _meat = (X * rss_g).T @ X

        return self._bread @ _meat @ self._bread

    def _vcov_crv1(self, clustid: np.ndarray, cluster_col: np.ndarray):
        X_long = self._data_long.select(self._coefnames).to_numpy()
        Y_long = self._data_long.select(self._depvar).to_numpy()

        yhat = X_long @ self._beta_hat
        uhat = Y_long.flatten() - yhat

        data_long = self._data_long.with_columns(
            [
                nw.lit(yhat.tolist()).alias("yhat"),
                nw.lit(uhat.tolist()).alias("uhat"),
                nw.lit(yhat + uhat).alias("yhat_g_boot_pos"),  # rademacher weights = 1
                nw.lit(yhat - uhat).alias("yhat_g_boot_neg"),  # rademacher weights = -1
            ]
        )

        rng = np.random.default_rng(self._seed)

        assert self._reps is not None, "boot_iter must not be None"
        assert self._k is not None, "self._k must not be None"
        beta_boot = np.zeros((self._reps, self._k))

        cluster = data_long[self._clustervar]
        cluster_ids = np.sort(np.unique(cluster).astype(np.int32))
        data_long = data_long.with_columns(nw.col(self._clustervar[0]).cast(nw.Int32))

        for b in tqdm(range(self._reps)):
            boot_df = nw.from_native(
                {
                    "coin_flip": rng.integers(0, 2, size=len(cluster_ids)),
                    f"{self._clustervar[0]}": cluster_ids,
                }
            )

            df_boot = data_long.join(boot_df, on=f"{self._clustervar[0]}", how="left")
            df_boot = df_boot.with_columns(
                [
                    nw.when(nw.col("coin_flip") == 1)
                    .then(nw.col("yhat_g_boot_pos"))
                    .otherwise(nw.col("yhat_g_boot_neg"))
                    .alias("yhat_boot")
                ]
            )

            comp_dict = _regression_compression(
                depvars=["yhat_boot"],
                covars=self._coefnames,
                fevars=self._fe.columns.tolist(),
                data_long=df_boot,
            )

            Y: np.ndarray = comp_dict.Y.to_numpy()
            X: np.ndarray = comp_dict.X.to_numpy()
            compression_count: np.ndarray = comp_dict.compression_count.to_numpy()
            Yw = Y * np.sqrt(compression_count)
            Xw = X * np.sqrt(compression_count)

            beta_boot[b, :] = np.linalg.lstsq(Xw.T @ Xw, Xw.T @ Yw, rcond=None)[
                0
            ].flatten()

        return np.cov(beta_boot.T)

    def predict(
        self,
        newdata: Optional[DataFrameType] = None,
        atol: float = 1e-6,
        btol: float = 1e-6,
        type: PredictionType = "link",
        se_fit: Optional[bool] = False,
        interval: Optional[PredictionErrorOptions] = None,
        alpha: float = 0.05,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Compute predicted values.

        Parameters
        ----------
        newdata : Optional[DataFrameType]
            The new data. If None, makes a prediction based on the uncompressed data set.
        atol : float
            The absolute tolerance.
        btol : float
            The relative tolerance.
        type : str
            The type of prediction.
        se_fit: Optional[bool], optional
            If True, the standard error of the prediction is computed. Only feasible
            for models without fixed effects. GLMs are not supported. Defaults to False.
        interval: str, optional
            The type of interval to compute. Can be either 'prediction' or None.
        alpha: float, optional
            The alpha level for the confidence interval. Defaults to 0.05. Only
            used if interval = "prediction" is not None.

        Returns
        -------
        Union[np.ndarray, pd.DataFrame]
            Returns a pd.Dataframe with columns "fit", "se_fit" and CIs if argument "interval=prediction".
            Otherwise, returns a np.ndarray with the predicted values of the model or the prediction
            standard errors if argument "se_fit=True".
        """
        raise NotImplementedError(
            "Predictions are not supported for compressed regression."
        )


@dataclass
class _RegressionCompressionData:
    Y: nw.DataFrame
    X: nw.DataFrame
    fe: Optional[nw.DataFrame]
    compression_count: nw.DataFrame
    Yprime: Optional[nw.DataFrame]
    Yprimeprime: Optional[nw.DataFrame]
    df_compressed: nw.DataFrame


def _regression_compression(
    depvars: list[str],
    covars: list[str],
    fevars: Optional[list[str]],
    data_long: nw.DataFrame,
    short: bool = False,
) -> _RegressionCompressionData:
    "Compress data for regression based on sufficient statistics."
    covars_updated = (
        covars.copy() + fevars.copy() if fevars is not None else covars.copy()
    )

    agg_expressions = []

    data_long = data_long.lazy()

    agg_expressions.append(nw.col(depvars[0]).count().alias("count"))

    if not short:
        for var in depvars:
            agg_expressions.append(nw.sum(var).alias(f"sum_{var}"))
            agg_expressions.append((nw.col(var) ** 2).sum().alias(f"sum_{var}_sq"))

    df_compressed = data_long.group_by(covars_updated).agg(agg_expressions)

    mean_expressions = []
    for var in depvars:
        mean_expressions.append(
            (nw.col(f"sum_{var}") / nw.col("count")).alias(f"mean_{var}")
        )
    df_compressed = df_compressed.with_columns(mean_expressions)

    df_compressed = df_compressed.collect()

    return _RegressionCompressionData(
        Y=df_compressed.select(f"mean_{depvars[0]}"),
        X=df_compressed.select(covars),
        fe=df_compressed.select(fevars) if fevars is not None else None,
        compression_count=df_compressed.select("count"),
        Yprime=df_compressed.select(f"sum_{depvars[0]}") if not short else None,
        Yprimeprime=df_compressed.select(f"sum_{depvars[0]}_sq") if not short else None,
        df_compressed=df_compressed,
    )


def _mundlak_transform(
    covars: list[str], fevars: list[str], data_long: nw.DataFrame
) -> Union[nw.DataFrame, list[str]]:
    "Compute the Mundlak transformation of the data."
    covars_updated = covars.copy()
    # Factorize and prepare group-wise mean calculations in one go
    for fevar in fevars:
        for var in covars:
            mean_var_name = f"mean_{var}_by_{fevar}"
            covars_updated.append(mean_var_name)
            # Add mean calculation for this var by factorized fevar
            data_long = data_long.with_columns(
                nw.col(var).mean().over(fevar).alias(mean_var_name)
            )

    return data_long, covars_updated
