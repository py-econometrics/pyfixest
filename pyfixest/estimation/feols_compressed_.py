from typing import Optional, Union
import pandas as pd
from pyfixest.estimation.FormulaParser import FixestFormula

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

    def prepare_model_matrix(self):

        "Prepare model inputs for estimation."
        super().prepare_model_matrix()

        # now run compression algos
        depvars = self._Y.columns.tolist()
        covars = self._X.columns.tolist()
        Y_polars = pl.DataFrame(pd.DataFrame(self._Y))
        X_polars = pl.DataFrame(pd.DataFrame(self._X))
        if fe is not None:
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
        covars = X.columns
        self._compression_count = compressed_dict.get(
            "compression_count"
        ).to_pandas()
        self._Yprime = compressed_dict.get("Yprime").to_pandas()
        self._Yprimeprime = compressed_dict.get("Yprimeprime").to_pandas()
        self._df_compressed = compressed_dict.get("df_compressed").to_pandas()








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