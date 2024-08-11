import re
import time
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from formulaic import Formula

from pyfixest.estimation.detect_singletons_ import detect_singletons
from pyfixest.estimation.FormulaParser import FixestFormula


def model_matrix_fixest(
    FixestFormula: FixestFormula,
    data: pd.DataFrame,
    drop_singletons: bool = False,
    weights: Optional[str] = None,
    drop_intercept=False,
    use_compression=False,
) -> dict:
    """
    Create model matrices for fixed effects estimation.

    This function processes the data and then calls
    `formulaic.Formula.get_model_matrix()` to create the model matrices.

    Parameters
    ----------
    FixestFormula : A pyfixest.estimation.FormulaParser.FixestFormula object
        that contains information on the model formula, the formula of the first
        and second stage, dependent variable, covariates, fixed effects, endogenous
        variables (if any), and instruments (if any).
    data : pd.DataFrame
        The input DataFrame containing the data.
    drop_singletons : bool
        Whether to drop singleton fixed effects. Default is False.
    weights : str or None
        A string specifying the name of the weights column in `data`. Default is None.
    data : pd.DataFrame
        The input DataFrame containing the data.
    drop_intercept : bool
        Whether to drop the intercept from the model matrix. Default is False.
        If True, the intercept is dropped ex post from the model matrix created
        by formulaic.
    use_compression: bool
        Whether to use regression compression to estimation losslessly via
        sufficient statistics. Default is False.

    Returns
    -------
    dict
        A dictionary with the following keys and value types:
        - 'Y' : pd.DataFrame
            The dependent variable.
        - 'X' : pd.DataFrame
            The Design Matrix.
        - 'fe' : Optional[pd.DataFrame]
            The model's fixed effects. None if not applicable.
        - 'endogvar' : Optional[pd.DataFrame]
            The model's endogenous variable(s), None if not applicable.
        - 'Z' : np.ndarray
            The model's set of instruments (exogenous covariates plus instruments).
            None if not applicable.
        - 'weights_df' : Optional[pd.DataFrame]
            DataFrame containing weights, None if weights are not used.
        - 'na_index' : np.ndarray
            Array indicating rows droppled beause of NA values or singleton
            fixed effects.
        - 'na_index_str' : str
            String representation of 'na_index'.
        - '_icovars' : Optional[list[str]]
            List of variables interacted with i() syntax, None if not applicable.
        - 'X_is_empty' : bool
            Flag indicating whether X is empty.
    """
    FixestFormula.check_syntax()

    fml_second_stage = FixestFormula.fml_second_stage
    fml_first_stage = FixestFormula.fml_first_stage
    fval = FixestFormula._fval
    _check_weights(weights, data)

    if use_compression and len(fval.split("+")) > 2:
        raise ValueError(
            "Compressed estimation is only supported for models with one fixed effect."
        )

    pattern = r"i\((?P<var1>\w+)(?:,(?P<var2>\w+))?(?:,ref=(?P<ref>.*?))?\)"

    fml_all = (
        fml_second_stage
        if fml_first_stage is None
        else f"{fml_second_stage} + {fml_first_stage}"
    )

    _list_of_ivars_dict = _get_ivars_dict(fml_all, pattern)

    fml_second_stage = re.sub(pattern, _transform_i_to_C, fml_second_stage)
    fml_first_stage = (
        re.sub(pattern, _transform_i_to_C, fml_first_stage)
        if fml_first_stage is not None
        else fml_first_stage
    )

    fval, data = _fixef_interactions(fval=fval, data=data)
    _is_iv = fml_first_stage is not None

    fml_kwargs = {
        "fml_second_stage": fml_second_stage,
        **({"fml_first_stage": fml_first_stage} if _is_iv else {}),
        **({"fe": wrap_factorize(fval)} if fval != "0" else {}),
        **({"weights": weights} if weights is not None else {}),
    }

    FML = Formula(**fml_kwargs)
    mm = FML.get_model_matrix(data, output="pandas", context={"factorize": factorize})
    endogvar = Z = weights_df = fe = None

    Y = mm["fml_second_stage"]["lhs"]
    X = mm["fml_second_stage"]["rhs"]
    X_is_empty = not X.shape[1] > 0
    if _is_iv:
        endogvar = mm["fml_first_stage"]["lhs"]
        Z = mm["fml_first_stage"]["rhs"]
    if fval != "0":
        fe = mm["fe"]
    if weights is not None:
        weights_df = mm["weights"]

    for df in [Y, X, Z, endogvar, weights_df]:
        if df is not None:
            cols_to_convert = df.select_dtypes(exclude=["int64", "float64"]).columns
            if cols_to_convert.size > 0:
                df[cols_to_convert] = df[cols_to_convert].astype("float64")
    if fe is not None:
        fe = fe.astype("int32")

    # check if Y, endogvar have dimension (N, 1) - else they are non-numeric
    if Y.shape[1] > 1:
        raise TypeError("The dependent variable must be numeric.")
    if endogvar is not None and endogvar.shape[1] > 1:
        raise TypeError("The endogenous variable must be numeric.")

    columns_to_drop = _get_columns_to_drop_and_check_ivars(_list_of_ivars_dict, X, data)

    if columns_to_drop and not X_is_empty:
        X.drop(columns_to_drop, axis=1, inplace=True)
        if Z is not None:
            Z.drop(columns_to_drop, axis=1, inplace=True)

    _icovars = _get_icovars(_list_of_ivars_dict, X)

    # drop intercept if specified i
    # n feols() call - mostly handy for did2s()
    if drop_intercept or fe is not None:
        if "Intercept" in X.columns:
            X.drop("Intercept", axis=1, inplace=True)
        if Z is not None and "Intercept" in Z.columns:
            Z.drop("Intercept", axis=1, inplace=True)

    # handle NaNs in fixed effects & singleton fixed effects
    if fe is not None and drop_singletons:
        dropped_singleton_bool = detect_singletons(fe.to_numpy())

        keep_idx = ~dropped_singleton_bool

        if np.any(dropped_singleton_bool == True):  # noqa: E712
            warnings.warn(
                f"{np.sum(dropped_singleton_bool)} singleton fixed effect(s) detected. These observations are dropped from the model."
            )

        if not np.all(keep_idx):
            Y = Y[keep_idx]
            if not X_is_empty:
                X = X.iloc[keep_idx]
            fe = fe[keep_idx]
            if Z is not None:
                Z = Z[keep_idx]
            if endogvar is not None:
                endogvar = endogvar[keep_idx]
            if weights_df is not None:
                weights_df = weights_df[keep_idx]

    na_index = _get_na_index(data.shape[0], Y.index)
    na_index_str = ",".join(str(x) for x in na_index)

    if use_compression:
        tic = time.time()
        depvars = Y.columns.tolist()
        covars = X.columns.tolist()
        Y_polars = pl.DataFrame(pd.DataFrame(Y))
        X_polars = pl.DataFrame(pd.DataFrame(X))
        fe_polars = pl.DataFrame(pd.DataFrame(fe))
        if fe is not None:
            data_long = pl.concat([Y_polars, X_polars, fe_polars], how="horizontal")
        else:
            data_long = pl.concat([Y_polars, X_polars], how="horizontal")

        print("pandas to polars time: ", time.time() - tic)

        tic = time.time()
        compressed_dict = _regression_compression(
            depvars=depvars,
            covars=covars,
            data_long=data_long,
            fevars=fval.split("+") if fval != "0" else None,
        )
        print(f"Compression time: {time.time() - tic}")
        Y = compressed_dict.get("Y").to_pandas()
        X = compressed_dict.get("X").to_pandas()
        compression_count = compressed_dict.get("compression_count").to_pandas()
        Yprime = compressed_dict.get("Yprime").to_pandas()
        Yprimeprime = compressed_dict.get("Yprimeprime").to_pandas()
        df_compressed = compressed_dict.get("df_compressed").to_pandas()

    else:
        Yprime = Yprimeprime = compression_count = df_compressed = None

    return {
        "Y": Y,
        "X": X,
        "fe": fe,
        "endogvar": endogvar,
        "Z": Z,
        "weights_df": weights_df,
        "na_index": na_index,
        "na_index_str": na_index_str,
        "_icovars": _icovars,
        "X_is_empty": X_is_empty,  #
        # all values below are None if use_compression is False
        "Yprime": Yprime,
        "Yprimeprime": Yprimeprime,
        "compression_count": compression_count,
        # "compressed_fml": compressed_fml,
        "df_compressed": df_compressed,
    }


def _regression_compression2(
    depvars: list[str],
    covars: list[str],
    data_long: pl.DataFrame,
    fevars: Optional[list[str]] = None,
) -> dict:
    "Compress data for regression based on sufficient statistics."
    covars_updated = covars.copy()

    if fevars:
        # Factorize and prepare group-wise mean calculations in one go
        for fevar in fevars:
            for var in covars:
                mean_var_name = f"mean_{var}_by_{fevar}"
                covars_updated.append(mean_var_name)
                # Add mean calculation for this var by factorized fevar
                data_long = data_long.with_columns(
                    pl.col(var).mean().over(fevar).alias(mean_var_name)
                )

    # Prepare aggregation expressions
    agg_expressions = [pl.count(depvars[0]).alias("count")]
    for var in depvars:
        agg_expressions.extend(
            [
                pl.sum(var).alias(f"sum_{var}"),
                pl.col(var).pow(2).sum().alias(f"sum_{var}_sq"),
            ]
        )

    # Perform the final grouping and aggregation
    df_compressed = data_long.groupby(covars_updated).agg(agg_expressions)

    # Calculate means
    mean_expressions = [
        (pl.col(f"sum_{var}") / pl.col("count")).alias(f"mean_{var}") for var in depvars
    ]
    df_compressed = df_compressed.with_columns(mean_expressions)

    # Add Intercept if necessary
    if fevars:
        df_compressed = df_compressed.with_columns(pl.lit(1).alias("Intercept"))
        columns_updated = covars_updated + ["Intercept"]
    else:
        columns_updated = covars_updated

    # Prepare the output dictionary
    compressed_dict = {
        "Y": df_compressed.select(f"mean_{depvars[0]}"),
        "X": df_compressed.select(columns_updated),
        "compression_count": df_compressed.select("count"),
        "Yprime": df_compressed.select(f"sum_{depvars[0]}"),
        "Yprimeprime": df_compressed.select(f"sum_{depvars[0]}_sq"),
        "df_compressed": df_compressed,
    }

    return compressed_dict


def _regression_compression(
    depvars: list[str],
    covars: list[str],
    data_long: pl.DataFrame,
    fevars: Optional[list[str]] = None,
) -> dict:
    "Compress data for regression based on sufficient statistics."
    covars_updated = covars.copy()

    agg_expressions = []
    agg_expressions_fe = []

    data_long = data_long.lazy()

    # import pdb; pdb.set_trace()
    if fevars:
        fevars2 = [f"factorize({fevar})" for fevar in fevars]
        for fevar in fevars2:
            for var in covars:
                mean_var_name = f"mean_{var}_by_{fevar}"
                covars_updated.append(mean_var_name)
                # agg_expressions.append(pl.col(var).mean().alias(mean_var_name))
                agg_expressions_fe.append(
                    pl.col(var).mean().over(fevar).alias(mean_var_name)
                )

    data_long = data_long.with_columns(agg_expressions_fe)
    agg_expressions.append(pl.count(depvars[0]).alias("count"))
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

    if fevars:
        df_compressed = df_compressed.with_columns(pl.lit(1).alias("Intercept"))
        columns_updated = covars_updated + ["Intercept"]
    else:
        columns_updated = covars_updated

    df_compressed = df_compressed.collect()
    compressed_dict = {
        "Y": df_compressed.select(f"mean_{depvars[0]}"),
        "X": df_compressed.select(columns_updated),
        "compression_count": df_compressed.select("count"),
        "Yprime": df_compressed.select(f"sum_{depvars[0]}"),
        "Yprimeprime": df_compressed.select(f"sum_{depvars[0]}_sq"),
        "df_compressed": df_compressed,
    }

    return compressed_dict


def _get_na_index(N: int, Y_index: pd.Series) -> np.ndarray:
    all_indices = np.arange(N)
    max_index = all_indices.max() + 1
    mask = np.ones(max_index, dtype=bool)
    Y_index_arr = Y_index.to_numpy()
    mask[Y_index_arr] = False
    na_index = np.nonzero(mask)[0]
    return na_index


def _get_columns_to_drop_and_check_ivars(
    _list_of_ivars_dict: Union[list[dict], None], X: pd.DataFrame, data: pd.DataFrame
) -> list[str]:
    columns_to_drop = []
    if _list_of_ivars_dict:
        for _i_ref in _list_of_ivars_dict:
            if _i_ref.get("var2"):
                var1 = _i_ref.get("var1", "")
                var2 = _i_ref.get("var2", "")
                ref = _i_ref.get("ref", "")

                if pd.api.types.is_categorical_dtype(  # type: ignore
                    data[var2]  # type: ignore
                ) or pd.api.types.is_object_dtype(data[var2]):  # type: ignore
                    raise ValueError(
                        f"""
                        The second variable in the i() syntax cannot be of type "category" or "object", but
                        but it is of type {data[var2].dtype}.
                        """
                    )
                else:
                    if ref and "_" in ref:
                        ref = ref.replace("_", "")

                pattern = rf"\[T\.{ref}(?:\.0)?\]:{var2}"
                if ref:
                    for column in X.columns:
                        if var1 in column and re.search(pattern, column):
                            columns_to_drop.append(column)

    return columns_to_drop


def _check_ivars(_ivars: list[str], data: pd.DataFrame) -> None:
    if _ivars and len(_ivars) == 2 and not _is_numeric(data[_ivars[1]]):
        raise ValueError(
            f"The second variable in the i() syntax must be numeric, but it is of type {data[_ivars[1]].dtype}."
        )


def _transform_i_to_C(match: re.Match) -> str:
    # Extracting the matched groups
    var1 = match.group("var1")
    var2 = match.group("var2")
    ref = match.group("ref")

    # Determine transformation based on captured groups
    if var2:
        # Case: i(X1,X2) or i(X1,X2,ref=1)
        base = f",contr.treatment(base={ref})" if ref else ""
        return f"C({var1}{base}):{var2}"
    else:
        # Case: i(X1) or i(X1,ref=1)
        base = f",contr.treatment(base={ref})" if ref else ""
        return f"C({var1}{base})"


def _fixef_interactions(fval: str, data: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """
    Add interacted fixed effects to the input data".

    Parameters
    ----------
    fval : str
        A string describing the fixed effects, e.g., "fe1 + fe2".
    data : pd.DataFrame
        The input DataFrame containing the data.

    Returns
    -------
    pd.DataFrame
        The input DataFrame. If the fixed effects contain interactions via "^",
        the function creates new columns in the DataFrame for the interacted
        fixed effects.
    """
    if "^" in fval:
        for val in fval.split("+"):
            if "^" in val:
                vars = val.split("^")
                data[val.replace("^", "_")] = (
                    data[vars[0]]
                    .astype(pd.StringDtype())
                    .str.cat(
                        data[vars[1:]].astype(pd.StringDtype()),
                        sep="^",
                        na_rep=None,  # a row containing a missing value in any of the columns (before concatenation) will have a missing value in the result: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.cat.html
                    )
                )

    return fval.replace("^", "_"), data


def _get_ivars_dict(fml: str, pattern: str) -> Union[list[dict]]:
    matches = re.finditer(pattern, fml)

    res = []
    if matches:
        for match in matches:
            match_dict = {}
            if match.group("var1"):
                match_dict["var1"] = match.group("var1")
            if match.group("var2"):
                match_dict["var2"] = match.group("var2")
            if match.group("ref"):
                match_dict["ref"] = match.group("ref")
            res.append(match_dict)

    return res


def _get_icovars(
    _list_of_ivars_dict: Union[None, list], X: pd.DataFrame
) -> Optional[list[str]]:
    """
    Get interacted variables.

    Get all interacted variables via i() syntax. Required for plotting of all
    interacted variables via iplot().

    Parameters
    ----------
    _ivars : list
        A list of interaction variables.
    X : pd.DataFrame
        The DataFrame containing the covariates.

    Returns
    -------
    list
        A list of interacted variables or None.
    """
    if _list_of_ivars_dict:
        _ivars = [
            (
                (d.get("var1"),)
                if d.get("var2") is None
                else (d.get("var1"), d.get("var2"))
            )
            for d in _list_of_ivars_dict
        ]

        _icovars_set = set()

        for _ivar in _ivars:
            if len(_ivar) == 1:
                _icovars_set.update(
                    [col for col in X.columns if f"C({_ivar[0]}" in col]
                )
            else:
                var1, var2 = _ivar
                pattern = rf"C\({var1}(,.*)?\)\[.*\]:{var2}"
                _icovars_set.update(
                    [
                        match.group()
                        for match in (re.search(pattern, x) for x in X.columns)
                        if match
                    ]
                )

        _icovars = list(_icovars_set)

    else:
        _icovars = None

    return _icovars


def _is_numeric(column: pd.Series) -> bool:
    """
    Check if a column is numeric.

    Args:
        column: pd.Series
            A pandas Series.

    Returns
    -------
        bool
            True if the column is numeric, False otherwise.
    """
    try:
        pd.to_numeric(column)
        return True  # noqa: TRY300
    except ValueError:
        return False


def _check_weights(weights: Union[str, None], data: pd.DataFrame) -> None:
    """
    Check if valid weights are in data.

    Args:
        weights: str or None
            A string specifying the name of the weights column in `data`.
            Default is None.
        data: pd.DataFrame
            The input DataFrame containing the data.

    Returns
    -------
        None
    """
    if weights is not None:
        if weights not in data.columns:
            raise ValueError(
                f"The weights column '{weights}' is not a column in the data."
            )
        # assert that weights is numeric and has only non-negative values
        if not _is_numeric(data[weights]):
            raise ValueError(
                f"The weights column '{weights}' must be numeric, but it is of type {type(data[weights][0])}."
            )

        if not _is_finite_positive(data[weights]):
            raise ValueError(
                f"The weights column '{weights}' must have only non-negative values."
            )


def _is_finite_positive(x: Union[pd.DataFrame, pd.Series, np.ndarray]) -> bool:
    """Check if a column is finite and positive."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.to_numpy()

    if np.isinf(x).any():
        return False

    return (x[~np.isnan(x)] > 0).all()


def factorize(fe: pd.DataFrame) -> pd.DataFrame:
    """
    Factorize / Convert fixed effects into integers.

    Parameters
    ----------
    - fe: A DataFrame of fixed effects.

    Returns
    -------
    - A DataFrame of fixed effects where each unique value is replaced by an integer.
      NaNs are not removed but set to -1.
    """
    if fe.dtype != "category":
        fe = fe.astype("category")
    res = fe.cat.codes
    res[res == -1] = np.nan
    return res


def wrap_factorize(pattern: str) -> str:
    """
    Transform fixed effect formula.

    This function wraps each variable in the input pattern with pd.factorize()
    so that formulaic does not accidentally one hot encodes fixed effects
    provided as categorical: we want to keep the fixed effects in their
    input column format.

    Parameters
    ----------
    - pattern: A string representing the pattern of variables, e.g., "a+b+c".

    Returns
    -------
    - A transformed string where each variable in the input pattern is wrapped
      with pd.factorize(), e.g., "factorize(a) + pd.factorize(b)".
    """
    variables = pattern.split("+")
    transformed_variables = ["factorize(" + var.strip() + ")" for var in variables]
    transformed_pattern = " + ".join(transformed_variables)

    return transformed_pattern
