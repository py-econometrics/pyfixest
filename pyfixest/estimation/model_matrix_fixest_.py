import re
import time
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from formulaic import Formula

from pyfixest.estimation.detect_singletons_ import detect_singletons
from pyfixest.estimation.FormulaParser import FixestFormula


def model_matrix_fixest(
    FixestFormula: FixestFormula,
    data: pd.DataFrame,
    drop_singletons: bool = False,
    weights: Optional[str] = None,
    drop_intercept=False,
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
    tic = time.time()
    FixestFormula.check_syntax()

    fml_second_stage = FixestFormula.fml_second_stage
    fml_first_stage = FixestFormula.fml_first_stage
    fval = FixestFormula._fval
    _check_weights(weights, data)

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
    toc = time.time()
    print(f"Time to create formula object: {toc - tic}")

    tic = time.time()
    mm = FML.get_model_matrix(data, output="pandas", context={"factorize": factorize})
    toc = time.time()
    print(f"Time to get mm: {toc - tic}")

    endogvar = Z = weights_df = fe = None

    tic = time.time()
    Y = mm["fml_second_stage"]["lhs"]
    X = mm["fml_second_stage"]["rhs"]
    X_is_empty = not X.shape[1] > 0
    if _is_iv:
        endogvar = mm["fml_first_stage"]["lhs"]
        Z = mm["fml_first_stage"]["rhs"]
    if fval != "0":
        fe = mm["fe"]
        #tic1 = time.time()
        #fe_names = fe.columns
        #factorize(fe)
        #toc1 = time.time()
        #print(f"Time to factorize fixed effects: {toc1 - tic1}")
    if weights is not None:
        weights_df = mm["weights"]
    toc = time.time()
    print(f"Time to get values out of model matrix: {toc - tic}")

    #import pdb; pdb.set_trace()


    tic = time.time()
    for df in [Y, X, Z, endogvar, weights_df]:
        if df is not None:
            cols_to_convert = df.select_dtypes(exclude=["int64", "float64"]).columns
            if cols_to_convert.size > 0:
                df[cols_to_convert] = df[cols_to_convert].astype("float64")
    if fe is not None:
        fe = fe.astype("int32")
    toc = time.time()
    print(f"Time to convert data types: {toc - tic}")

    tic = time.time()
    # check if Y, endogvar have dimension (N, 1) - else they are non-numeric
    if Y.shape[1] > 1:
        raise TypeError("The dependent variable must be numeric.")
    if endogvar is not None and endogvar.shape[1] > 1:
        raise TypeError("The endogenous variable must be numeric.")

    columns_to_drop = _get_columns_to_drop_and_check_ivars(_list_of_ivars_dict, X, data)

    if columns_to_drop and not X_is_empty:
        X.drop(columns_to_drop, axis=1, inplace=True)
        if _is_iv:
            Z.drop(columns_to_drop, axis=1, inplace=True)

    _icovars = _get_icovars(_list_of_ivars_dict, X)

    # drop intercept if specified i
    # n feols() call - mostly handy for did2s()
    if drop_intercept or fe is not None:
        if "Intercept" in X.columns:
            X.drop("Intercept", axis=1, inplace=True)
        if _is_iv and "Intercept" in Z.columns:
            Z.drop("Intercept", axis=1, inplace=True)

    toc = time.time()
    print(f"Some other checks: {toc - tic}")

    # handle NaNs in fixed effects & singleton fixed effects
    if fe is not None:

        # find values where fe == -1, these are the NaNs
        # see the pd.factorize() documentation for more details
        #fe_values = fe.values  # Convert to NumPy array if not already
        #fe_na = np.any(fe_values == -1, axis=1)
        #keep_cols = ~fe_na
        #keep_indices = np.where(keep_cols)[0]

        if drop_singletons:

            tic = time.time()
            dropped_singleton_bool = detect_singletons(fe.to_numpy())

            keep_idx = ~dropped_singleton_bool

            toc = time.time()
            print(f"Time to detect singleton fixed effects: {toc - tic}")

            if np.any(dropped_singleton_bool == True):  # noqa: E712
                warnings.warn(
                    f"{np.sum(dropped_singleton_bool)} singleton fixed effect(s) detected. These observations are dropped from the model."
                )

            if not np.all(keep_idx):
                tic = time.time()
                Y = Y[keep_idx]
                if not X_is_empty:
                    X = X.iloc[keep_idx]
                fe = fe[keep_idx]
                if _is_iv:
                    Z = Z[keep_idx]
                    endogvar = endogvar[keep_idx]
                if weights_df is not None:
                    weights_df = weights_df[keep_idx]

                toc = time.time()
                print(f"Time to DROP NaNs in fixed effects & singleton fixed effects: {toc - tic}")



    tic = time.time()
    na_index = _get_na_index(data.shape[0], Y.index)
    toc = time.time()
    print(f"Time to create na_index: {toc - tic}")

    tic = time.time()
    na_index_str = ",".join(str(x) for x in na_index)
    toc = time.time()
    print(f"Time to create na_index_str: {toc - tic}")

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
        "X_is_empty": X_is_empty,
    }


def _get_na_index(N, Y_index):

    all_indices = np.arange(N)
    max_index = all_indices.max() + 1
    mask = np.ones(max_index, dtype=bool)
    if not isinstance(Y_index, np.ndarray):
        Y_index = Y_index.to_numpy()
    else:
        Y_index = Y_index
    mask[Y_index] = False
    na_index = np.nonzero(mask)[0]

    return na_index

def _get_columns_to_drop_and_check_ivars(_list_of_ivars_dict, X, data):

    columns_to_drop = []
    for _i_ref in _list_of_ivars_dict:
        if _i_ref.get("var2"):

            var1 = _i_ref.get("var1")
            var2 = _i_ref.get("var2")
            ref = _i_ref.get("ref")

            if pd.api.types.is_categorical_dtype(
                data[var2]
            ) or pd.api.types.is_object_dtype(data[var2]):
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


def _check_ivars(_ivars, data):

    if _ivars and len(_ivars) == 2 and not _is_numeric(data[_ivars[1]]):
        raise ValueError(
            f"The second variable in the i() syntax must be numeric, but it is of type {data[_ivars[1]].dtype}."
        )


def _transform_i_to_C(match):
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


def _fixef_interactions(fval, data):
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
                data[val.replace("^", "_")] = data[vars].apply(
                    lambda x: (
                        "^".join(x.dropna().astype(str)) if x.notna().all() else np.nan
                    ),
                    axis=1,
                )

    return fval.replace("^", "_"), data


def _get_ivars_dict(fml, pattern):

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
    else:
        res = None

    return res


def _get_icovars(_list_of_ivars_dict: list, X: pd.DataFrame) -> Optional[list[str]]:
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
                    [col for col in X.columns if f"C({_ivar[0]})" in col]
                )
            if len(_ivar) == 2:
                var1, var2 = _ivar
                pattern = rf"C\({var1},.*\)\[.*\]:{var2}"
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


def _is_numeric(column):
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


def _check_weights(weights, data):
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


def _is_finite_positive(x: Union[pd.DataFrame, pd.Series, np.ndarray]):
    """Check if a column is finite and positive."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.to_numpy()

    if x.any() in [np.inf, -np.inf]:
        return False
    else:
        if (x[~np.isnan(x)] > 0).all():
            return True


def factorize(fe: pd.DataFrame) -> pd.DataFrame:
    """
    Factorize fixed effects into integers.

    Parameters
    ----------
    - fe: A DataFrame of fixed effects.

    Returns
    -------
    - A DataFrame of fixed effects where each unique value is replaced by an integer.
      NaNs are not removed but set to -1.
    """

    if pd.api.types.is_integer_dtype(fe):
        return fe
    elif pd.api.types.is_float_dtype(fe):
        return fe
    else:
        if fe.dtype != "category":
            fe = fe.astype("category")
        res = fe.cat.codes
        res[res == -1] = np.nan
        return res

def wrap_factorize(pattern):
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
