from formulaic import model_matrix
import pandas as pd
import re
from typing import Optional, Tuple, List
import numpy as np


def model_matrix_fixest(
    fml: str, data: pd.DataFrame, weights: Optional[str] = None
) -> Tuple[
    pd.DataFrame,  # Y
    pd.DataFrame,  # X
    Optional[pd.DataFrame],  # I
    Optional[pd.DataFrame],  # fe
    np.ndarray,  # na_index
    np.ndarray,  # fe_na
    str,  # na_index_str
    Optional[List[str]],  # z_names
    Optional[str],  # weights
    bool,  # has_weights,
    Optional[List[str]],  # icovars (list of variables interacted with i() syntax)
]:
    """
    Create model matrices for fixed effects estimation.

    This function preprocesses the data and then calls `formulaic.model_matrix()`
    to create the model matrices.

    Args:
        fml (str): A two-sided formula string using fixest formula syntax.
        weights (str or None): Weights as a string if provided, or None if no weights, e.g., "weights".

    Returns:
        Tuple[
            pd.DataFrame,  # Y
            pd.DataFrame,  # X
            Optional[pd.DataFrame],  # I
            Optional[pd.DataFrame],  # fe
            np.array,  # na_index
            np.array,  # fe_na
            str,  # na_index_str
            Optional[List[str]],  # z_names
            Optional[str],  # weights
            bool  # has_weights
            Optional[List[str]]

        ]: A tuple of the following elements:
            - Y: A DataFrame of the dependent variable.
            - X: A DataFrame of the covariates. If `combine = True`, contains covariates and fixed effects as dummies.
            - I: A DataFrame of the Instruments, None if no IV.
            - fe: A DataFrame of the fixed effects, None if no fixed effects specified. Only applicable if `combine = False`.
            - na_index: An array with indices of dropped columns.
            - fe_na: An array with indices of dropped columns due to fixed effect singletons or NaNs in the fixed effects.
            - na_index_str: na_index, but as a comma-separated string. Used for caching of demeaned variables.
            - z_names: Names of all covariates, minus the endogenous variables, plus the instruments. None if no IV.
            - weights: Weights as a string if provided, or None if no weights, e.g., "weights".
            - has_weights: A boolean indicating whether weights are used.
            - icovars: A list of interaction variables provided via `i()`. None if no interaction variables via `i()` provided.

    Attributes:
        list or None: icovars - A list of interaction variables. None if no interaction variables via `i()` provided.
    """

    fml = fml.replace(" ", "")
    _is_iv = _check_is_iv(fml)

    _ivars = _find_ivars(fml)
    _ivars = _deparse_ivars(_ivars)
    _ivars, _drop_ref = _clean_ivars(_ivars, data)

    # step 1: deparse formula
    fml_parts = fml.split("|")
    depvar, covar = fml_parts[0].split("~")

    # covar to : interaction (as formulaic does not know about i() syntax
    for x in covar.split("+"):
        is_ivar = _find_ivars(x)
        if is_ivar[1]:
            covar = covar.replace(x, "C(" + is_ivar[0][0] + ")" + ":" + is_ivar[0][1])
            break

    if len(fml_parts) == 3:
        fval, fml_iv = fml_parts[1], fml_parts[2]
    elif len(fml_parts) == 2:
        if _is_iv:
            fval, fml_iv = "0", fml_parts[1]
        else:
            fval, fml_iv = fml_parts[1], None
    else:
        fval = "0"
        fml_iv = None

    if _is_iv:
        endogvar, instruments = fml_iv.split("~")
    else:
        endogvar, instruments = None, None

    # step 2: create formulas
    fml_exog = depvar + " ~ " + covar
    if _is_iv:
        fml_iv_full = fml_iv + "+" + covar + "-" + endogvar

    # clean fixed effects
    if fval != "0":
        fe, fe_na = _clean_fe(data, fval)
        # fml_exog += " | " + fval
    else:
        fe = None
        fe_na = None
    # fml_iv already created

    Y, X = model_matrix(fml_exog, data)
    if _is_iv:
        endogvar, Z = model_matrix(fml_iv_full, data)
    else:
        endogvar, Z = None, None

    Y, X, endogvar, Z = [
        pd.DataFrame(x) if x is not None else x for x in [Y, X, endogvar, Z]
    ]

    # check if Y, endogvar have dimension (N, 1) - else they are non-numeric
    if Y.shape[1] > 1:
        raise TypeError(
            f"The dependent variable must be numeric, but it is of type {data[depvar].dtype}."
        )
    if endogvar is not None:
        if endogvar.shape[1] > 1:
            raise TypeError(
                f"The endogenous variable must be numeric, but it is of type {data[endogvar].dtype}."
            )

    # step 3: catch NaNs (before converting to numpy arrays)
    na_index_stage2 = list(set(data.index) - set(Y.index))

    if _is_iv:
        na_index_stage1 = list(set(data.index) - set(Z.index))
        diff1 = list(set(na_index_stage1) - set(na_index_stage2))
        diff2 = list(set(na_index_stage2) - set(na_index_stage1))
        if diff1:
            Y = Y.drop(diff1, axis=0)
            X = X.drop(diff1, axis=0)
        if diff2:
            Z = Z.drop(diff2, axis=0)
            endogvar = endogvar.drop(diff2, axis=0)
        na_index = list(set(na_index_stage1 + na_index_stage2))
    else:
        na_index = na_index_stage2

    # drop variables before collecting variable names
    if _ivars is not None:
        if _drop_ref is not None:
            X = X.drop(_drop_ref, axis=1)
            if _is_iv:
                Z = Z.drop(_drop_ref, axis=1)

    # drop reference level, if specified
    if _ivars is not None:
        x_names = X.columns.tolist()
        _icovars = [
            s for s in x_names if s.startswith(_ivars[0]) and s.endswith(_ivars[1])
        ]
    else:
        _icovars = None

    if fe is not None:
        fe = fe.drop(na_index, axis=0)
        # drop intercept
        X = X.drop("Intercept", axis=1)
        # x_names.remove("Intercept")
        if _is_iv:
            Z = Z.drop("Intercept", axis=1)
        #    z_names.remove("Intercept")

        # drop NaNs in fixed effects (not yet dropped via na_index)
        fe_na_remaining = list(set(fe_na) - set(na_index))
        if fe_na_remaining:
            Y = Y.drop(fe_na_remaining, axis=0)
            X = X.drop(fe_na_remaining, axis=0)
            fe = fe.drop(fe_na_remaining, axis=0)
            if _is_iv:
                Z = Z.drop(fe_na_remaining, axis=0)
                endogvar = endogvar.drop(fe_na_remaining, axis=0)
            na_index += fe_na_remaining
            na_index = list(set(na_index))

    na_index_str = ",".join(str(x) for x in na_index)

    return Y, X, fe, endogvar, Z, na_index, na_index_str, _icovars


def _find_ivars(x):
    """
    Find interaction variables in i() syntax.
    Args:
        x (str): A string containing the interaction variables in i() syntax.
    Returns:
        list: A list of interaction variables or None
    """

    i_match = re.findall(r"i\((.*?)\)", x)
    if i_match:
        return [x.replace(" ", "") for x in i_match[0].split(",")], "i"
    else:
        return x, None


def _deparse_ivars(x):
    """
    Deparse the result of _find_ivars() into a dictionary.
    Args:
        x (list): A list of interaction variables.
    Returns:
        dict: A dictionary of interaction variables. Keys are the reference variables, values are the interaction variables.
              If no reference variable is provided, the key is None.
    """

    if x[1] is not None:
        ivars = dict()
        # check for reference by searching for "="
        i_split = x[0][-1].split("=")
        if len(i_split) > 1:
            ref = x[0][-1].split("=")[1]
            _ivars_list = x[0][:-1]
        else:
            ref = None
            _ivars_list = x[0]
        ivars[ref] = _ivars_list
    else:
        ivars = None

    return ivars


def _clean_ivars(ivars, data):
    """
    Clean variables interacted via i(X1, X2, ref = a) syntax.

    Args:
        ivars (list): The list of variables specified in the i() syntax.
        data (pandas.DataFrame): The dataframe containing the data used for the model fitting.
    Returns:
        ivars (list): The list of variables specified in the i() syntax minus the reference level
        drop_ref (str): The dropped reference level specified in the i() syntax. None if no level is dropped
    """

    if ivars is not None:
        if list(ivars.keys())[0] is not None:
            ref = list(ivars.keys())[0]
            ivars = ivars[ref]
            drop_ref = "C(" + ivars[0] + "[T." + ref + "]" + "):" + ivars[1]
        else:
            ivars = ivars[None]
            drop_ref = None

        # type checking for ivars variable
        _check_ivars(data, ivars)

    else:
        ivars = None
        drop_ref = None

    return ivars, drop_ref


def _check_ivars(data, ivars):
    """
    Checks if the variables in the i() syntax are of the correct type.
    Args:
        data (pandas.DataFrame): The dataframe containing the data used for the model fitting.
        ivars (list): The list of variables specified in the i() syntax.
    Returns:
        None
    """

    i0_type = data[ivars[0]].dtype
    i1_type = data[ivars[1]].dtype
    if not i0_type in ["category", "O"]:
        raise ValueError(
            "Column "
            + ivars[0]
            + " is not of type 'O' or 'category', which is required in the first position of i(). Instead it is of type "
            + i0_type.name
            + ". If a reference level is set, it is required that the variable in the first position of 'i()' is of type 'O' or 'category'."
        )
        if not i1_type in ["int64", "float64", "int32", "float32"]:
            raise ValueError(
                "Column "
                + ivars[1]
                + " is not of type 'int' or 'float', which is required in the second position of i(). Instead it is of type "
                + i1_type.name
                + ". If a reference level is set, iti is required that the variable in the second position of 'i()' is of type 'int' or 'float'."
            )


def _check_is_iv(fml):
    """
    Check if the formula contains an IV.
    Args:
        fml (str): The formula string.
    Returns:
        bool: True if the formula contains an IV, False otherwise.

    """
    # check if ~ contained twice in fml
    if fml.count("~") == 1:
        _is_iv = False
    elif fml.count("~") == 2:
        _is_iv = True
    else:
        raise ValueError("The formula must contain at most two '~'.")

    return _is_iv


def _clean_fe(data: pd.DataFrame, fval: str) -> Tuple[pd.DataFrame, List[int]]:
    """
    Clean and transform fixed effects in a DataFrame.

    This is a helper function used in `_model_matrix_fixest()`. The function converts
    the fixed effects to integers and marks fixed effects with NaNs. It's important
    to note that NaNs are not removed at this stage; this is done in `_model_matrix_fixest()`.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        fval (str): A string describing the fixed effects, e.g., "fe1 + fe2".

    Returns:
        Tuple[pd.DataFrame, List[int]]: A tuple containing two items:
            - fe (pd.DataFrame): The DataFrame with cleaned fixed effects. NaNs are
            present in this DataFrame.
            - fe_na (List[int]): A list of columns in 'fe' that contain NaN values.
    """

    fval_list = fval.split("+")

    # find interacted fixed effects via "^"
    interacted_fes = [x for x in fval_list if len(x.split("^")) > 1]

    for x in interacted_fes:
        vars = x.split("^")
        data[x] = data[vars].apply(
            lambda x: "^".join(x.dropna().astype(str)) if x.notna().all() else np.nan,
            axis=1,
        )

    fe = data[fval_list]

    for x in fe.columns:
        if fe[x].dtype != "category":
            if len(fe[x].unique()) == fe.shape[0]:
                raise ValueError(
                    f"Fixed effect {x} has only unique values. " "This is not allowed."
                )

    fe_na = fe.isna().any(axis=1)
    fe = fe.apply(lambda x: pd.factorize(x)[0])
    fe_na = fe_na[fe_na].index.tolist()

    return fe, fe_na
