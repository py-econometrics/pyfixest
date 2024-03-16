from multiprocessing import Value
import re
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from formulaic import Formula
#from formulaic.errors import FactorEvaluationError

from pyfixest.errors import (
    EndogVarsAsCovarsError,
    InstrumentsAsCovarsError,
    InvalidReferenceLevelError,
)
from pyfixest.estimation.detect_singletons_ import detect_singletons
from pyfixest.estimation.FormulaParser import FixestFormula


def model_matrix_fixest(
    #fml: str,

    FixestFormula: FixestFormula,
    data: pd.DataFrame,
    drop_singletons: bool = False,
    weights: Optional[str] = None,
    drop_intercept=False,
    i_ref1: Optional[Union[list, str, int]] = None,
    i_ref2: Optional[Union[list, str, int]] = None,
) -> tuple[
    pd.DataFrame,  # Y
    pd.DataFrame,  # X
    Optional[pd.DataFrame],  # I
    Optional[pd.DataFrame],  # fe
    np.ndarray,  # na_index
    np.ndarray,  # fe_na
    str,  # na_index_str
    Optional[list[str]],  # z_names
    Optional[str],  # weights
    bool,  # has_weights,
    Optional[list[str]],  # icovars (list of variables interacted with i() syntax)
]:
    """
    Create model matrices for fixed effects estimation.

    This function processes the data and then calls
    `formulaic.Formula.get_model_matrix()` to create the model matrices.

    Parameters
    ----------
    fml : str
        A two-sided formula string using fixest formula syntax.
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
    i_ref1 : str or list
        The reference level for the first variable in the i() syntax.
    i_ref2 : str or list
        The reference level for the second variable in the i() syntax.

    Returns
    -------
    tuple
        A tuple of the following elements:
        - Y : pd.DataFrame
            A DataFrame of the dependent variable.
        - X : pd.DataFrame
            A DataFrame of the covariates. If `combine = True`, contains covariates
            and fixed effects as dummies.
        - I : Optional[pd.DataFrame]
            A DataFrame of the Instruments, None if no IV.
        - fe : Optional[pd.DataFrame]
            A DataFrame of the fixed effects, None if no fixed effects specified.
            Only applicable if `combine = False`.
        - na_index : np.array
            An array with indices of dropped columns.
        - fe_na : np.array
            An array with indices of dropped columns due to fixed effect singletons
            or NaNs in the fixed effects.
        - na_index_str : str
            na_index, but as a comma-separated string. Used for caching of demeaned
            variables.
        - z_names : Optional[list[str]]
            Names of all covariates, minus the endogenous variables,
            plus the instruments.
            None if no IV.
        - weights : Optional[str]
            Weights as a string if provided, or None if no weights, e.g., "weights".
        - has_weights : bool
            A boolean indicating whether weights are used.
        - icovars : Optional[list[str]]
            A list of interaction variables provided via `i()`. None if no interaction
            variables via `i()` provided.

    Attributes
    ----------
    list or None
        icovars - A list of interaction variables. None if no interaction variables
        via `i()` provided.
    """
    # check if weights are valid

    fml_second_stage = FixestFormula.fml_second_stage
    fml_first_stage = FixestFormula.fml_first_stage
    #depvar = FixestFormula._depvar
    covars = FixestFormula._covar
    fval = FixestFormula._fval
    instruments = FixestFormula._instruments
    endogvars = FixestFormula._endogvars
    #import pdb; pdb.set_trace()
    _check_syntax(covars, instruments, endogvars)
    _check_weights(weights, data)

    #import pdb; pdb.set_trace()
    #_ivars = [_find_ivars(x) for x in [fml_second_stage, fml_first_stage] if x is not None][0]
    #_check_ivars(_ivars, data)
    #_check_i_refs2(_ivars, i_ref1, i_ref2, data)

    #import pdb; pdb.set_trace()
    pattern = r'i\((?P<var1>\w+)(?:,(?P<var2>\w+))?(?:,ref1=(?P<ref1>\w+|\d+\.?\d*))?\)'
    fml_second_stage = re.sub(pattern, transform_i_to_C, fml_second_stage)
    fml_first_stage = re.sub(pattern, transform_i_to_C, fml_first_stage) if fml_first_stage is not None else fml_first_stage
    #columns_to_drop = _get_i_refs_to_drop(_ivars, i_ref1, i_ref2, X)

    endogvar = Z = weights_df = fe = None

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
            df[cols_to_convert] = df[cols_to_convert].astype("float64")
    if fe is not None:
        fe = fe.astype("int64")

    # check if Y, endogvar have dimension (N, 1) - else they are non-numeric
    if Y.shape[1] > 1:
        raise TypeError("The dependent variable must be numeric.")
    if endogvar is not None and endogvar.shape[1] > 1:
        raise TypeError("The endogenous variable must be numeric.")


    #if columns_to_drop and not X_is_empty:
    #    X.drop(columns_to_drop, axis=1, inplace=True)
    #    if _is_iv:
    #        Z.drop(columns_to_drop, axis=1, inplace=True)
    #_icovars = _get_icovars(_ivars, X)

    # drop intercept if specified i
    # n feols() call - mostly handy for did2s()
    if drop_intercept or fe is not None:
        if "Intercept" in X.columns:
            X.drop("Intercept", axis=1, inplace=True)
        if _is_iv and "Intercept" in Z.columns:
            Z.drop("Intercept", axis=1, inplace=True)

    # handle NaNs in fixed effects & singleton fixed effects
    if fe is not None:

        # find values where fe == -1, these are the NaNs
        # see the pd.factorize() documentation for more details
        fe_na = np.any(fe == -1, axis=1)
        keep_indices = np.where(~fe_na)[0]

        if drop_singletons:
            dropped_singleton_bool = detect_singletons(fe.to_numpy())
            keep_singleton_indices = np.where(~dropped_singleton_bool)[0]

            if np.any(dropped_singleton_bool == True):  # noqa: E712
                warnings.warn(
                    f"{np.sum(dropped_singleton_bool)} singleton fixed effect(s) detected. These observations are dropped from the model."
                )
            keep_indices = np.intersect1d(keep_indices, keep_singleton_indices)

        Y = Y.iloc[keep_indices]
        if not X_is_empty:
            X = X.iloc[keep_indices]
        fe = fe.iloc[keep_indices]
        if _is_iv:
            Z = Z.iloc[keep_indices]
            endogvar = endogvar.iloc[keep_indices]
        if weights_df is not None:
            weights_df = weights_df.iloc[keep_indices]

    # overwrite na_index
    na_index = list(set(range(data.shape[0])).difference(Y.index))
    na_index_str = ",".join(str(x) for x in na_index)

    _icovars = None

    return (
        Y,
        X,
        fe,
        endogvar,
        Z,
        weights_df,
        na_index,
        na_index_str,
        _icovars,
        X_is_empty,
    )


def _check_syntax(covars, instruments, endogvars):

    if instruments is not None:
        if any(
            element in covars.split("+") for element in endogvars.split("+")
            ):
            raise EndogVarsAsCovarsError(
                "Endogenous variables are specified as covariates in the first part of the three-part formula. This is not allowed."
                )

        if any(
            element in covars.split("+") for element in instruments.split("+")
            ):
            raise InstrumentsAsCovarsError(
                "Instruments are specified as covariates in the first part of the three-part formula. This is not allowed."
            )


def _check_ivars(_ivars, data):

    if _ivars and len(_ivars) == 2 and not _is_numeric(data[_ivars[1]]):
        raise ValueError(
            f"The second variable in the i() syntax must be numeric, but it is of type {data[_ivars[1]].dtype}."
        )

def transform_i_to_C(match):
    # Extracting the matched groups
    var1, var2, ref1 = match.group('var1'), match.group('var2'), match.group('ref1')

    # Determine transformation based on captured groups
    if var2:
        # Case: i(X1,X2) or i(X1,X2,ref1=1)
        base = f",contr.treatment(base={ref1})" if ref1 else ""
        return f"C({var1}{base}):{var2}"
    else:
        # Case: i(X1) or i(X1,ref1=1)
        base = f",contr.treatment(base={ref1})" if ref1 else ""
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


def _find_ivars(x):
    """
    Find interaction variables in i() syntax.

    Parameters
    ----------
    x : str
        A string containing the interaction variables in i() syntax.

    Returns
    -------
    list
        A list of interaction variables or None
    """
    i_match = re.findall(r"i\((.*?)\)", x)

    if len(i_match) > 1:
        raise ValueError("Only one i() interaction allowed per estimation.")

    if i_match:
        return i_match[0].split(",")
    else:
        return None


def _check_is_iv(fml):
    """
    Check if the formula contains an IV.

    Parameters
    ----------
    fml : str
        The formula string.

    Returns
    -------
    bool
        True if the formula contains an IV, False otherwise.
    """
    # check if ~ contained twice in fml
    if fml.count("~") == 1:
        _is_iv = False
    elif fml.count("~") == 2:
        _is_iv = True
    else:
        raise ValueError("The formula must contain at most two '~'.")

    return _is_iv


def _get_icovars(_ivars: list[str], X: pd.DataFrame) -> Optional[list[str]]:
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
    if _ivars is not None:
        x_names = X.columns.tolist()
        if len(_ivars) == 2:
            _icovars = [
                s
                for s in x_names
                if s.startswith("C(" + _ivars[0]) and s.endswith(_ivars[1])
            ]
        else:
            _icovars = [
                s for s in x_names if s.startswith("C(" + _ivars[0]) and s.endswith("]")
            ]
    else:
        _icovars = None

    return _icovars


def _check_i_refs2(ivars, i_ref1, i_ref2, data) -> None:
    """
    Check reference levels.

    Check if the reference level have the same type as the variable
    (if not, string matching might fail).

    Parameters
    ----------
    ivars : list
        A list of interaction variables of maximum length 2.
    i_ref1 : list
        A list of reference levels for the first variable in the i() syntax.
    i_ref2 : list
        A list of reference levels for the second variable in the i() syntax.
    data : pd.DataFrame
        The DataFrame containing the covariates.

    Returns
    -------
    None
    """
    if ivars:
        ivar1 = ivars[0]
        if len(ivars) == 2:
            ivar2 = ivars[1]

        if i_ref1:
            # check that the type of each value in i_ref1 is the same as the
            # type of the variable
            ivar1_col = data[ivar1]
            for i in i_ref1:
                if pd.api.types.is_integer_dtype(ivar1_col) and not isinstance(i, int):
                    raise InvalidReferenceLevelError(
                        f"If the first interacted variable via i() syntax, '{ivar1}', is of type 'int', the reference level {i} must also be of type 'int', but it is of type {type(i)}."
                    )
                if pd.api.types.is_float_dtype(ivar1_col) and not isinstance(i, float):
                    raise InvalidReferenceLevelError(
                        f"If the first interacted variable via i() syntax, '{ivar1}', is of type 'float', the reference level {i} must also be of type 'float', but it is of type {type(i)}."
                    )
        if i_ref2:
            ivar2_col = data[ivar2]
            for i in i_ref2:
                if pd.api.types.is_integer_dtype(ivar2_col) and not isinstance(i, int):
                    raise InvalidReferenceLevelError(
                        f"If the second interacted variable via i() syntax, '{ivar2}', is of type 'int', the reference level {i} must also be of type 'int', but it is of type {type(i)}."
                    )
                if pd.api.types.is_float_dtype(ivar2_col) and not isinstance(i, float):
                    raise InvalidReferenceLevelError(
                        f"If the second interacted variable via i() syntax, '{ivar2}', is of type 'float', the reference level {i} must also be of type 'float', but it is of type {type(i)}."
                    )


def _get_i_refs_to_drop(_ivars, i_ref1, i_ref2, X):
    """
    Identify reference levels to drop.

    Collect all variables that (still) need to be dropped as reference levels
    from the model matrix.

    Parameters
    ----------
    _ivars : list
        A list of interaction variables of maximum length 2.
    i_ref1 : list
        A list of reference levels for the first variable in the i() syntax.
    i_ref2 : list
        A list of reference levels for the second variable in the i() syntax.
    X : pd.DataFrame
        The DataFrame containing the covariates.
    """
    columns_to_drop = []

    # now drop reference levels / variables before collecting variable names
    if _ivars:
        if i_ref1:
            # different logic for one vs two interaction variables in i() due
            # to how contr.treatment() works
            if len(_ivars) == 1:
                if len(i_ref1) == 1:
                    pass  # do nothing, reference level already dropped via contr.treatment()
                else:
                    i_ref_to_drop = i_ref1[1:]
                    # collect all column names that contain the reference level
                    for x in i_ref_to_drop:
                        # decode formulaic naming conventions
                        columns_to_drop += [
                            s
                            for s in X.columns
                            if f"C({_ivars[0]},contr.treatment(base={i_ref1[0]}))[T.{x}]"
                            in s
                        ]
            elif len(_ivars) == 2:
                i_ref_to_drop = i_ref1
                for x in i_ref_to_drop:
                    # decode formulaic naming conventions
                    columns_to_drop += [
                        s
                        for s in X.columns
                        if f"C({_ivars[0]},contr.treatment(base={i_ref1[0]}))[T.{x}]:{_ivars[1]}"
                        in s
                    ]
            else:
                raise ValueError(
                    "Something went wrong with the i() syntax. Please report this issue to the package author via github."
                )
        if i_ref2:
            raise ValueError("The function argument `i_ref2` is not yet supported.")

    return columns_to_drop


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
    return pd.factorize(fe)[0]


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
