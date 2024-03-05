import re
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from formulaic import Formula

from pyfixest.errors import InvalidReferenceLevelError
from pyfixest.estimation.detect_singletons_ import detect_singletons


def model_matrix_fixest(
    fml: str,
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

    This function processes the data and then calls `formulaic.Formula.get_model_matrix()`
    to create the model matrices.

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
    _check_weights(weights, data)
    _ivars = _find_ivars(fml)[0]

    if _ivars and len(_ivars) == 2 and not _is_numeric(data[_ivars[1]]):
        raise ValueError(
            f"The second variable in the i() syntax must be numeric, but it is of type {data[_ivars[1]].dtype}."
        )
    _check_i_refs2(_ivars, i_ref1, i_ref2, data)

    depvar, covar, endogvar, instruments, fval = deparse_fml(fml, i_ref1, i_ref2)
    _is_iv = True if endogvar is not None else False
    endogvar = Z = weights_df = fe = None

    fml_kwargs = {
        "Y": depvar,
        "X": covar if fval != 0 or drop_intercept else f"1+{covar}",
        **({"endog": endogvar, "instruments": instruments} if _is_iv else {}),
        **({"fe": fval_to_numeric(fval)} if fval != "0" else {}),
        **({"weights": weights} if weights is not None else {})
    }

    FML = Formula(**fml_kwargs)
    mm = FML.get_model_matrix(data, output="pandas", context = {"to_numeric": pd.to_numeric})

    for x in fml_kwargs.keys():

        if x == "Y":
            Y = mm["Y"]
        elif x == "X":
            X = mm["X"]
            # special case: sometimes it is useful to run models "Y ~ 0 | f1"
            # to demean Y + to use the predict method
            X_is_empty = False if X.shape[1] > 0 else True
        elif x == "endog":
            endogvar = mm["endog"]
        elif x == "instruments":
            Z = mm["instruments"]
        elif x == "fe":
            fe = mm["fe"]
        elif x == "weights":
            weights_df = mm["weights"]

    # make sure that all of the following are of type float64: Y, X, Z, endogvar, fe, weights_df
    for df in [Y, X, Z, endogvar, fe, weights_df]:
        if df is not None:
            cols_to_convert = df.select_dtypes(exclude=['int64', 'float64']).columns
            df[cols_to_convert] = df[cols_to_convert].astype('float64')

    # check if Y, endogvar have dimension (N, 1) - else they are non-numeric
    if Y.shape[1] > 1:
        raise TypeError(
            f"The dependent variable must be numeric, but it is of type {data[depvar].dtype}."
        )
    if endogvar is not None and endogvar.shape[1] > 1:
        raise TypeError(
            f"The endogenous variable must be numeric, but it is of type {data[endogvar].dtype}."
        )

    columns_to_drop = _get_i_refs_to_drop(_ivars, i_ref1, i_ref2, X)

    if columns_to_drop and not X_is_empty:
        X.drop(columns_to_drop, axis=1, inplace=True)
        if _is_iv:
            Z.drop(columns_to_drop, axis=1, inplace=True)

    # drop reference level, if specified
    # ivars are needed for plotting of all interacted variables via iplot()

    _icovars = _get_icovars(_ivars, X)

    # drop intercept if specified in feols() call - mostly handy for did2s()
    if drop_intercept:
        if "Intercept" in X.columns:
            X.drop("Intercept", axis=1, inplace=True)
        if _is_iv and "Intercept" in Z.columns:
            Z.drop("Intercept", axis=1, inplace=True)

    # handle singleton fixed effects

    if fe is not None and drop_singletons:
        dropped_singleton_bool = detect_singletons(fe.to_numpy())
        keep_singleton_indices = np.where(~dropped_singleton_bool)[0]
        if np.any(dropped_singleton_bool == True):  # noqa: E712
            warnings.warn(
                f"{np.sum(dropped_singleton_bool)} singleton fixed effect(s) detected. These observations are dropped from the model."
            )
            Y = Y.iloc[keep_singleton_indices]
            if not X_is_empty:
                X = X.iloc[keep_singleton_indices]
            fe = fe.iloc[keep_singleton_indices]
            if _is_iv:
                Z = Z.iloc[keep_singleton_indices]
                endogvar = endogvar.iloc[keep_singleton_indices]
            if weights_df is not None:
                weights_df = weights_df.iloc[keep_singleton_indices]


    # overwrite na_index
    na_index = list(set(range(data.shape[0])).difference(Y.index))
    na_index_str = ",".join(str(x) for x in na_index)

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


def fval_to_numeric(pattern):
    """
    Transforms a pattern of variables separated by '+' into a string where
    each variable is wrapped with to_numeric().

    Parameters:
    - pattern: A string representing the pattern of variables, e.g., "a+b+c".

    Returns:
    - A transformed string where each variable in the input pattern is wrapped
      with pd.to_numeric(), e.g., "pd.to_numeric(a) + pd.to_numeric(b) + pd.to_numeric(c)".
    """
    variables = pattern.split('+')
    transformed_variables = ['to_numeric(' + var.strip() + ')' for var in variables]
    transformed_pattern = ' + '.join(transformed_variables)

    return transformed_pattern

def deparse_fml(
    fml: str,
    i_ref1: Optional[Union[list, str, int]],
    i_ref2: Optional[Union[list, str, int]],
):

    fml = fml.replace(" ", "")
    _is_iv = _check_is_iv(fml)

    # step 1: deparse formula
    fml_parts = fml.split("|")
    depvar, covar = fml_parts[0].split("~")

    # covar to : interaction (as formulaic does not know about i() syntax
    for x in covar.split("+"):
        # id there an i() interaction?
        is_ivar = _find_ivars(x)
        # if yes:
        if is_ivar[1]:
            # if a reference level i_ref1 is set: code contrast as
            # C(var, contr.treatment(base=i_ref1[0]))
            # if no reference level is set: code contrast as C(var)
            if i_ref1:
                inner_C = f"C({_ivars[0]},contr.treatment(base={i_ref1[0]}))"
            else:
                inner_C = f"C({_ivars[0]})"

            # if there is a second variable interacted via i() syntax,
            # i.e. i(var1, var2), then code contrast as
            # C(var1, contr.treatment(base=i_ref1[0])):var2, where var2 = i_ref2[1]
            if len(_ivars) == 2:
                interact_vars = f"{inner_C}:{_ivars[1]}"
            elif len(_ivars) == 1:
                interact_vars = f"{inner_C}"
            else:
                raise ValueError(
                    "Something went wrong with the i() syntax. Please report this issue to the package author via github."
                )
            covar = covar.replace(x, interact_vars)
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
        endogvar, instruments = None, None  # noqa: F841

    return depvar, covar, endogvar, instruments, fval



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

    if i_match:
        return i_match[0].split(","), "i"
    else:
        return None, None


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


def _clean_fe(data: pd.DataFrame, fval: str) -> tuple[pd.DataFrame, list[int]]:
    """
    Clean and transform fixed effects in a DataFrame.

    This is a helper function used in `_model_matrix_fixest()`. The function converts
    the fixed effects to integers and marks fixed effects with NaNs. It's important
    to note that NaNs are not removed at this stage; this is done in
    `_model_matrix_fixest()`.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.
    fval : str
        A string describing the fixed effects, e.g., "fe1 + fe2".

    Returns
    -------
    tuple[pd.DataFrame, list[int]]
        A tuple containing two items:
        - fe (pd.DataFrame): The DataFrame with cleaned fixed effects. NaNs are
        present in this DataFrame.
        - fe_na (list[int]): A list of columns in 'fe' that contain NaN values.
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
        if fe[x].dtype != "category" and len(fe[x].unique()) == fe.shape[0]:
            raise ValueError(
                f"Fixed effect {x} has only unique values. " "This is not allowed."
            )

    fe_na = fe.isna().any(axis=1)
    fe = fe.apply(lambda x: pd.factorize(x)[0])
    fe_na = fe_na[fe_na].index.tolist()

    return fe, fe_na


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
