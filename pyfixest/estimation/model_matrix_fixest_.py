import ast
import re
import warnings
from collections.abc import Mapping
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from formulaic import Formula
from formulaic.errors import FactorEvaluationError

from pyfixest.estimation.detect_singletons_ import detect_singletons
from pyfixest.estimation.FormulaParser import FixestFormula


def model_matrix_fixest(
    FixestFormula: FixestFormula,
    data: pd.DataFrame,
    drop_singletons: bool = False,
    weights: Optional[str] = None,
    drop_intercept=False,
    context: Union[int, Mapping[str, Any]] = 0,
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
    context : int or Mapping[str, Any]
        A dictionary containing additional context variables to be used by
        formulaic during the creation of the model matrix. This can include
        custom factorization functions, transformations, or any other
        variables that need to be available in the formula environment.

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
        - 'model_spec' : formulaic ModelSpec
            The model specification used to create the model matrices.

    Examples
    --------
    ```{python}
    import pyfixest as pf
    from pyfixest.estimation.model_matrix_fixest_ import model_matrix_fixest

    data = pf.get_data()
    fit = pf.feols("Y ~ X1 + f1 + f2", data=data)
    FixestFormula = fit.FixestFormula

    mm = model_matrix_fixest(FixestFormula, data)
    mm
    ```
    """
    FixestFormula.check_syntax()

    fml_second_stage = FixestFormula.fml_second_stage
    fml_first_stage = FixestFormula.fml_first_stage
    fval = FixestFormula._fval
    _check_weights(weights, data)

    fml_second_stage = _fixest_to_formulaic(fml_second_stage, data=data)
    if fml_first_stage is not None:
        fml_first_stage = _fixest_to_formulaic(fml_first_stage, data=data)

    fval, data = _fixef_interactions(fval=fval, data=data)
    _is_iv = fml_first_stage is not None

    mm = Formula(
        **{
            "fml_second_stage": fml_second_stage,
            **({"fml_first_stage": fml_first_stage} if _is_iv else {}),
            **({"fe": wrap_factorize(fval)} if fval != "0" else {}),
            **({"weights": weights} if weights is not None else {}),
        }
    ).get_model_matrix(
        data, ensure_full_rank=False, output="pandas", context={"factorize": factorize}
    )
    endogvar = Z = weights_df = fe = None

    model_spec = mm.model_spec

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

    # drop infinite values
    inf_idx_list = []
    for df in [Y, X, Z, endogvar, weights_df]:
        if df is not None:
            inf_idx = np.where(df.isin([np.inf, -np.inf]).any(axis=1))[0].tolist()
            inf_idx_list.extend(inf_idx)

    inf_idx = list(set(inf_idx_list))
    if len(inf_idx) > 0:
        warnings.warn(
            f"{len(inf_idx)} rows with infinite values detected. These rows are dropped from the model."
        )

        keep_mask = np.ones(Y.shape[0], dtype=bool)
        keep_mask[inf_idx] = False

        Y, X, Z, endogvar, weights_df, fe = _drop_rows(
            idx=keep_mask,
            X_is_empty=X_is_empty,
            Y=Y,
            fe=fe,
            X=X,
            Z=Z,
            endogvar=endogvar,
            weights_df=weights_df,
        )

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
            Y, X, Z, endogvar, weights_df, fe = _drop_rows(
                idx=keep_idx,
                X_is_empty=X_is_empty,
                Y=Y,
                fe=fe,
                X=X,
                Z=Z,
                endogvar=endogvar,
                weights_df=weights_df,
            )

    na_index = _get_na_index(data.shape[0], Y.index)
    na_index_str = ",".join(str(x) for x in na_index)

    # rename fixed effects columns wrapped in factorize()
    if fe is not None:
        fe.rename(
            columns=lambda x: x.replace("factorize(", "").replace(")", ""), inplace=True
        )

    return {
        "Y": Y,
        "X": X,
        "fe": fe,
        "endogvar": endogvar,
        "Z": Z,
        "weights_df": weights_df,
        "na_index": na_index,
        "na_index_str": na_index_str,
        "X_is_empty": X_is_empty,
        "model_spec": model_spec,
    }


def _drop_rows(
    idx: np.ndarray,
    X_is_empty: bool,
    Y: pd.DataFrame,
    fe: Optional[pd.DataFrame],
    X: pd.DataFrame,
    Z: Optional[pd.DataFrame],
    endogvar: Optional[pd.DataFrame],
    weights_df: Optional[pd.DataFrame],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
]:
    """
    Drop rows from dataframes.

    Drop rows from dataframes based on either a list of indices to drop or a
    boolean array of indices to keep.

    Parameters
    ----------
    Y : pd.DataFrame
        The dependent variable.
    X : pd.DataFrame
        The regressors.
    Z : Optional[pd.DataFrame]
        The instruments.
    endogvar : Optional[pd.DataFrame]
        The endogenous variables.
    weights_df : Optional[pd.DataFrame]
        The weights.
    fe : Optional[pd.DataFrame]
        The fixed effects.
    idx_to_drop : Optional[list[int]], optional
        A list of integer indices to drop. The default is None.
    idx_to_keep : Optional[np.ndarray], optional
        A boolean numpy array indicating which rows to keep. The default is None.

    Returns
    -------
    A tuple of the dataframes with the rows dropped.

    """
    Y = Y[idx]
    if fe is not None:
        fe = fe[idx]
    if not X_is_empty:
        X = X.iloc[idx]
    if Z is not None:
        Z = Z[idx]
    if endogvar is not None:
        endogvar = endogvar[idx]
    if weights_df is not None:
        weights_df = weights_df[idx]

    return Y, X, Z, endogvar, weights_df, fe


def _get_na_index(N: int, Y_index: pd.Index) -> np.ndarray:
    all_indices = np.arange(N)
    max_index = all_indices.max() + 1
    mask = np.ones(max_index, dtype=bool)
    Y_index_arr = Y_index.to_numpy()
    mask[Y_index_arr] = False
    na_index = np.where(mask)[0]
    return na_index


def _fixest_to_formulaic(formula_string: str, data: pd.DataFrame) -> str:
    terms = re.findall(r"(i\(.+?\))", formula_string)
    if not terms:
        return formula_string  # fixest:i() syntax not used in formula

    def is_categorical(variable: str, data: pd.DataFrame) -> bool:
        if variable not in data.columns:
            raise FactorEvaluationError(
                f"Unable to evaluate factor `{variable}`. [NameError: `{variable}` is not present in the dataset or evaluation context.]"
            )
        values = data[variable]
        # Copy of formulaic.materializers.pandas.PandasMaterializer._is_categorical
        return values.dtype in ("object", "str") or isinstance(
            values.dtype, pd.CategoricalDtype
        )

    def build_term(variable: str, reference: str | None, categorical: bool) -> str:
        if reference is None and not categorical:
            return variable  # do not encode as categorical
        if reference is None:
            contrast = ",contr.treatment(drop=False)"
        else:
            contrast = f",contr.treatment(base={reference},drop=True)"
        return f"C({variable}{contrast})"

    variable_to_reference: dict[str, str] = {"factor_var": "ref", "var": "ref2"}
    for term in terms:
        expression = ast.parse(term, mode="eval")
        args = [ast.unparse(arg) for arg in expression.body.args]
        kwargs = {kw.arg: ast.unparse(kw.value) for kw in expression.body.keywords}
        formulaic_terms: list[str] = []
        for i, (var, ref) in enumerate(variable_to_reference.items()):
            if var not in kwargs and i >= len(args):
                continue
            variable = kwargs[var] if var in kwargs else args[i]
            reference = kwargs.get(ref)
            categorical = (
                True if var == "factor_var" else is_categorical(variable, data)
            )
            formulaic_term = build_term(
                variable=variable, reference=reference, categorical=categorical
            )
            formulaic_terms.append(formulaic_term)
        formula_string = formula_string.replace(term, ":".join(formulaic_terms))
    return formula_string


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

    return bool((x[~np.isnan(x)] > 0).all())


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
