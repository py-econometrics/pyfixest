"""Recover the (swept-out) fixed-effects coefficients of a fitted model."""

import numpy as np
from formulaic import Formula
from scipy.sparse import diags
from scipy.sparse.linalg import lsqr

from pyfixest.utils.dev_utils import _extract_variable_level


def _fixef_impl(
    model, atol: float = 1e-06, btol: float = 1e-06
) -> dict[str, dict[str, float]]:
    "Implementation of Feols.fixef; see the method docstring for details."
    weights_sqrt = np.sqrt(model._weights).flatten()

    blocked_transforms = ["i(", "^", "poly("]
    for bt in blocked_transforms:
        if bt in model._fml:
            raise NotImplementedError(
                f"The fixef() method is currently not supported for models with '{bt}' transformations."
            )

    if not model._has_fixef:
        raise ValueError("The regression model does not have fixed effects.")

    if model._is_iv:
        raise NotImplementedError(
            "The fixef() method is currently not supported for IV models."
        )

    depvars, rhs = model._fml.split("~")
    covars, fixef_vars = rhs.split("|")

    fixef_vars_list = fixef_vars.split("+")
    fixef_vars_C = [f"C({x})" for x in fixef_vars_list]
    fixef_fml = "+".join(fixef_vars_C)

    Y, X = Formula(f"{depvars} ~ {covars}").get_model_matrix(
        model._data, output="pandas", context=model._context
    )
    Y = Y.to_numpy().flatten().astype(np.float64)
    if model._X_is_empty:
        uhat = Y.flatten()
    else:
        # drop intercept, potentially multicollinear vars
        X = X[model._coefnames].to_numpy()
        if model._method == "fepois" or model._method.startswith("feglm"):
            # determine residuals from estimated linear predictor
            # equation (5.2) in Stammann (2018) http://arxiv.org/abs/1707.01815
            Y = model._Y_hat_link
            # _Y_hat_link contains the offset as part of eta; subtract it so
            # that _sumFE represents the pure FE contribution and predict()
            # can add the offset back from newdata without double-counting.
            if model._offset_name is not None:
                assert model._offset is not None
                Y = Y - model._offset.flatten()
        uhat = (Y - X @ model._beta_hat).flatten()
    D2 = Formula("-1+" + fixef_fml).get_model_matrix(model._data, output="sparse")
    cols = D2.model_spec.column_names

    if model._has_weights:
        uhat *= weights_sqrt
        weights_diag = diags(weights_sqrt, 0)
        D2 = weights_diag.dot(D2)

    alpha = lsqr(D2, uhat, atol=atol, btol=btol)[0]

    res: dict[str, dict[str, float]] = {}
    for i, col in enumerate(cols):
        variable, level = _extract_variable_level(col)
        # check if res already has a key variable
        if variable not in res:
            res[variable] = dict()
            res[variable][level] = alpha[i]
            continue
        else:
            if level not in res[variable]:
                res[variable][level] = alpha[i]

    model._fixef_dict = res
    model._alpha = alpha
    model._sumFE = D2.dot(alpha)

    return model._fixef_dict
