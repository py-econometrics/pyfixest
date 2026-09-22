from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t

from pyfixest.estimation.formula import FORMULAIC_TRANSFORMS
from pyfixest.estimation.formula.formulaic_compat import (
    materialize_model_spec_with_unseen_mask,
)
from pyfixest.estimation.formula.model_matrix import _ModelMatrixKey
from pyfixest.estimation.internals.literals import (
    PredictionErrorOptions,
    PredictionType,
)
from pyfixest.estimation.internals.model_state import (
    EstimationOptions,
    EstimationSample,
    FittedValues,
    VarianceCovariance,
)
from pyfixest.estimation.post_estimation.fixed_effects import (
    FixedEffectEstimates,
    check_fe_dtype_compatibility,
    predict_fixed_effects,
    warn_on_unseen_fixed_effect_levels,
)
from pyfixest.utils.dev_utils import DataFrameType, _narwhals_to_pandas


def _get_prediction_se(
    variance_covariance: VarianceCovariance, X: np.ndarray
) -> np.ndarray:
    """
    Compute prediction standard error for each row in X.

    Parameters
    ----------
    variance_covariance : VarianceCovariance
        Fitted covariance estimate.
    X : np.ndarray
        The design matrix for newdata.

    Returns
    -------
    se : np.ndarray
        The prediction standard error for each observation.
    """
    return np.sqrt(np.einsum("ij,jk,ik->i", X, variance_covariance.vcov, X))


def _compute_prediction_error(
    sample_info: EstimationSample,
    variance_covariance: VarianceCovariance,
    residuals: Callable[[], np.ndarray] | None,
    k: int,
    nobs: int,
    yhat: np.ndarray,
    X: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    """
    Fill a DataFrame with predictions and confidence intervals.

    X must have shape (nobs, k). Rows containing NaN in X produce NaN se_fit and
    confidence intervals via natural NaN propagation in the einsum.

    Parameters
    ----------
    sample_info : EstimationSample
        Sample counts of the fitted model.
    variance_covariance : VarianceCovariance
        Fitted covariance estimate.
    residuals : Callable[[], np.ndarray]
        Accessor for response residuals of the fitted model.
    k : int
        Number of estimated coefficients.
    nobs : int
        The number of rows in the prediction DataFrame.
    yhat : np.ndarray
        The predicted values.
    X : np.ndarray
        The design matrix, shape (nobs, k).
    alpha : float
        The confidence level.

    Returns
    -------
    prediction_df : pd.DataFrame
        The DataFrame with predictions, prediction SEs and confidence intervals.
    """
    columns = ["fit", "se_fit", "ci_low", "ci_high"]

    prediction_df = pd.DataFrame(np.nan, index=range(nobs), columns=columns)

    df_resid = sample_info.n_obs - k
    z_crit = t.ppf(1 - alpha / 2, df_resid)
    assert residuals is not None
    sigma2 = np.sum(residuals() ** 2) / df_resid

    prediction_df["fit"] = yhat
    prediction_df["se_fit"] = _get_prediction_se(
        variance_covariance=variance_covariance, X=X
    )
    prediction_df["ci_low"] = prediction_df["fit"] - z_crit * np.sqrt(
        prediction_df["se_fit"] ** 2 + sigma2
    )
    prediction_df["ci_high"] = prediction_df["fit"] + z_crit * np.sqrt(
        prediction_df["se_fit"] ** 2 + sigma2
    )

    return prediction_df


def _run_predict(
    *,
    newdata: DataFrameType | None,
    type: PredictionType,
    se_fit: bool | None,
    interval: PredictionErrorOptions | None,
    alpha: float,
    has_fixef: bool,
    method: str,
    options: EstimationOptions,
    sample_info: EstimationSample,
    fitted_values: FittedValues,
    variance_covariance: VarianceCovariance | None,
    model_spec: Mapping[_ModelMatrixKey, Any] | None,
    coefnames: list[str],
    beta_hat: np.ndarray,
    k: int,
    fixef_estimates: FixedEffectEstimates | None,
    recover_fixef: Callable[[], FixedEffectEstimates],
    in_sample_design: np.ndarray | None,
    residuals: Callable[[], np.ndarray] | None,
) -> np.ndarray | pd.DataFrame:
    """Predict from retained typed state and materialized formula specifications."""
    if newdata is None:
        # note: no need to worry about fixed effects, as not supported with
        # prediction errors; will throw error later;
        X = in_sample_design
        assert X is not None
        y_hat = getattr(fitted_values, type)
        n_observations = sample_info.n_rows
    else:
        assert model_spec is not None
        newdata = _narwhals_to_pandas(newdata).reset_index(drop=True)
        n_observations = newdata.shape[0]
        context = FORMULAIC_TRANSFORMS | {**options.context}
        # Use na_action="drop" on each sub-spec separately because dependent variable
        # may not be available in newdata, then intersect indices so a NaN in *any* variable
        # (covariate or FE) marks the whole row as NaN in the output.
        rhs_spec = model_spec[_ModelMatrixKey.main].rhs
        X_mm, unseen = materialize_model_spec_with_unseen_mask(
            rhs_spec, newdata, context
        )
        valid_idx = X_mm.index.to_numpy()
        # rows with a categorical level unseen during fitting (in C()/i()) would
        # be silently encoded as the reference level -> drop them to NaN instead,
        # matching how unseen fixed-effect levels are handled below.
        valid_idx = valid_idx[~unseen[valid_idx]]
        if has_fixef:
            fe_spec = model_spec[_ModelMatrixKey.fixed_effects]
            check_fe_dtype_compatibility(fe_spec, newdata)
            # na_action="ignore" keeps unseen-level rows as NaN codes
            fe_mm = fe_spec.get_model_matrix(
                newdata, context=context, na_action="ignore"
            )
            warn_on_unseen_fixed_effect_levels(fe_mm, fe_spec, newdata)
            valid_fixed_effects = fe_mm.notna().all(axis="columns").to_numpy()
            valid_idx = valid_idx[valid_fixed_effects[valid_idx]]
            if fixef_estimates is None:
                fixef_estimates = recover_fixef()
            fe_hat = predict_fixed_effects(
                model_matrix=fe_mm.loc[valid_idx],
                coefficients=fixef_estimates.coefficients,
            )

        X_coef = X_mm.loc[valid_idx, coefnames].to_numpy()
        y_hat = np.full(n_observations, np.nan)
        y_hat[valid_idx] = X_coef @ beta_hat
        if has_fixef:
            y_hat[valid_idx] += fe_hat
        # Pad X to full size; NaN rows yield NaN SE/CI via einsum propagation.
        X = np.full((n_observations, X_coef.shape[1]), np.nan)
        X[valid_idx] = X_coef
        if options.offset is not None:
            offset_mm = model_spec[_ModelMatrixKey.offset].get_model_matrix(
                newdata,
                context=context,
                na_action="drop",
                output="pandas",
            )
            if not offset_mm.index.equals(newdata.index):
                raise ValueError(
                    f"Offset expression '{options.offset}' evaluates to missing "
                    "values in `newdata`."
                )

            y_hat += offset_mm.iloc[:, 0].to_numpy()

        if type == "response" and method == "fepois":
            y_hat = np.exp(y_hat)

    if se_fit or interval == "prediction":
        assert variance_covariance is not None
        prediction_df = _compute_prediction_error(
            sample_info=sample_info,
            variance_covariance=variance_covariance,
            residuals=residuals,
            k=k,
            nobs=n_observations,
            yhat=y_hat,
            X=X,
            alpha=alpha,
        )
        if interval == "prediction":
            return prediction_df
        else:
            return prediction_df["se_fit"].to_numpy()
    else:
        return y_hat
