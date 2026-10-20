"""Deprecated fitted-model attributes of pyfixest 0.60.

pyfixest 0.70 moved the loose private attributes of fitted models into typed
state such as `fit.model`, `fit.coeftable`, or `fit.variance_covariance`.
`DEPRECATED_ATTRIBUTES` lists every 0.60 name whose state has moved, and
`DeprecatedAttributes` turns each row into a class-level data descriptor on the
model classes it names:

- an `Alias` returns the exact 0.60 value, rebuilt from the typed state, and
  emits a `PyfixestDeprecationWarning` naming the replacement;
- a `Removed` entry raises `AttributeError`, explaining why the 0.60 value
  cannot be rebuilt and naming the replacement.

A row maps model-class names to entries. A subclass inherits the entry of its
nearest listed ancestor, so a row names `Feols` for the common case and a
subclass only where its 0.60 value meant something else. Rows marked
`silent=True` are still read by released versions of marginaleffects and
maketables; they return their value without a warning until those packages
read the typed state.

To deprecate another attribute, add one row to `DEPRECATED_ATTRIBUTES`; the
docs migration page and the tests read the same table.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyfixest.errors import PyfixestDeprecationWarning

# The release that moved the attributes into typed state.
REMOVED_IN = "0.70.0"


@dataclass(frozen=True, slots=True, kw_only=True)
class Alias:
    """A deprecated name whose 0.60 value is rebuilt from typed state.

    Parameters
    ----------
    replacement : str
        The typed state to use instead, as users write it, e.g.
        ``"fit.coeftable.se"``.
    read : Callable[[Any], Any]
        Rebuild the 0.60 value from a fitted model. It raises
        `AttributeError` while the typed state it reads is not published or
        was dropped by the storage options.
    """

    replacement: str
    read: Callable[[Any], Any]

    def resolve(self, name: str, fit: object) -> Any:
        """Return the 0.60 value of `fit.<name>`."""
        try:
            return self.read(fit)
        except AttributeError as exc:
            # Keep `hasattr` and `require_retained` working: state that is not
            # published yet, or was dropped by `lean=True`, stays missing.
            raise AttributeError(
                f"`fit.{name}` is not available on this fit: {exc}. "
                f"`fit.{name}` is deprecated; use `{self.replacement}` instead."
            ) from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class Removed:
    """A deprecated name whose 0.60 value cannot be rebuilt exactly.

    Parameters
    ----------
    replacement : str
        The typed state or method to use instead.
    reason : str
        Why the 0.60 value is gone, completing "was removed because ...".
    """

    replacement: str
    reason: str

    def resolve(self, name: str, fit: object) -> Any:
        """Explain why `fit.<name>` is gone and what replaces it."""
        raise AttributeError(
            f"Sorry, `fit.{name}` was removed in pyfixest {REMOVED_IN} because "
            f"{self.reason}; use `{self.replacement}` instead."
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DeprecatedAttribute:
    """One deprecated 0.60 attribute of fitted models.

    Parameters
    ----------
    name : str
        The 0.60 attribute name.
    classes : Mapping[str, Alias | Removed]
        Entries keyed by the name of the model class that defines them;
        subclasses inherit the entry of their nearest listed ancestor.
    silent : bool
        Return the value without a warning, for names that released
        third-party packages still read.
    """

    name: str
    classes: Mapping[str, Alias | Removed]
    silent: bool = False


class _DeprecatedAttributeDescriptor:
    """Class-level data descriptor serving one deprecated attribute."""

    def __init__(self, *, name: str, entry: Alias | Removed, silent: bool) -> None:
        self.name = name
        self.entry = entry
        self.silent = silent

    def __get__(self, fit: object, owner: type | None = None) -> Any:
        if fit is None:
            return self
        value = self.entry.resolve(self.name, fit)
        if not self.silent:
            warnings.warn(
                f"`fit.{self.name}` is deprecated and will be removed in a future "
                f"release; use `{self.entry.replacement}` instead.",
                PyfixestDeprecationWarning,
                stacklevel=2,
            )
        return value

    def __set__(self, fit: object, value: Any) -> None:
        raise AttributeError(self._read_only_message())

    def __delete__(self, fit: object) -> None:
        raise AttributeError(self._read_only_message())

    def _read_only_message(self) -> str:
        return (
            f"`fit.{self.name}` is a read-only deprecated attribute; the fitted "
            f"state now lives in `{self.entry.replacement}`."
        )


class DeprecatedAttributes:
    """Mixin installing the `DEPRECATED_ATTRIBUTES` rows that name a subclass."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for attribute in DEPRECATED_ATTRIBUTES:
            entry = attribute.classes.get(cls.__name__)
            if entry is None:
                continue
            if attribute.name in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__} defines {attribute.name!r}, which is also a "
                    "deprecated attribute in DEPRECATED_ATTRIBUTES."
                )
            setattr(
                cls,
                attribute.name,
                _DeprecatedAttributeDescriptor(
                    name=attribute.name, entry=entry, silent=attribute.silent
                ),
            )


# ---------------------------------------------------------------------------
# Rebuilding 0.60 values.
#
# pyfixest 0.60 stored the OLS and IV arrays premultiplied by sqrt(w), and the
# GLM arrays premultiplied by the square root of the final IRLS weights.
# ---------------------------------------------------------------------------


def _sqrt_weighted(fit: Any, values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Premultiply rows by the square root of the observation weights."""
    weights = fit.observation_weights.values
    if weights is None:
        return values
    sqrt_weights = np.sqrt(weights)
    return values * (sqrt_weights if values.ndim == 1 else sqrt_weights[:, None])


def _sqrt_working_weighted(
    fit: Any, values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Premultiply rows by the square root of the final IRLS weights."""
    sqrt_weights = np.sqrt(fit.working_state.working_weights.reshape(-1))
    return values * (sqrt_weights if values.ndim == 1 else sqrt_weights[:, None])


def _linear_weights(fit: Any) -> NDArray[np.float64]:
    """0.60 `_weights`: the observation weights as a column, ones if unweighted."""
    weights = fit.observation_weights.values
    if weights is None:
        return np.ones((fit.sample_info.n_rows, 1))
    return weights.reshape((-1, 1))


def _glm_sum_fe(fit: Any) -> NDArray[np.float64]:
    """0.60 GLM `_sumFE`: weighted fits scaled it by sqrt(IRLS weights)."""
    sum_fe = fit.fixef_estimates.sumFE
    if not fit.options.has_weights:
        return sum_fe
    return _sqrt_working_weighted(fit, sum_fe)


def _clustered_ssc(fit: Any) -> NDArray[np.float64]:
    """0.60 `_ssc`: one-way clustered fits stored a (1, 1) array."""
    covariance = fit.variance_covariance
    if len(covariance.spec.clustervar) == 1:
        return covariance.ssc.reshape((1, 1))
    return covariance.ssc


def _cluster_counts(fit: Any) -> list[int]:
    """0.60 `_G`: the `G_df="min"` rule padded the minimum to three entries."""
    counts = list(fit.variance_covariance.G)
    if counts and fit.options.ssc.G_df == "min":
        return [min(counts)] * 3
    return counts


def _sample_split_value(fit: Any) -> str | int | float:
    """0.60 `_sample_split_value`; unsplit and full-sample fits held a sentinel."""
    split = fit.model.sample_split
    if split is None or split.value is None:
        raise AttributeError("this fit is not estimated on one value of a split")
    return split.value


def _eff_f(fit: Any) -> float:
    """0.60 `_eff_F`, set only once `IV_Diag()` or `eff_F()` ran."""
    eff_f = fit.first_stage.diagnostics.eff_f
    if eff_f is None:
        raise AttributeError("run `IV_Diag()` or `eff_F()` first")
    return eff_f


def _event_study_design(fit: Any) -> Any:
    """Return the design of a saturated event study; other fits carry none."""
    try:
        return fit.event_study_design
    except AttributeError:
        raise AttributeError(
            "only `event_study(estimator='saturated')` fits record their design"
        ) from None


def _collin_index(collinearity: Any) -> list[bool]:
    """0.60 `_collin_index`: empty unless a column was dropped."""
    return list(collinearity.mask) if collinearity.any_dropped else []


def _poisson_null_measure(fit: Any, value: float) -> float | None:
    """0.60 left the Poisson null measures `None` for weighted fits."""
    return None if fit.options.has_weights else value


_NO_SANDWICH_Q = (
    "quantile regression fits carry no sandwich components; compute them from "
    "`fit.within_data`"
)
_IV_PROJECTED = (
    "IV fits now score the first-stage projection of the design, as fixest "
    "does, instead of the instruments"
)
_NO_CROSS_PRODUCTS = (
    "the solver's weighted cross products are no longer retained; compute them "
    "from `fit.within_data` and `fit.observation_weights`"
)


def _row(name: str, *, silent: bool = False, **classes: Alias | Removed):
    return DeprecatedAttribute(name=name, classes=classes, silent=silent)


def _alias(replacement: str, read: Callable[[Any], Any]) -> Alias:
    return Alias(replacement=replacement, read=read)


def _removed(replacement: str, reason: str) -> Removed:
    return Removed(replacement=replacement, reason=reason)


# Keep rows grouped by the typed state that replaces them.
DEPRECATED_ATTRIBUTES: tuple[DeprecatedAttribute, ...] = (
    # fit.model
    _row(
        "_fml",
        silent=True,
        Feols=_alias("fit.model.formula", lambda fit: fit.model.formula),
    ),
    _row(
        "_method",
        silent=True,
        Feols=_alias("fit.model.method", lambda fit: fit.model.method),
    ),
    _row("_is_iv", Feols=_alias("fit.model.is_iv", lambda fit: fit.model.is_iv)),
    _row(
        "_model_name",
        Feols=_alias("fit.model.model_name", lambda fit: fit.model.model_name),
    ),
    _row(
        "_fixef",
        silent=True,
        Feols=_alias("fit.model.fixef", lambda fit: fit.model.fixef or None),
    ),
    _row(
        "_has_fixef",
        silent=True,
        Feols=_alias("fit.model.has_fixef", lambda fit: fit.model.has_fixef),
    ),
    _row(
        "_icovars",
        Feols=_alias(
            "fit.model.interacted_covariates",
            lambda fit: list(fit.model.interacted_covariates) or None,
        ),
    ),
    _row(
        "_depvar",
        silent=True,
        Feols=_alias("fit.model.depvar", lambda fit: fit.model.depvar),
    ),
    _row(
        "_sample_split_value",
        Feols=_alias("fit.model.sample_split.value", _sample_split_value),
    ),
    # fit.options
    _row(
        "_ssc_dict",
        Feols=_alias(
            "fit.options.ssc", lambda fit: dataclasses.asdict(fit.options.ssc)
        ),
    ),
    _row(
        "_collin_tol",
        Feols=_alias("fit.options.collin_tol", lambda fit: fit.options.collin_tol),
    ),
    _row(
        "_solver",
        Feols=_alias("fit.options.solver", lambda fit: fit.options.solver),
    ),
    _row(
        "maxiter",
        Feglm=_alias("fit.options.maxiter", lambda fit: fit.options.maxiter),
    ),
    _row("tol", Feglm=_alias("fit.options.tol", lambda fit: fit.options.tol)),
    _row(
        "_quantile",
        Quantreg=_alias("fit.options.quantile", lambda fit: fit.options.quantile),
    ),
    # fit.sample_info, fit.observation_weights, fit.model_matrix
    _row(
        "_N",
        silent=True,
        Feols=_alias("fit.sample_info.n_obs", lambda fit: fit.sample_info.n_obs),
    ),
    _row(
        "_weights",
        Feols=_alias("fit.observation_weights.values", _linear_weights),
        Feglm=_alias(
            "fit.working_state.working_weights",
            lambda fit: fit.working_state.working_weights.reshape((-1, 1)),
        ),
    ),
    _row(
        "_fe",
        Feols=_removed(
            "fit.model_matrix.fixed_effects",
            "the fixed-effect codes are now numbered by sorted level instead of "
            "by first appearance, and GLM fits no longer hold them as an array",
        ),
    ),
    _row(
        "na_index",
        Feols=_removed(
            "fit.sample_info.dropped_row_index",
            "it only listed the rows a GLM dropped for separation, by position "
            "in the already filtered data",
        ),
    ),
    # fit.within_data and fit.working_state
    _row(
        "_Y",
        Feols=_alias(
            "fit.within_data.response",
            lambda fit: _sqrt_weighted(fit, fit.within_data.response),
        ),
        Feglm=_alias(
            "fit.working_state.working_response_within",
            lambda fit: _sqrt_working_weighted(
                fit, fit.working_state.working_response_within
            ),
        ),
        Fepois=_alias(
            "fit.working_state.working_response_within",
            lambda fit: _sqrt_working_weighted(
                fit, fit.working_state.working_response_within
            ).reshape((-1, 1)),
        ),
        Quantreg=_alias(
            "fit.within_data.response", lambda fit: fit.within_data.response
        ),
    ),
    _row(
        "_X",
        Feols=_alias(
            "fit.within_data.design",
            lambda fit: _sqrt_weighted(fit, fit.within_data.design),
        ),
        Feglm=_alias(
            "fit.working_state.design_within",
            lambda fit: _sqrt_working_weighted(fit, fit.working_state.design_within),
        ),
        Quantreg=_alias("fit.within_data.design", lambda fit: fit.within_data.design),
    ),
    _row(
        "_Z",
        Feols=_alias(
            "fit.within_data.design",
            lambda fit: _sqrt_weighted(fit, fit.within_data.design),
        ),
        Feiv=_alias(
            "fit.within_data.instruments",
            lambda fit: _sqrt_weighted(fit, fit.within_data.instruments),
        ),
        Feglm=_alias(
            "fit.working_state.design_within",
            lambda fit: _sqrt_working_weighted(fit, fit.working_state.design_within),
        ),
        Quantreg=_alias("fit.within_data.design", lambda fit: fit.within_data.design),
    ),
    _row(
        "_Yd",
        Feols=_removed(
            "fit.model_matrix.dependent",
            "it held the demeaned response before the collinearity check",
        ),
    ),
    _row(
        "_Xd",
        Feols=_removed(
            "fit.model_matrix.independent",
            "it held the demeaned design before the collinearity check",
        ),
    ),
    _row(
        "_Zd",
        Feiv=_removed(
            "fit.model_matrix.instruments",
            "it held the demeaned instruments before the collinearity check",
        ),
    ),
    _row(
        "_endogvard",
        Feiv=_removed(
            "fit.model_matrix.endogenous",
            "it held the demeaned endogenous variables before the collinearity check",
        ),
    ),
    _row(
        "_u_hat",
        Feglm=_alias(
            "fit.working_state.working_residuals",
            lambda fit: _sqrt_working_weighted(
                fit, fit.working_state.working_residuals.reshape(-1)
            ),
        ),
    ),
    # fit.fitted_values
    _row(
        "_Y_hat_link",
        Feols=_alias("fit.fitted_values.link", lambda fit: fit.fitted_values.link),
    ),
    _row(
        "_Y_hat_response",
        Feols=_alias(
            "fit.fitted_values.response", lambda fit: fit.fitted_values.response
        ),
    ),
    # fit.sandwich
    _row(
        "_scores",
        Feols=_alias("fit.sandwich.scores", lambda fit: fit.sandwich.scores),
        Feiv=_removed("fit.sandwich.scores", _IV_PROJECTED),
        Quantreg=_removed("fit.variance_covariance", _NO_SANDWICH_Q),
    ),
    _row(
        "_hessian",
        Feols=_alias("fit.sandwich.hessian", lambda fit: fit.sandwich.hessian),
        Feiv=_removed("fit.sandwich.hessian", _IV_PROJECTED),
        Quantreg=_removed("fit.variance_covariance", _NO_SANDWICH_Q),
    ),
    _row(
        "_bread",
        Feols=_alias("fit.sandwich.bread", lambda fit: fit.sandwich.bread),
        Quantreg=_removed("fit.variance_covariance", _NO_SANDWICH_Q),
    ),
    _row(
        "_tZX",
        Feols=_alias("fit.sandwich.hessian", lambda fit: fit.sandwich.hessian),
        Feiv=_removed("fit.sandwich.hessian", _NO_CROSS_PRODUCTS),
        Quantreg=_removed("fit.variance_covariance", _NO_SANDWICH_Q),
    ),
    _row("_tXZ", Feols=_removed("fit.sandwich", _NO_CROSS_PRODUCTS)),
    _row("_tZy", Feols=_removed("fit.sandwich", _NO_CROSS_PRODUCTS)),
    _row("_tZZinv", Feols=_removed("fit.sandwich", _NO_CROSS_PRODUCTS)),
    # fit.variance_covariance
    _row(
        "_vcov",
        silent=True,
        Feols=_alias(
            "fit.variance_covariance.vcov", lambda fit: fit.variance_covariance.vcov
        ),
    ),
    _row(
        "_vcov_type",
        silent=True,
        Feols=_alias(
            "fit.variance_covariance.spec.vcov_type",
            lambda fit: fit.variance_covariance.spec.vcov_type,
        ),
    ),
    _row(
        "_vcov_type_detail",
        Feols=_alias(
            "fit.variance_covariance.spec.vcov_type_detail",
            lambda fit: fit.variance_covariance.spec.vcov_type_detail,
        ),
    ),
    _row(
        "_is_clustered",
        Feols=_alias(
            "fit.variance_covariance.spec.is_clustered",
            lambda fit: fit.variance_covariance.spec.is_clustered,
        ),
    ),
    _row(
        "_clustervar",
        silent=True,
        Feols=_alias(
            "fit.variance_covariance.spec.clustervar",
            lambda fit: list(fit.variance_covariance.spec.clustervar) or None,
        ),
    ),
    _row(
        "_G",
        silent=True,
        Feols=_alias("fit.variance_covariance.G", _cluster_counts),
    ),
    _row("_ssc", Feols=_alias("fit.variance_covariance.ssc", _clustered_ssc)),
    _row(
        "_df_k",
        Feols=_alias(
            "fit.variance_covariance.df_k", lambda fit: fit.variance_covariance.df_k
        ),
    ),
    _row(
        "_df_t",
        Feols=_alias(
            "fit.variance_covariance.df_t", lambda fit: fit.variance_covariance.df_t
        ),
    ),
    # fit.coeftable
    _row("_se", Feols=_alias("fit.coeftable.se", lambda fit: fit.coeftable.se)),
    _row(
        "_tstat", Feols=_alias("fit.coeftable.tstat", lambda fit: fit.coeftable.tstat)
    ),
    _row(
        "_pvalue",
        silent=True,
        Feols=_alias("fit.coeftable.pvalue", lambda fit: fit.coeftable.pvalue),
    ),
    _row(
        "_conf_int",
        Feols=_alias("fit.coeftable.conf_int", lambda fit: fit.coeftable.conf_int),
    ),
    # fit.fitstat
    _row(
        "_rmse",
        silent=True,
        Feols=_alias("fit.fitstat.rmse", lambda fit: fit.fitstat.rmse),
    ),
    _row(
        "_r2", silent=True, Feols=_alias("fit.fitstat.r2", lambda fit: fit.fitstat.r2)
    ),
    _row(
        "_r2_within",
        silent=True,
        Feols=_alias("fit.fitstat.r2_within", lambda fit: fit.fitstat.r2_within),
    ),
    _row("_adj_r2", Feols=_alias("fit.fitstat.adj_r2", lambda fit: fit.fitstat.adj_r2)),
    _row(
        "_adj_r2_within",
        silent=True,
        Feols=_alias(
            "fit.fitstat.adj_r2_within", lambda fit: fit.fitstat.adj_r2_within
        ),
    ),
    _row(
        "deviance",
        silent=True,
        # 0.60 left the deviance `None` for linear and quantile fits.
        Feols=_alias("fit.fitstat.deviance", lambda fit: None),
        Feglm=_alias("fit.fitstat.deviance", lambda fit: fit.fitstat.deviance),
    ),
    _row(
        "_loglik",
        Fepois=_alias("fit.fitstat.loglik", lambda fit: fit.fitstat.loglik),
    ),
    _row(
        "_loglik_null",
        Fepois=_alias(
            "fit.fitstat.loglik_null",
            lambda fit: _poisson_null_measure(fit, fit.fitstat.loglik_null),
        ),
    ),
    _row(
        "_pseudo_r2",
        Fepois=_alias(
            "fit.fitstat.pseudo_r2",
            lambda fit: _poisson_null_measure(fit, fit.fitstat.pseudo_r2),
        ),
    ),
    # fit.wald
    _row("_dfn", Feols=_alias("fit.wald.df1", lambda fit: fit.wald.df1)),
    _row("_dfd", Feols=_alias("fit.wald.df2", lambda fit: fit.wald.df2)),
    _row(
        "_wald_statistic",
        Feols=_alias("fit.wald.wald_statistic", lambda fit: fit.wald.wald_statistic),
    ),
    _row(
        "_f_statistic",
        Feols=_alias("fit.wald.f_statistic", lambda fit: fit.wald.f_statistic),
    ),
    _row("_p_value", Feols=_alias("fit.wald.pvalue", lambda fit: fit.wald.pvalue)),
    # fit.fixef_estimates
    _row(
        "_fixef_dict",
        Feols=_removed(
            "fit.fixef_estimates",
            "`fixef()` now returns a data frame, and the nested-dict layout is gone",
        ),
    ),
    _row(
        "_alpha",
        Feols=_alias(
            "fit.fixef_estimates.alpha", lambda fit: fit.fixef_estimates.alpha
        ),
    ),
    _row(
        "_sumFE",
        Feols=_alias(
            "fit.fixef_estimates.sumFE",
            lambda fit: _sqrt_weighted(fit, fit.fixef_estimates.sumFE),
        ),
        Feglm=_alias("fit.fixef_estimates.sumFE", _glm_sum_fe),
    ),
    # fit.first_stage
    _row(
        "_pi_hat",
        Feiv=_alias(
            "fit.first_stage.coefficients", lambda fit: fit.first_stage.coefficients
        ),
    ),
    _row(
        "_X_hat",
        Feiv=_alias(
            "fit.first_stage.fitted_values",
            lambda fit: _sqrt_weighted(fit, fit.first_stage.fitted_values),
        ),
    ),
    _row(
        "_v_hat",
        Feiv=_alias(
            "fit.first_stage.residuals",
            lambda fit: _sqrt_weighted(fit, fit.first_stage.residuals),
        ),
    ),
    _row(
        "_model_1st_stage",
        Feiv=_alias("fit.first_stage.model", lambda fit: fit.first_stage.model),
    ),
    _row(
        "_non_exo_instruments",
        Feiv=_alias(
            "fit.first_stage.instruments",
            lambda fit: list(fit.first_stage.instruments),
        ),
    ),
    _row(
        "_p_iv",
        Feiv=_alias(
            "fit.first_stage.instruments",
            lambda fit: len(fit.first_stage.instruments),
        ),
    ),
    _row(
        "_f_stat_1st_stage",
        silent=True,
        Feiv=_alias(
            "fit.first_stage.diagnostics.f_stat",
            lambda fit: fit.first_stage.diagnostics.f_stat,
        ),
    ),
    _row(
        "_p_value_1st_stage",
        Feiv=_alias(
            "fit.first_stage.diagnostics.p_value",
            lambda fit: fit.first_stage.diagnostics.p_value,
        ),
    ),
    _row("_eff_F", Feiv=_alias("fit.first_stage.diagnostics.eff_f", _eff_f)),
    # fit.collinearity and fit.collinearity_instruments
    _row(
        "_collin_vars",
        Feols=_alias(
            "fit.collinearity.dropped_coef_names",
            lambda fit: list(fit.collinearity.dropped_coef_names),
        ),
    ),
    _row(
        "_collin_index",
        Feols=_alias(
            "fit.collinearity.mask", lambda fit: _collin_index(fit.collinearity)
        ),
    ),
    _row(
        "_collin_vars_z",
        Feiv=_alias(
            "fit.collinearity_instruments.dropped_coef_names",
            lambda fit: list(fit.collinearity_instruments.dropped_coef_names),
        ),
    ),
    _row(
        "_collin_index_z",
        Feiv=_alias(
            "fit.collinearity_instruments.mask",
            lambda fit: _collin_index(fit.collinearity_instruments),
        ),
    ),
    # fit.capabilities
    _row(
        "_support_crv3_inference",
        Feols=_alias(
            "fit.capabilities.crv3_inference",
            lambda fit: fit.capabilities.crv3_inference,
        ),
    ),
    _row(
        "_support_iid_inference",
        # Every IV and GLM fit supported iid inference; 0.60 never read it.
        Feiv=_alias("fit.capabilities", lambda fit: True),
        Feglm=_alias("fit.capabilities", lambda fit: True),
    ),
    # fit.event_study_design
    _row(
        "_res_cohort_eventtime_dict",
        Feols=_alias(
            "fit.event_study_design.cohort_event_times",
            lambda fit: _event_study_design(fit).cohort_event_times,
        ),
    ),
    *(
        _row(
            f"_{name}",
            Feols=_alias(
                f"fit.event_study_design.{name}",
                lambda fit, name=name: getattr(_event_study_design(fit), name),
            ),
        )
        for name in ("yname", "gname", "tname", "idname", "att")
    ),
)
