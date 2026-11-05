"""Deprecated fitted-model attributes of pyfixest 0.60.

pyfixest 0.70 moved the loose private attributes of fitted models into typed
state such as `fit.model`, `fit.coeftable`, or `fit.variance_covariance`.
`DEPRECATED_ATTRIBUTES` lists the compatibility aliases and explicit removals, and
`DeprecatedAttributes` turns each row into a class-level data descriptor on the
model classes it names:

- an `Alias` rebuilds the 0.60 representation from current fitted state and
  emits a `PyfixestDeprecationWarning` naming the replacement;
- a `Removed` entry raises `AttributeError`, explaining why the 0.60
  representation is no longer supported and naming the replacement.

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
from typing import TYPE_CHECKING, Any

from pyfixest.errors import PyfixestDeprecationWarning
from pyfixest.estimation.api.utils import _ALL_SAMPLE, _AllSampleSentinel

if TYPE_CHECKING:
    from pyfixest.did.saturated_twfe import EventStudyDesign
    from pyfixest.estimation.internals.model_state import CollinearityCheck
    from pyfixest.estimation.models.feiv_ import Feiv
    from pyfixest.estimation.models.feols_ import Feols
    from pyfixest.estimation.models.fepois_ import Fepois

# The release that moved the attributes into typed state.
REMOVED_IN = "0.70.0"


@dataclass(frozen=True, slots=True, kw_only=True)
class Alias:
    """Rebuild a deprecated attribute's 0.60 representation and name its replacement.

    Parameters
    ----------
    replacement : str
        The typed state to use instead, as users write it, e.g.
        ``"fit.coeftable.se"``.
    read : Callable[[Any], Any]
        Rebuild the former representation from the current fitted model. It raises
        `AttributeError` while the typed state it reads is not published or
        was dropped by the storage options.
    """

    replacement: str
    read: Callable[[Any], Any]

    def resolve(self, name: str, fit: object) -> Any:
        """Return the compatibility value of `fit.<name>`."""
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
    """Reject access to a removed 0.60 attribute with a reason and replacement.

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
class DeprecatedAttributeSpec:
    """Map a deprecated attribute to alias or removal rules by model class.

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
    """Apply one class's deprecation rule with warnings and read-only access."""

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
    """Install deprecation descriptors from the registry when subclasses are defined."""

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
# Metadata conversions preserve the former defaults, types, and availability.
# ---------------------------------------------------------------------------


def _cluster_counts(fit: Feols) -> list[int]:
    """0.60 `_G`: the `G_df="min"` rule padded the minimum to three entries."""
    counts = list(fit.variance_covariance.G)
    if counts and fit.options.ssc.G_df == "min":
        return [min(counts)] * 3
    return counts


def _sample_split_value(fit: Feols) -> str | int | float | _AllSampleSentinel:
    """Rebuild the 0.60 sample-split value."""
    split = fit.model.sample_split
    if split is None or split.value is None:
        return _ALL_SAMPLE
    return split.value


def _na_index(fit: Feols) -> frozenset[int]:
    """0.60 `_na_index` excluded rows subsequently dropped for separation."""
    if fit.sample_info.dropped_by_stage.separation:
        raise AttributeError(
            "the row positions before separation are no longer retained; "
            "fit.sample_info.dropped_row_index includes all filtering stages"
        )
    return fit.sample_info.dropped_row_index


def _poisson_n_fe(fit: Fepois) -> int:
    """After separation, 0.60 Poisson counted only FE with multiple levels."""
    counts = fit.fixef_counts
    if fit.sample_info.dropped_by_stage.separation:
        return sum(n > 1 for n in counts.n_levels_by_fe)
    return counts.n_fixef


def _eff_f(fit: Feiv) -> float:
    """0.60 `_eff_F`, set only once `IV_Diag()` or `eff_F()` ran."""
    eff_f = fit.first_stage.diagnostics.eff_f
    if eff_f is None:
        raise AttributeError("run `IV_Diag()` or `eff_F()` first")
    return eff_f


def _event_study_design(fit: Feols) -> EventStudyDesign:
    """Return the design of a saturated event study; other fits carry none."""
    try:
        # Saturated event studies attach this state dynamically.
        return getattr(fit, "event_study_design")  # noqa: B009
    except AttributeError:
        raise AttributeError(
            "only `event_study(estimator='saturated')` fits record their design"
        ) from None


def _collin_index(collinearity: CollinearityCheck) -> list[bool]:
    """0.60 `_collin_index`: empty unless a column was dropped."""
    return list(collinearity.mask) if collinearity.any_dropped else []


def _poisson_null_measure(fit: Fepois, value: float) -> float | None:
    """0.60 left the Poisson null measures `None` for weighted fits."""
    return None if fit.options.has_weights else value


def _attribute_spec(name: str, *, silent: bool = False, **classes: Alias | Removed):
    return DeprecatedAttributeSpec(name=name, classes=classes, silent=silent)


def _alias(replacement: str, read: Callable[[Any], Any]) -> Alias:
    return Alias(replacement=replacement, read=read)


def _removed(replacement: str, reason: str) -> Removed:
    return Removed(replacement=replacement, reason=reason)


# Keep rows grouped by the typed state that replaces them.
DEPRECATED_ATTRIBUTES: tuple[DeprecatedAttributeSpec, ...] = (
    # fit.model
    _attribute_spec(
        "FixestFormula",
        Feols=_removed(
            "fit.model.parsed_formula",
            "the parsed formula interface changed, including its stages and fixed effects",
        ),
    ),
    _attribute_spec(
        "_model_spec",
        Feols=_alias("fit.model.model_spec", lambda fit: fit.model.model_spec),
    ),
    _attribute_spec(
        "_fml",
        silent=True,
        Feols=_alias("fit.model.formula", lambda fit: fit.model.formula),
    ),
    _attribute_spec(
        "_method",
        silent=True,
        Feols=_alias("fit.model.method", lambda fit: fit.model.method),
    ),
    _attribute_spec(
        "_is_iv", Feols=_alias("fit.model.is_iv", lambda fit: fit.model.is_iv)
    ),
    _attribute_spec(
        "_model_name",
        Feols=_alias("fit.model.model_name", lambda fit: fit.model.model_name),
    ),
    _attribute_spec(
        "_fixef",
        silent=True,
        Feols=_alias("fit.model.fixef", lambda fit: fit.model.fixef or None),
    ),
    _attribute_spec(
        "_has_fixef",
        silent=True,
        Feols=_alias("fit.model.has_fixef", lambda fit: fit.model.has_fixef),
    ),
    _attribute_spec(
        "_icovars",
        Feols=_alias(
            "fit.model.interacted_covariates",
            lambda fit: list(fit.model.interacted_covariates) or None,
        ),
    ),
    _attribute_spec(
        "_depvar",
        silent=True,
        Feols=_alias("fit.model.depvar", lambda fit: fit.model.depvar),
    ),
    _attribute_spec(
        "_sample_split_value",
        silent=True,
        Feols=_alias("fit.model.sample_split.value", _sample_split_value),
    ),
    _attribute_spec(
        "_sample_split_var",
        silent=True,
        Feols=_alias(
            "fit.model.sample_split",
            lambda fit: (
                None if fit.model.sample_split is None else fit.model.sample_split.var
            ),
        ),
    ),
    # fit.options
    *(
        _attribute_spec(
            f"_{name}",
            Feols=_alias(
                f"fit.options.{name}",
                lambda fit, name=name: getattr(fit.options, name),
            ),
        )
        for name in (
            "drop_singletons",
            "drop_intercept",
            "fixef_tol",
            "store_data",
            "copy_data",
            "lean",
            "context",
            "demeaner",
        )
    ),
    _attribute_spec(
        "_offset_name",
        Feols=_alias("fit.options.offset", lambda fit: fit.options.offset),
    ),
    _attribute_spec(
        "_fixef_maxiter",
        Feols=_alias(
            "fit.options.demeaner.fixef_maxiter",
            lambda fit: fit.options.demeaner.fixef_maxiter,
        ),
    ),
    *(
        _attribute_spec(
            f"_{name}",
            Quantreg=_alias(
                f"fit.options.{name}",
                lambda fit, name=name: getattr(fit.options, name),
            ),
        )
        for name in ("quantile_tol", "quantile_maxiter", "seed")
    ),
    _attribute_spec(
        "_accelerate",
        Feglm=_alias("fit.options.accelerate", lambda fit: fit.options.accelerate),
    ),
    _attribute_spec(
        "_ssc_dict",
        Feols=_alias(
            "fit.options.ssc", lambda fit: dataclasses.asdict(fit.options.ssc)
        ),
    ),
    _attribute_spec(
        "_collin_tol",
        Feols=_alias("fit.options.collin_tol", lambda fit: fit.options.collin_tol),
    ),
    _attribute_spec(
        "_solver",
        Feols=_alias("fit.options.solver", lambda fit: fit.options.solver),
    ),
    _attribute_spec(
        "maxiter",
        Feglm=_alias("fit.options.maxiter", lambda fit: fit.options.maxiter),
    ),
    _attribute_spec(
        "tol", Feglm=_alias("fit.options.tol", lambda fit: fit.options.tol)
    ),
    _attribute_spec(
        "separation_check",
        Feglm=_alias(
            "fit.options.separation_check", lambda fit: fit.options.separation_check
        ),
    ),
    _attribute_spec(
        "_has_weights",
        Feols=_alias("fit.options.has_weights", lambda fit: fit.options.has_weights),
    ),
    _attribute_spec(
        "_weights_name",
        Feols=_alias("fit.options.weights", lambda fit: fit.options.weights),
    ),
    _attribute_spec(
        "_weights_type",
        Feols=_alias("fit.options.weights_type", lambda fit: fit.options.weights_type),
    ),
    _attribute_spec(
        "_quantile",
        Quantreg=_alias("fit.options.quantile", lambda fit: fit.options.quantile),
    ),
    # fit.sample_info, fit.observation_weights, fit.model_matrix
    _attribute_spec(
        "_N",
        silent=True,
        Feols=_alias("fit.sample_info.n_obs", lambda fit: fit.sample_info.n_obs),
    ),
    _attribute_spec(
        "_N_rows",
        Feols=_alias("fit.sample_info.n_rows", lambda fit: fit.sample_info.n_rows),
        Feglm=_alias(
            "fit.sample_info",
            lambda fit: (
                fit.sample_info.n_rows + fit.sample_info.dropped_by_stage.separation
            ),
        ),
        Fepois=_alias("fit.sample_info.n_rows", lambda fit: fit.sample_info.n_rows),
    ),
    _attribute_spec(
        "_na_index", Feols=_alias("fit.sample_info.dropped_row_index", _na_index)
    ),
    _attribute_spec(
        "n_separation_na",
        Feols=_alias(
            "fit.sample_info.dropped_by_stage.separation",
            lambda fit: fit.sample_info.dropped_by_stage.separation,
        ),
    ),
    _attribute_spec(
        "_k_fe",
        Feols=_removed(
            "fit.fixef_counts.n_levels_by_fe",
            "its encoded Series labels and process-dependent ordering are no "
            "longer supported",
        ),
    ),
    _attribute_spec(
        "_n_fe",
        Feols=_alias("fit.fixef_counts.n_fixef", lambda fit: fit.fixef_counts.n_fixef),
        Fepois=_alias("fit.fixef_counts.n_fixef", _poisson_n_fe),
    ),
    _attribute_spec(
        "_fe",
        Feols=_removed(
            "fit.model_matrix.fixed_effects",
            "the fixed-effect codes are now numbered by sorted level instead of "
            "by first appearance, and GLM fits no longer hold them as an array",
        ),
    ),
    _attribute_spec(
        "na_index",
        Feols=_removed(
            "fit.sample_info.dropped_row_index",
            "it only listed the rows a GLM dropped for separation, by position "
            "in the already filtered data",
        ),
    ),
    # fit.within_data and fit.working_state
    _attribute_spec(
        "_Yd",
        Feols=_removed(
            "fit.model_matrix.dependent",
            "it held the demeaned response before the collinearity check",
        ),
    ),
    _attribute_spec(
        "_Xd",
        Feols=_removed(
            "fit.model_matrix.independent",
            "it held the demeaned design before the collinearity check",
        ),
    ),
    _attribute_spec(
        "_Zd",
        Feiv=_removed(
            "fit.model_matrix.instruments",
            "it held the demeaned instruments before the collinearity check",
        ),
    ),
    _attribute_spec(
        "_endogvard",
        Feiv=_removed(
            "fit.model_matrix.endogenous",
            "it held the demeaned endogenous variables before the collinearity check",
        ),
    ),
    # fit.fitted_values
    # fit.sandwich
    # fit.variance_covariance
    _attribute_spec(
        "_vcov",
        silent=True,
        Feols=_alias(
            "fit.variance_covariance.vcov", lambda fit: fit.variance_covariance.vcov
        ),
    ),
    _attribute_spec(
        "_vcov_type",
        silent=True,
        Feols=_alias(
            "fit.variance_covariance.spec.vcov_type",
            lambda fit: fit.variance_covariance.spec.vcov_type,
        ),
    ),
    _attribute_spec(
        "_vcov_type_detail",
        Feols=_alias(
            "fit.variance_covariance.spec.vcov_type_detail",
            lambda fit: fit.variance_covariance.spec.vcov_type_detail,
        ),
    ),
    _attribute_spec(
        "_is_clustered",
        Feols=_alias(
            "fit.variance_covariance.spec.is_clustered",
            lambda fit: fit.variance_covariance.spec.is_clustered,
        ),
    ),
    _attribute_spec(
        "_clustervar",
        silent=True,
        Feols=_alias(
            "fit.variance_covariance.spec.clustervar",
            lambda fit: list(fit.variance_covariance.spec.clustervar) or None,
        ),
    ),
    _attribute_spec(
        "_G",
        silent=True,
        Feols=_alias("fit.variance_covariance.G", _cluster_counts),
    ),
    _attribute_spec(
        "_df_k",
        Feols=_alias(
            "fit.variance_covariance.df_k", lambda fit: fit.variance_covariance.df_k
        ),
    ),
    _attribute_spec(
        "_df_t",
        Feols=_alias(
            "fit.variance_covariance.df_t", lambda fit: fit.variance_covariance.df_t
        ),
    ),
    # fit.coeftable
    _attribute_spec(
        "_se", Feols=_alias("fit.coeftable.se", lambda fit: fit.coeftable.se)
    ),
    _attribute_spec(
        "_tstat", Feols=_alias("fit.coeftable.tstat", lambda fit: fit.coeftable.tstat)
    ),
    _attribute_spec(
        "_pvalue",
        silent=True,
        Feols=_alias("fit.coeftable.pvalue", lambda fit: fit.coeftable.pvalue),
    ),
    _attribute_spec(
        "_conf_int",
        Feols=_alias("fit.coeftable.conf_int", lambda fit: fit.coeftable.conf_int),
    ),
    # fit.fitstat
    _attribute_spec(
        "get_performance",
        Feols=_alias("fit.fitstat", lambda fit: fit._publish_fit_statistics),
    ),
    _attribute_spec(
        "_rmse",
        silent=True,
        Feols=_alias("fit.fitstat.rmse", lambda fit: fit.fitstat.rmse),
    ),
    _attribute_spec(
        "_r2", silent=True, Feols=_alias("fit.fitstat.r2", lambda fit: fit.fitstat.r2)
    ),
    _attribute_spec(
        "_r2_within",
        silent=True,
        Feols=_alias("fit.fitstat.r2_within", lambda fit: fit.fitstat.r2_within),
    ),
    _attribute_spec(
        "_adj_r2",
        silent=True,
        Feols=_alias("fit.fitstat.adj_r2", lambda fit: fit.fitstat.adj_r2),
    ),
    _attribute_spec(
        "_r2_adj",
        silent=True,
        Feols=_alias("fit.fitstat.adj_r2", lambda fit: fit.fitstat.adj_r2),
    ),
    _attribute_spec(
        "_adj_r2_within",
        silent=True,
        Feols=_alias(
            "fit.fitstat.adj_r2_within", lambda fit: fit.fitstat.adj_r2_within
        ),
    ),
    _attribute_spec(
        "deviance",
        silent=True,
        # 0.60 left the deviance `None` for linear and quantile fits.
        Feols=_alias("fit.fitstat.deviance", lambda fit: None),
        Feglm=_alias("fit.fitstat.deviance", lambda fit: fit.fitstat.deviance),
    ),
    _attribute_spec(
        "_loglik",
        Fepois=_alias("fit.fitstat.loglik", lambda fit: fit.fitstat.loglik),
    ),
    _attribute_spec(
        "_loglik_null",
        Fepois=_alias(
            "fit.fitstat.loglik_null",
            lambda fit: _poisson_null_measure(fit, fit.fitstat.loglik_null),
        ),
    ),
    _attribute_spec(
        "_pseudo_r2",
        Fepois=_alias(
            "fit.fitstat.pseudo_r2",
            lambda fit: _poisson_null_measure(fit, fit.fitstat.pseudo_r2),
        ),
    ),
    _attribute_spec(
        "_pearson_chi2",
        Fepois=_alias("fit.fitstat.pearson_chi2", lambda fit: fit.fitstat.pearson_chi2),
    ),
    # fit.wald
    _attribute_spec("_dfn", Feols=_alias("fit.wald.df1", lambda fit: fit.wald.df1)),
    _attribute_spec("_dfd", Feols=_alias("fit.wald.df2", lambda fit: fit.wald.df2)),
    _attribute_spec(
        "_wald_statistic",
        Feols=_alias("fit.wald.wald_statistic", lambda fit: fit.wald.wald_statistic),
    ),
    _attribute_spec(
        "_f_statistic",
        Feols=_alias("fit.wald.f_statistic", lambda fit: fit.wald.f_statistic),
    ),
    _attribute_spec(
        "_p_value", Feols=_alias("fit.wald.pvalue", lambda fit: fit.wald.pvalue)
    ),
    # fit.fixef_estimates
    _attribute_spec(
        "_fixef_dict",
        Feols=_removed(
            "fit.fixef_estimates",
            "`fixef()` now returns a data frame, and the nested-dict layout is gone",
        ),
    ),
    _attribute_spec(
        "_alpha",
        Feols=_removed(
            "fit.fixef()",
            "the dummy-coefficient layout changed, including unused categorical "
            "levels; fixef() returns coefficients labelled by fixed-effect term and level",
        ),
    ),
    # fit.first_stage
    _attribute_spec(
        "_model_1st_stage",
        Feiv=_alias("fit.first_stage.model", lambda fit: fit.first_stage.model),
    ),
    _attribute_spec(
        "_non_exo_instruments",
        Feiv=_alias(
            "fit.first_stage.instruments",
            lambda fit: list(fit.first_stage.instruments),
        ),
    ),
    _attribute_spec(
        "_p_iv",
        Feiv=_alias(
            "fit.first_stage.instruments",
            lambda fit: len(fit.first_stage.instruments),
        ),
    ),
    _attribute_spec(
        "_f_stat_1st_stage",
        silent=True,
        Feiv=_alias(
            "fit.first_stage.diagnostics.f_stat",
            lambda fit: fit.first_stage.diagnostics.f_stat,
        ),
    ),
    _attribute_spec(
        "_p_value_1st_stage",
        Feiv=_alias(
            "fit.first_stage.diagnostics.p_value",
            lambda fit: fit.first_stage.diagnostics.p_value,
        ),
    ),
    _attribute_spec("_eff_F", Feiv=_alias("fit.first_stage.diagnostics.eff_f", _eff_f)),
    # fit.collinearity and fit.collinearity_instruments
    _attribute_spec(
        "_collin_vars",
        Feols=_alias(
            "fit.collinearity.dropped_coef_names",
            lambda fit: list(fit.collinearity.dropped_coef_names),
        ),
    ),
    _attribute_spec(
        "_collin_index",
        Feols=_alias(
            "fit.collinearity.mask", lambda fit: _collin_index(fit.collinearity)
        ),
    ),
    _attribute_spec(
        "_collin_vars_z",
        Feiv=_alias(
            "fit.collinearity_instruments.dropped_coef_names",
            lambda fit: list(fit.collinearity_instruments.dropped_coef_names),
        ),
    ),
    _attribute_spec(
        "_collin_index_z",
        Feiv=_alias(
            "fit.collinearity_instruments.mask",
            lambda fit: _collin_index(fit.collinearity_instruments),
        ),
    ),
    # fit.capabilities
    *(
        _attribute_spec(
            name,
            Feols=_alias(
                f"fit.capabilities.{field}",
                lambda fit, field=field: getattr(fit.capabilities, field),
            ),
        )
        for name, field in (
            ("_support_hac_inference", "hac_inference"),
            ("_support_decomposition", "decomposition"),
            ("_supports_wildboottest", "wildboottest"),
            ("_supports_cluster_causal_variance", "cluster_causal_variance"),
        )
    ),
    _attribute_spec(
        "_support_crv3_inference",
        Feols=_alias(
            "fit.capabilities.crv3_inference",
            lambda fit: fit.capabilities.crv3_inference,
        ),
    ),
    _attribute_spec(
        "_support_iid_inference",
        # Every IV and GLM fit supported iid inference; 0.60 never read it.
        # There is no IID capability field: vcov validates the requested type.
        Feiv=_alias("fit.vcov", lambda fit: True),
        Feglm=_alias("fit.vcov", lambda fit: True),
    ),
    # fit.ritest_statistics
    *(
        _attribute_spec(
            f"_ritest_{name}",
            Feols=_alias(
                f"fit.ritest_statistics.{name}",
                lambda fit, name=name: getattr(fit.ritest_statistics, name),
            ),
        )
        for name in ("statistics", "sample_stat", "pvalue")
    ),
    # fit.event_study_design
    _attribute_spec(
        "_res_cohort_eventtime_dict",
        Feols=_alias(
            "fit.event_study_design.cohort_event_times",
            lambda fit: _event_study_design(fit).cohort_event_times,
        ),
    ),
    *(
        _attribute_spec(
            f"_{name}",
            Feols=_alias(
                f"fit.event_study_design.{name}",
                lambda fit, name=name: getattr(_event_study_design(fit), name),
            ),
        )
        for name in ("yname", "gname", "tname", "idname", "att")
    ),
)
