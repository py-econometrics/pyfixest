"""Read-only compatibility aliases used by released table integrations."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pyfixest.errors import PyfixestDeprecationWarning

if TYPE_CHECKING:
    from pyfixest.estimation.models.feols_ import Feols

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


def _cluster_counts(fit: Feols) -> list[int]:
    """0.60 `_G`: the `G_df="min"` rule padded the minimum to three entries."""
    counts = list(fit.variance_covariance.G)
    if counts and fit.options.ssc.G_df == "min":
        return [min(counts)] * 3
    return counts


def _sample_split_value(fit: Feols) -> str | int | float | None:
    """Rebuild the 0.60 sample-split value."""
    split = fit.model.sample_split
    if split is None:
        return None
    return "all" if split.value is None else split.value


def _attribute_spec(name: str, *, silent: bool = False, **classes: Alias | Removed):
    return DeprecatedAttributeSpec(name=name, classes=classes, silent=silent)


def _alias(replacement: str, read: Callable[[Any], Any]) -> Alias:
    return Alias(replacement=replacement, read=read)


DEPRECATED_ATTRIBUTES: tuple[DeprecatedAttributeSpec, ...] = (
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
    _attribute_spec(
        "_N",
        silent=True,
        Feols=_alias("fit.sample_info.n_obs", lambda fit: fit.sample_info.n_obs),
    ),
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
        "_pvalue",
        silent=True,
        Feols=_alias("fit.coeftable.pvalue", lambda fit: fit.coeftable.pvalue),
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
        "_f_stat_1st_stage",
        silent=True,
        Feiv=_alias(
            "fit.first_stage.diagnostics.f_stat",
            lambda fit: fit.first_stage.diagnostics.f_stat,
        ),
    ),
)
