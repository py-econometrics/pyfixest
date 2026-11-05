"""Check legacy attributes alongside the 0.60 release comparisons."""

import warnings

import numpy as np
import pytest

import pyfixest as pf
from pyfixest.estimation.models._deprecated_attributes import (
    DEPRECATED_ATTRIBUTES,
    Alias,
    Removed,
)
from tests.test_release_contract import (
    ALIAS_CHANGES,
    ALIAS_EXCEPTIONS,
    ALIAS_SPECIAL,
    LEGACY_ALIAS_CASES,
    LEGACY_CLASSES,
    legacy_aliases,
)


def test_every_alias_entry_has_a_release_comparison_or_exception():
    """Each class-specific entry, not just each name, must be compared."""
    registry = {attribute.name: attribute for attribute in DEPRECATED_ATTRIBUTES}
    compared = set()
    for case, (kind, _) in LEGACY_ALIAS_CASES.items():
        for name in legacy_aliases(kind):
            if (case, name) not in ALIAS_CHANGES:
                classes = registry[name].classes
                owner = next(c for c in reversed(LEGACY_CLASSES[kind]) if c in classes)
                compared.add((owner, name))
    uncompared = set(ALIAS_SPECIAL) | set(ALIAS_EXCEPTIONS)
    entries = {
        (owner, attribute.name)
        for attribute in DEPRECATED_ATTRIBUTES
        for owner, entry in attribute.classes.items()
        if isinstance(entry, Alias) and attribute.name not in uncompared
    }
    assert sorted(entries - compared) == []
    assert uncompared <= set(registry)


def test_aliases_without_comparable_values_are_accessible():
    data = pf.get_data(N=500, seed=2026).dropna()
    fit = pf.feols("Y ~ X1 + X2", data)
    fit.ritest(
        "X1",
        reps=9,
        rng=np.random.default_rng(2026),
        store_ritest_statistics=True,
        choose_algorithm="slow",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for name in ALIAS_EXCEPTIONS:
            getattr(fit, name)


def test_removed_attributes_raise():
    data = pf.get_data(N=500, seed=2026).dropna()
    fits = {
        "Feols": pf.feols("Y ~ X1", data),
        "Feiv": pf.feols("Y ~ X2 + [X1 ~ Z1]", data),
    }
    for attribute in DEPRECATED_ATTRIBUTES:
        for model, entry in attribute.classes.items():
            if isinstance(entry, Removed):
                with pytest.raises(AttributeError):
                    getattr(fits[model], attribute.name)
