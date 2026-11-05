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
    ALIAS_EXACT,
    ALIAS_EXCEPTIONS,
    ALIAS_NUMERIC,
    ALIAS_SPECIAL,
)


def test_every_alias_has_a_release_comparison_or_exception():
    aliases = {
        attribute.name
        for attribute in DEPRECATED_ATTRIBUTES
        if any(isinstance(entry, Alias) for entry in attribute.classes.values())
    }
    compared = {
        name
        for names in (*ALIAS_EXACT.values(), *ALIAS_NUMERIC.values())
        for name in names
    }
    assert aliases == compared | set(ALIAS_SPECIAL) | set(ALIAS_EXCEPTIONS)


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
