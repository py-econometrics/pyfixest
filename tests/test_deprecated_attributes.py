"""Check the retained and removed fitted-model compatibility surface."""

import pytest

import pyfixest as pf
from pyfixest.estimation.models._deprecated_attributes import (
    DEPRECATED_ATTRIBUTES,
    REMOVED_PRIVATE_ATTRIBUTES,
    Alias,
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
    """Each retained class-specific alias is compared or explicitly excepted."""
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
    assert not set(registry) & REMOVED_PRIVATE_ATTRIBUTES


def test_removed_private_attributes_point_to_migration_guide():
    fit = pf.feols("Y ~ X1", pf.get_data(N=500, seed=2026).dropna())
    with pytest.raises(AttributeError, match=r"private attribute.*migration guide"):
        _ = fit._se


def test_removed_names_do_not_shadow_current_internal_state():
    data = pf.get_data(N=500, seed=2026).dropna()
    fit = pf.feols("Y ~ X1", data)
    assert fit._u_hat is not None
    iv = pf.feols("Y ~ X2 + [X1 ~ Z1]", data)
    assert iv._coefnames_z
    with pytest.raises(AttributeError, match="has no attribute 'not_a_legacy_name'"):
        getattr(fit, "not_a_legacy_name")  # noqa: B009


def test_eff_f_reports_diagnostics_requirement_before_it_is_computed():
    data = pf.get_data(N=500, seed=2026).dropna()
    fit = pf.feols("Y ~ X2 + [X1 ~ Z1]", data)
    with pytest.raises(AttributeError, match=r"run `IV_Diag\(\)` or `eff_F\(\)` first"):
        _ = fit._eff_F
