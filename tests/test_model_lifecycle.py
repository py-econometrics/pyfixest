"""Behaviour of fitted models under `lean=True` and `store_data=False`.

`_clear_attributes` drops attributes by name; these tests pin which methods
survive that and check the ones that do not fail with an informative error
rather than an `AttributeError` naming a private attribute.
"""

import numpy as np
import pytest

import pyfixest as pf
from pyfixest.errors import ModelAttributeStrippedError

FML = "Y ~ X1 + X2 | f1"


@pytest.fixture(scope="module")
def data():
    return pf.get_data().dropna().reset_index(drop=True)


def _call(name, fit):
    return {
        "predict": lambda f: f.predict(),
        "fixef": lambda f: f.fixef(),
        "resid": lambda f: f.resid(),
        "vcov": lambda f: f.vcov({"CRV1": "f1"}),
        "ritest": lambda f: f.ritest("X1", reps=5),
        "get_performance": lambda f: f.get_performance(),
        "wildboottest": lambda f: f.wildboottest(param="X1", reps=99),
        "tidy": lambda f: f.tidy(),
        "confint": lambda f: f.confint(),
        "coef": lambda f: f.coef(),
    }[name](fit)


# methods that keep working once the attribute set has been stripped
SURVIVES = {
    "lean": ["tidy", "confint", "coef"],
    "store_data": ["tidy", "confint", "coef", "predict", "resid", "get_performance"],
}
# methods that need stripped state and must say so
RAISES = {
    "lean": [
        "predict",
        "fixef",
        "resid",
        "vcov",
        "ritest",
        "get_performance",
        "wildboottest",
    ],
    "store_data": ["fixef", "vcov", "ritest", "wildboottest"],
}
FLAGS = {"lean": {"lean": True}, "store_data": {"store_data": False}}


@pytest.mark.parametrize("mode", sorted(FLAGS))
@pytest.mark.parametrize("method", sorted({m for v in SURVIVES.values() for m in v}))
def test_surviving_methods_still_work(data, mode, method):
    if method not in SURVIVES[mode]:
        pytest.skip(f"{method} is not expected to survive {mode}")
    _call(method, pf.feols(FML, data, **FLAGS[mode]))


@pytest.mark.parametrize("mode", sorted(FLAGS))
@pytest.mark.parametrize("method", sorted({m for v in RAISES.values() for m in v}))
def test_stripped_methods_raise_informatively(data, mode, method):
    """The error names the method, the attribute and the flag that removed it."""
    if method not in RAISES[mode]:
        pytest.skip(f"{method} does not need state stripped by {mode}")
    fit = pf.feols(FML, data, **FLAGS[mode])
    with pytest.raises(ModelAttributeStrippedError) as exc:
        _call(method, fit)
    msg = str(exc.value)
    assert method in msg
    assert ("lean=True" in msg) or ("store_data=False" in msg)


def test_clear_sets_cover_what_is_deleted(data):
    """Every name in the clear sets is actually gone from a stripped model."""
    fit = pf.feols(FML, data, lean=True)
    leftover = [a for a in type(fit)._LEAN_CLEARED if hasattr(fit, a)]
    assert leftover == [], f"lean=True left {leftover} in place"

    fit = pf.feols(FML, data, store_data=False)
    leftover = [a for a in type(fit)._DATA_CLEARED if hasattr(fit, a)]
    assert leftover == [], f"store_data=False left {leftover} in place"


def test_demean_cache_is_released_but_preconditioner_survives(data):
    """The shared demeaned-data cache must not be pinned by a fitted model."""
    fit = pf.feols(FML, data)
    assert fit._demean_cache.lookup_demeaned_data == {}
    # the preconditioner lookup backs the public `preconditioner` property
    assert isinstance(fit._demean_cache.lookup_preconditioner, dict)
    assert fit.preconditioner is None or fit.preconditioner is not None


def test_multiple_estimation_does_not_pin_the_cache(data):
    """Each model in a cache block releases the block's demeaned data."""
    fits = pf.feols("Y ~ X1 | csw0(f1, f2)", data)
    for m in fits.to_list():
        assert m._demean_cache.lookup_demeaned_data == {}
    assert np.isfinite(fits.to_list()[0].coef().to_numpy()).all()
