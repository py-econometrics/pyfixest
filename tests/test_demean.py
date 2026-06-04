import pickle

import numpy as np
import pandas as pd
import pyhdfe
import pytest

import pyfixest as pf
from pyfixest.core import demean as demean_rs
from pyfixest.core.demean import demean_within
from pyfixest.demeaners import LsmrDemeaner, MapDemeaner
from pyfixest.estimation.cupy.demean_cupy_ import demean_cupy32, demean_cupy64
from pyfixest.estimation.internals.demean_ import (
    _resolve_preconditioner,
    demean_model,
    dispatch_demean,
)
from pyfixest.estimation.jax.demean_jax_ import demean_jax
from pyfixest.estimation.numba.demean_nb import demean as demean_numba
from tests._torch_test_utils import HAS_TORCH, torch_param

GENERIC_DEMEAN_FUNCS = [
    pytest.param(demean_numba, id="demean_numba"),
    pytest.param(demean_jax, id="demean_jax"),
    pytest.param(demean_rs, id="demean_rs"),
    pytest.param(demean_cupy32, id="demean_cupy32"),
    pytest.param(demean_cupy64, id="demean_cupy64"),
]

if HAS_TORCH:
    from pyfixest.estimation.torch.demean_torch_ import demean_torch

    GENERIC_DEMEAN_FUNCS.append(pytest.param(demean_torch, id="demean_torch"))


MODEL_DEMEANERS = [
    pytest.param(MapDemeaner(backend="numba"), id="numba"),
    pytest.param(MapDemeaner(backend="jax"), id="jax"),
    pytest.param(MapDemeaner(backend="rust"), id="rust"),
    pytest.param(LsmrDemeaner(), id="within"),
    pytest.param(LsmrDemeaner(backend="cupy", device="cpu"), id="lsmr_scipy"),
]

if HAS_TORCH:
    MODEL_DEMEANERS.append(
        pytest.param(
            LsmrDemeaner(backend="torch", device="cpu"),
            id="lsmr_torch_cpu",
        )
    )


TORCH_DEVICE_DEMEANERS = [
    torch_param(("demean_torch_cpu", 1e-6, 1e-8), id="demean_torch_cpu"),
    torch_param(("demean_torch_mps", 1e-3, 1e-3), id="demean_torch_mps", require="mps"),
    torch_param(
        ("demean_torch_cuda", 1e-6, 1e-8), id="demean_torch_cuda", require="cuda"
    ),
    torch_param(
        ("demean_torch_cuda32", 1e-3, 1e-3),
        id="demean_torch_cuda32",
        require="cuda",
    ),
]


@pytest.fixture(scope="module")
def demean_data():
    rng = np.random.default_rng(929291)

    n_obs = 1_000
    n_cols = 10
    x = rng.normal(0, 1, n_cols * n_obs).reshape((n_obs, n_cols))
    f1 = rng.choice(list(range(n_cols)), n_obs).reshape((n_obs, 1))
    f2 = rng.choice(list(range(n_cols)), n_obs).reshape((n_obs, 1))
    flist = np.concatenate((f1, f2), axis=1).astype(np.uint64)
    weights = rng.uniform(0, 1, n_obs)

    return x, flist, weights


@pytest.mark.parametrize("demean_func", GENERIC_DEMEAN_FUNCS)
def test_demean(benchmark, demean_func, demean_data):
    x, flist, weighted = demean_data

    # without weights
    weights = np.ones(x.shape[0])
    algorithm = pyhdfe.create(flist)
    res_pyhdfe = algorithm.residualize(x)
    res_pyfixest, _ = demean_func(x, flist, weights, tol=1e-10)
    assert np.allclose(res_pyhdfe[10, 0:], res_pyfixest[10, 0:], rtol=1e-06, atol=1e-08)

    # with weights
    algorithm = pyhdfe.create(flist)
    res_pyhdfe = algorithm.residualize(x, weighted.reshape((x.shape[0], 1)))
    res_pyfixest, _ = benchmark(demean_func, x, flist, weighted, tol=1e-10)
    assert np.allclose(res_pyhdfe[10, 0:], res_pyfixest[10, 0:], rtol=1e-06, atol=1e-08)


@pytest.mark.parametrize(("backend_name", "rtol", "atol"), TORCH_DEVICE_DEMEANERS)
def test_torch_device_backends_match_pyhdfe(backend_name, rtol, atol, demean_data):
    from pyfixest.estimation.torch.demean_torch_ import (
        demean_torch_cpu,
        demean_torch_cuda,
        demean_torch_cuda32,
        demean_torch_mps,
    )

    backend_map = {
        "demean_torch_cpu": demean_torch_cpu,
        "demean_torch_mps": demean_torch_mps,
        "demean_torch_cuda": demean_torch_cuda,
        "demean_torch_cuda32": demean_torch_cuda32,
    }

    x, flist, weights = demean_data
    demean_func = backend_map[backend_name]

    algorithm = pyhdfe.create(flist)
    res_pyhdfe = algorithm.residualize(x)
    res_torch, success = demean_func(x, flist, np.ones(x.shape[0]), tol=1e-10)
    assert success, f"{backend_name} did not converge on unweighted demeaning"
    np.testing.assert_allclose(res_torch, res_pyhdfe, rtol=rtol, atol=atol)

    res_pyhdfe = algorithm.residualize(x, weights.reshape((x.shape[0], 1)))
    res_torch, success = demean_func(x, flist, weights, tol=1e-10)
    assert success, f"{backend_name} did not converge on weighted demeaning"
    np.testing.assert_allclose(res_torch, res_pyhdfe, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("demeaner", "rtol", "atol"),
    [
        (
            LsmrDemeaner(),
            1e-6,
            1e-8,
        ),
        (
            LsmrDemeaner(
                preconditioner="none",
                fixef_atol=1e-10,
                fixef_btol=1e-10,
                fixef_maxiter=10_000,
            ),
            1e-6,
            1e-8,
        ),
    ],
)
def test_within_solver_variants_match_pyhdfe(demeaner, rtol, atol, demean_data):
    x, flist, weights = demean_data

    algorithm = pyhdfe.create(flist)
    expected_unweighted = algorithm.residualize(x)
    result_unweighted, success = dispatch_demean(
        x=x,
        flist=flist,
        weights=np.ones(x.shape[0]),
        demeaner=demeaner,
    )
    assert success
    np.testing.assert_allclose(
        result_unweighted, expected_unweighted, rtol=rtol, atol=atol
    )

    expected_weighted = algorithm.residualize(x, weights.reshape((x.shape[0], 1)))
    result_weighted, success = dispatch_demean(
        x=x,
        flist=flist,
        weights=weights,
        demeaner=demeaner,
    )
    assert success
    np.testing.assert_allclose(result_weighted, expected_weighted, rtol=rtol, atol=atol)


def test_demean_within_returns_preconditioner_for_reuse(demean_data):
    x, flist, weights = demean_data

    result, success, preconditioner = demean_within(
        x=x,
        flist=flist.astype(np.uint32, copy=False),
        weights=weights,
    )
    assert success
    assert isinstance(preconditioner, pf.WithinPreconditioner)
    assert preconditioner.nrows == preconditioner.ncols

    # Reusing the preconditioner yields the same demeaned output.
    result_reused, success_reused, preconditioner_reused = demean_within(
        x=x,
        flist=flist.astype(np.uint32, copy=False),
        weights=weights,
        preconditioner=preconditioner,
    )
    assert success_reused
    assert preconditioner_reused is not None
    np.testing.assert_allclose(result_reused, result, rtol=1e-10, atol=1e-10)


def test_demean_within_preconditioner_pickle_roundtrip(demean_data):
    x, flist, weights = demean_data

    result, success, preconditioner = demean_within(
        x=x,
        flist=flist.astype(np.uint32, copy=False),
        weights=weights,
    )
    assert success
    assert preconditioner is not None

    restored = pickle.loads(pickle.dumps(preconditioner))
    assert isinstance(restored, pf.WithinPreconditioner)
    assert restored.nrows == preconditioner.nrows
    assert restored.ncols == preconditioner.ncols

    result_reused, success_reused, _ = demean_within(
        x=x,
        flist=flist.astype(np.uint32, copy=False),
        weights=weights,
        preconditioner=restored,
    )
    assert success_reused
    np.testing.assert_allclose(result_reused, result, rtol=1e-10, atol=1e-10)


def test_demean_within_rejects_invalid_preconditioner():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 2))
    flist = np.column_stack(
        [rng.integers(0, 3, size=20), rng.integers(0, 3, size=20)]
    ).astype(np.uint32)
    weights = np.ones(20)

    with pytest.raises(TypeError):
        demean_within(
            x=x,
            flist=flist,
            weights=weights,
            preconditioner=object(),  # type: ignore[arg-type]
        )


def test_demean_within_rejects_unknown_preconditioner_string():
    """Unknown string preconditioners must raise on every FE-count path.

    Regression test: pre-fix, the single-FE MAP fallback bypassed string
    validation and silently accepted garbage like ``preconditioner="bogus"``.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 2))
    weights = np.ones(20)

    single = np.asfortranarray(rng.integers(0, 3, size=(20, 1)).astype(np.uint32))
    multi = np.asfortranarray(
        np.column_stack(
            [rng.integers(0, 3, size=20), rng.integers(0, 3, size=20)]
        ).astype(np.uint32)
    )
    for flist in (single, multi):
        with pytest.raises(ValueError, match="bogus"):
            demean_within(
                x=x,
                flist=flist,
                weights=weights,
                preconditioner="bogus",
            )


def test_demean_within_rejects_mismatched_preconditioner(demean_data):
    """Reusing a preconditioner built for a smaller FE design must raise."""
    x, flist, weights = demean_data

    # Build a preconditioner on a *smaller* 2-FE design (fewer levels per factor).
    flist_small = (flist % 5).astype(np.uint32, copy=False)
    _, _, small_pre = demean_within(
        x=x,
        flist=flist_small,
        weights=weights,
    )
    assert small_pre is not None

    with pytest.raises(ValueError, match="DOF"):
        demean_within(
            x=x,
            flist=flist.astype(np.uint32, copy=False),
            weights=weights,
            preconditioner=small_pre,
        )


def test_within_preconditioner_value_equality(demean_data):
    """``WithinPreconditioner`` uses value semantics for ``__eq__`` / ``__hash__``.

    Three contracts are pinned down:

    1. Two distinct Python objects that wrap the same factorization compare
       equal under ``==``. We use ``pickle`` round-tripping to manufacture a
       second object with identical content but a different identity.
    2. ``hash(a) == hash(b)`` whenever ``a == b`` (Python's rule for sane
       behaviour in sets and dicts).
    3. Preconditioners built from *different* designs do not collide:
       ``__eq__`` actually inspects content rather than returning True
       indiscriminately.

    Motivation: the documented "save in session 1, reload in session 2"
    workflow relies on users being able to check ``loaded == fit.preconditioners[0]``
    and key dicts/sets by preconditioner content. Identity-based equality
    (Python's default) would silently break both.
    """
    x, flist, weights = demean_data

    # Build the reference preconditioner from the full 2-FE design.
    _, _, original = demean_within(
        x=x,
        flist=flist.astype(np.uint32, copy=False),
        weights=weights,
    )
    # Defensive: demean_data is multi-FE so this path returns a built
    # preconditioner. If a future refactor changes that, fail loudly here
    # instead of further down with a confusing ``NoneType has no __eq__``.
    assert original is not None

    # Round-tripping through pickle gives us a second Python object backed by
    # the same serialized factorization — exactly the "different object, same
    # content" case __eq__ needs to handle.
    restored = pickle.loads(pickle.dumps(original))
    assert original is not restored, "pickle should produce a fresh object"
    assert original == restored, "same content must compare equal"
    assert hash(original) == hash(restored), "equal objects must hash equally"

    # Negative case: a *different* design (fewer levels per factor via % 5)
    # produces a preconditioner with different n_dofs and different internal
    # state. __eq__ must distinguish it from the original.
    _, _, different = demean_within(
        x=x,
        flist=(flist % 5).astype(np.uint32, copy=False),
        weights=weights,
    )
    assert different is not None
    assert original != different, "different factorizations must compare unequal"


def test_lsmr_within_reuses_preconditioner_store(demean_data):
    """End-to-end coverage of ``dispatch_demean``'s preconditioner-cache policy.

    Context
    -------
    ``dispatch_demean`` is the orchestration layer that the model classes
    (``Feols``, ``Fepois``, ``Feglm``) call to demean their working data on
    each iteration. The ``preconditioner_store`` argument is a mutable list
    threaded through by the model class — it is the cache that turns IWLS's
    "build the Schwarz factorization once, reuse it across iterations" claim
    into reality. This test exercises that cache directly, bypassing the
    model classes so failures point at the cache logic instead of the IWLS
    loop.

    Three behaviours are pinned down:

    Leg 1 — first call seeds the cache
        With an empty store, a default ``LsmrDemeaner(backend="within")``
        builds a fresh Schwarz preconditioner and appends it to the store.
        Verifies the store ends up with exactly one ``WithinPreconditioner``
        entry.

    Leg 2 — subsequent calls reuse the cached preconditioner under changed
    weights (the IWLS hot path)
        IWLS changes the working weights at every iteration (``W := W * mu``).
        A pure implementation would rebuild the preconditioner each time,
        which is the expensive step. The cache lets us reuse the *original*
        factorization even though the weights — and therefore the true
        normal-equations operator — have changed. The factorization is
        "stale" in that sense, but LSMR still converges to the correct
        demeaned values, just possibly in a few more iterations. We simulate
        a weight change with ``weights + 0.25``, then check both that no new
        entry was appended (the cache was hit, not bypassed) and that the
        demeaned output still matches an independent ground truth.

        Ground truth comes from ``pyhdfe.residualize`` — a separate, mature
        FE-residualization implementation. Matching it under changed weights
        is the strongest evidence that the stale-preconditioner path is
        mathematically correct, not just self-consistent.

    Leg 3 — passing a cached preconditioner back via ``LsmrDemeaner``
    reproduces the original solve
        This is the documented "save in session 1, reload in session 2"
        workflow in miniature. A fresh ``LsmrDemeaner(preconditioner=store[0])``
        with the *original* weights should reproduce leg 1's output bit-for-bit
        because we are running the same LSMR with the same factorization on
        the same data.

    Tolerance choices
    -----------------
    Leg 2 uses ``rtol=1e-6, atol=1e-8`` because pyhdfe and ``within`` are two
    different iterative solvers; they agree to within solver tolerance, not
    machine precision. Leg 3 uses ``1e-10`` on both because we are comparing
    ``within`` against itself with identical inputs — only floating-point
    summation order from rayon parallelism could differ.
    """
    x, flist, weights = demean_data
    store: list = []
    demeaner = LsmrDemeaner(backend="within")

    # ----- Leg 1: empty store → build a fresh preconditioner and cache it.
    result, success = dispatch_demean(
        x=x,
        flist=flist,
        weights=weights,
        demeaner=demeaner,
        preconditioner_store=store,
    )
    assert success
    assert len(store) == 1, "first call must seed the cache with one entry"
    assert isinstance(store[0], pf.WithinPreconditioner)

    # ----- Leg 2: changed weights, same store. Mimics an IWLS iteration where
    # only the working weights moved. The cache must hit (no new append) and
    # the result must still be numerically correct under the *new* weights.
    adjusted_weights = weights + 0.25
    result_reused, success = dispatch_demean(
        x=x,
        flist=flist,
        weights=adjusted_weights,
        demeaner=demeaner,
        preconditioner_store=store,
    )
    assert success
    assert len(store) == 1, "cache hit must not append a second entry"

    # Independent ground truth: pyhdfe residualizes the same design+weights
    # via its own MAP solver. Matching it proves the stale-preconditioner
    # path produces correct demeaned values, not just plausible ones.
    expected_reused = pyhdfe.create(flist).residualize(
        x, adjusted_weights.reshape((x.shape[0], 1))
    )
    np.testing.assert_allclose(
        result_reused,
        expected_reused,
        rtol=1e-6,
        atol=1e-8,
        err_msg="stale-preconditioner LSMR must still converge to the correct residuals",
    )

    # ----- Leg 3: explicit preconditioner reuse. Mimics the across-sessions
    # workflow: pull `store[0]` (or a pickled+restored copy) and pass it
    # through `LsmrDemeaner(preconditioner=...)`. With the original weights,
    # the result must reproduce leg 1 down to floating-point noise.
    result_explicit, success = dispatch_demean(
        x=x,
        flist=flist,
        weights=weights,
        demeaner=LsmrDemeaner(backend="within", preconditioner=store[0]),
    )
    assert success
    np.testing.assert_allclose(
        result_explicit,
        result,
        rtol=1e-10,
        atol=1e-10,
        err_msg="same precond + same weights + same data must reproduce the original solve",
    )


def test_lsmr_within_does_not_store_unused_explicit_preconditioner(demean_data):
    """The cache only holds preconditioners that were actually applied to the solve.

    Scenario
    --------
    A user passes an explicit ``WithinPreconditioner`` via ``LsmrDemeaner``,
    but the underlying data has a *single* fixed effect. ``demean_within``
    special-cases single-FE designs and routes them to the MAP fallback,
    which does not consult the preconditioner — the user-supplied object is
    silently ignored.

    What we are pinning down
    ------------------------
    The dispatcher must *not* append the ignored preconditioner to the
    cache. Entries in ``preconditioner_store`` (and therefore in
    ``fit.preconditioners``) are supposed to reflect factorizations that
    actually participated in a solve. Storing an unused one would mislead
    any code that inspects, pickles, or hashes the cache.

    Why this is the only regression test for that guard
    ---------------------------------------------------
    The dispatcher's append branch is gated on ``built is not None`` —
    ``built`` comes from ``demean_within`` and is ``None`` precisely when
    the single-FE MAP fallback ran. Every *other* path either has
    ``built is not None`` (multi-FE schwarz ran, storing is correct) or has
    no explicit preconditioner to store in the first place. So this is the
    one combination that exercises the guard; without this test a refactor
    that drops the ``built is not None`` check would silently regress the
    cache contract.
    """
    x, flist, weights = demean_data

    # Build a real preconditioner on the full 2-FE design so we have a
    # genuine ``WithinPreconditioner`` to pass back in below. We only need
    # the third return value; the demeaned output is incidental.
    _, _, preconditioner = demean_within(
        x=x,
        flist=flist.astype(np.uint32, copy=False),
        weights=weights,
    )
    assert preconditioner is not None

    # The setup: explicit preconditioner + single-FE design (``flist[:, :1]``
    # slices off the second factor). ``demean_within`` will route this to
    # the MAP fallback and the preconditioner will go unused.
    store: list = []
    _, success = dispatch_demean(
        x=x,
        flist=flist[:, :1],
        weights=weights,
        demeaner=LsmrDemeaner(backend="within", preconditioner=preconditioner),
        preconditioner_store=store,
    )
    assert success
    # The load-bearing assertion: the cache stayed empty even though an
    # explicit preconditioner was passed. If this ever fails, the dispatcher
    # is appending things that never touched the data.
    assert store == [], "unused explicit preconditioner must not pollute the cache"


def test_demean_within_unpreconditioned_matches_pyhdfe(demean_data):
    x, flist, weights = demean_data

    expected = pyhdfe.create(flist).residualize(x, weights.reshape((x.shape[0], 1)))
    result, success, built = demean_within(
        x=x,
        flist=flist.astype(np.uint32, copy=False),
        weights=weights,
        preconditioner="none",
        tol=1e-10,
        maxiter=10_000,
    )

    assert success
    assert built is None  # preconditioner="none" never builds one
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    ("backend", "requested", "expected"),
    [
        # within: supports schwarz, none, diag; auto -> schwarz
        ("within", "auto", "schwarz"),
        ("within", "schwarz", "schwarz"),
        ("within", "none", "none"),
        ("within", "diag", "diag"),
        # torch: supports diag; auto -> diag
        ("torch", "auto", "diag"),
        ("torch", "diag", "diag"),
        # cupy: supports diag, none; auto -> diag
        ("cupy", "auto", "diag"),
        ("cupy", "diag", "diag"),
        ("cupy", "none", "none"),
    ],
)
def test_resolve_preconditioner_compatible_silent(backend, requested, expected):
    """Compatible (and ``auto``) requests resolve without warning."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        assert _resolve_preconditioner(backend, requested) == expected


@pytest.mark.parametrize(
    ("backend", "requested", "fallback"),
    [
        ("torch", "schwarz", "diag"),
        ("torch", "none", "diag"),
        ("cupy", "schwarz", "diag"),
    ],
)
def test_resolve_preconditioner_incompatible_warns(backend, requested, fallback):
    """Unsupported requests warn and fall back to the backend's natural default."""
    with pytest.warns(
        UserWarning,
        match=rf"{requested!r}.*{backend!r}.*{fallback!r}",
    ):
        assert _resolve_preconditioner(backend, requested) == fallback


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_sparse_dummy_reencodes_non_contiguous_groups():
    from pyfixest.estimation.torch._sparse_dummy import _build_sparse_dummy
    from tests._torch_test_utils import torch

    flist = np.array([[2], [9], [2], [5]], dtype=np.uint64)

    D = _build_sparse_dummy(flist, torch.device("cpu"), torch.float64).to_dense()
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    assert torch.equal(D, expected)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_feols_warns_for_experimental_torch_demeaner():
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 1.0, 0.0, 1.0],
            "fe": [0, 0, 1, 1],
        }
    )

    with pytest.warns(UserWarning, match="experimental"):
        pf.feols(
            "y ~ x | fe",
            data=data,
            demeaner=LsmrDemeaner(backend="torch", device="cpu"),
        )


@pytest.mark.parametrize("demeaner", MODEL_DEMEANERS)
def test_demean_model_no_fixed_effects(benchmark, demeaner):
    """Test demean_model when there are no fixed effects."""
    # Create sample data
    N = 1000
    Y = pd.DataFrame({"y": np.random.randn(N)})
    X = pd.DataFrame({"x1": np.random.randn(N), "x2": np.random.randn(N)})
    weights = np.ones(N)
    lookup_dict = {}

    # Test without fixed effects
    Yd, Xd = benchmark(
        demean_model,
        Y=Y,
        X=X,
        fe=None,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        demeaner=demeaner,
    )

    # When no fixed effects, output should equal input
    assert np.allclose(Y.values, Yd.values)
    assert np.allclose(X.values, Xd.values)
    assert Yd.columns.equals(Y.columns)
    assert Xd.columns.equals(X.columns)


@pytest.mark.parametrize("demeaner", MODEL_DEMEANERS)
def test_demean_model_with_fixed_effects(benchmark, demeaner):
    """Test demean_model with fixed effects."""
    # Create sample data
    N = 1000
    rng = np.random.default_rng(42)

    Y = pd.DataFrame({"y": rng.normal(0, 1, N)})
    X = pd.DataFrame({"x1": rng.normal(0, 1, N), "x2": rng.normal(0, 1, N)})
    fe = pd.DataFrame({"fe1": rng.integers(0, 10, N), "fe2": rng.integers(0, 5, N)})
    weights = np.ones(N)
    lookup_dict = {}

    # Run demean_model
    Yd, Xd = benchmark(
        demean_model,
        Y=Y,
        X=X,
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        demeaner=demeaner,
    )

    # Verify results are different from input (since we're demeaning)
    assert not np.allclose(Y.values, Yd.values)
    assert not np.allclose(X.values, Xd.values)

    # Verify column names are preserved
    assert Yd.columns.equals(Y.columns)
    assert Xd.columns.equals(X.columns)

    # Verify results are cached in lookup_dict
    assert frozenset() in lookup_dict
    cached_data = lookup_dict[frozenset()]
    assert np.allclose(cached_data[Y.columns].values, Yd.values)
    assert np.allclose(cached_data[X.columns].values, Xd.values)


@pytest.mark.parametrize("demeaner", MODEL_DEMEANERS)
def test_demean_model_with_weights(benchmark, demeaner):
    """Test demean_model with weights."""
    N = 1000
    rng = np.random.default_rng(42)

    Y = pd.DataFrame({"y": rng.normal(0, 1, N)})
    X = pd.DataFrame({"x1": rng.normal(0, 1, N), "x2": rng.normal(0, 1, N)})
    fe = pd.DataFrame({"fe1": rng.integers(0, 10, N)})
    weights = rng.uniform(0.5, 1.5, N)
    lookup_dict = {}

    # Run with weights
    Yd, Xd = benchmark(
        demean_model,
        Y=Y,
        X=X,
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        demeaner=demeaner,
    )

    # Run without weights for comparison (fresh lookup dict to avoid cache hit)
    Yd_unweighted, Xd_unweighted = demean_model(
        Y=Y,
        X=X,
        fe=fe,
        weights=np.ones(N),
        lookup_demeaned_data={},
        na_index=frozenset(),
        demeaner=demeaner,
    )

    # Results should be different with weights vs without
    assert not np.allclose(Yd.values, Yd_unweighted.values)
    assert not np.allclose(Xd.values, Xd_unweighted.values)


@pytest.mark.parametrize("demeaner", MODEL_DEMEANERS)
def test_demean_model_caching(benchmark, demeaner):
    """Test the caching behavior of demean_model."""
    N = 1000
    rng = np.random.default_rng(42)

    Y = pd.DataFrame({"y": rng.normal(0, 1, N)})
    X = pd.DataFrame({"x1": rng.normal(0, 1, N), "x2": rng.normal(0, 1, N)})
    fe = pd.DataFrame({"fe1": rng.integers(0, 10, N)})
    weights = np.ones(N)
    lookup_dict = {}

    # First run - should compute and cache
    Yd1, Xd1 = demean_model(
        Y=Y,
        X=X,
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        demeaner=demeaner,
    )

    # Second run - should use cache
    Yd2, Xd2 = benchmark(
        demean_model,
        Y=Y,
        X=X,
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        demeaner=demeaner,
    )

    # Results should be identical
    assert np.allclose(Yd1.values, Yd2.values)
    assert np.allclose(Xd1.values, Xd2.values)

    # Add new variable and verify partial caching
    X_new = X.copy()
    X_new["x3"] = rng.normal(0, 1, N)

    _, Xd3 = demean_model(
        Y=Y,
        X=X_new,
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        demeaner=demeaner,
    )

    # Original columns should match previous results
    assert np.allclose(Xd3[["x1", "x2"]].values, Xd2.values)
    # New column should be different
    assert "x3" in Xd3.columns


@pytest.mark.parametrize(
    "demeaner",
    [
        pytest.param(MapDemeaner(backend="numba", fixef_maxiter=1), id="numba"),
        pytest.param(MapDemeaner(backend="jax", fixef_maxiter=1), id="jax"),
        pytest.param(MapDemeaner(backend="rust", fixef_maxiter=1), id="rust"),
        pytest.param(
            LsmrDemeaner(backend="cupy", device="cpu", fixef_maxiter=1),
            id="lsmr_scipy",
        ),
        pytest.param(
            LsmrDemeaner(backend="torch", device="cpu", fixef_maxiter=1),
            id="lsmr_torch_cpu",
        ),
    ],
)
def test_demean_model_maxiter_convergence_failure(demeaner):
    """Test that demean_model fails when maxiter is too small."""
    N = 100
    rng = np.random.default_rng(42)

    Y = pd.DataFrame({"y": rng.normal(0, 1, N)})
    X = pd.DataFrame({"x1": rng.normal(0, 1, N)})
    # Many fixed effects to make convergence difficult
    fe = pd.DataFrame(
        {"fe1": rng.choice(N // 10, N), "fe2": rng.choice(N // 10, N)}
    )  # Each obs is its own FE
    weights = np.ones(N)
    lookup_dict = {}

    # Should fail with very small maxiter
    with pytest.raises(ValueError, match="Demeaning failed after 1 iterations"):
        demean_model(
            Y=Y,
            X=X,
            fe=fe,
            weights=weights,
            lookup_demeaned_data=lookup_dict,
            na_index=frozenset(),
            demeaner=demeaner,
        )


def test_feols_integration_maxiter():
    """Integration test: Test fixef_maxiter flows from demeaner to demean."""
    import pyfixest as pf

    N = 1000  # More observations
    rng = np.random.default_rng(42)

    # Create data with many (but not N) fixed effects
    data = pd.DataFrame(
        {
            "y": rng.normal(0, 1, N),
            "x": rng.normal(0, 1, N),
            "fe": rng.integers(0, 50, N),  # 50 fixed effects, not N
        }
    )

    # Should fail with tiny maxiter
    with pytest.raises(ValueError, match="Demeaning failed after 1 iterations"):
        pf.feols("y ~ x | fe", data=data, demeaner=MapDemeaner(fixef_maxiter=1))

    # Should work with default
    model = pf.feols("y ~ x | fe", data=data)
    assert model is not None


@pytest.mark.parametrize("demeaner", MODEL_DEMEANERS)
def test_demean_complex_fixed_effects(benchmark, demeaner):
    """Benchmark demean functions with complex multi-level fixed effects."""
    X, flist, weights = generate_complex_fixed_effects_data()

    X_demeaned, success = benchmark.pedantic(
        dispatch_demean,
        args=(X, flist, weights, demeaner),
        iterations=1,
        rounds=1,
        warmup_rounds=0,
    )

    assert success, "Benchmarked demeaning should succeed"
    assert X_demeaned.shape == X.shape


def generate_complex_fixed_effects_data():
    """
    Complex fixed effects example ported from fixest R-implementation:
    https://github.com/lrberge/fixest/blob/ac1be27fda5fc381c0128b861eaf5bda88af846c/_BENCHMARK/Data%20generation.R#L125 .

    """
    rng = np.random.default_rng(42)
    n = 10**5  # Large dataset for benchmarking
    nb_indiv = n // 20
    nb_firm = max(1, round(n / 160))
    nb_year = max(1, round(n**0.3))
    # Generate fixed effect IDs
    id_indiv = rng.choice(nb_indiv, n, replace=True)
    id_firm_base = rng.integers(0, 21, n) + np.maximum(1, id_indiv // 8 - 10)
    id_firm = np.minimum(id_firm_base, nb_firm - 1)
    id_year = rng.choice(nb_year, n, replace=True)
    # Create variables
    x1 = (
        5 * np.cos(id_indiv)
        + 5 * np.sin(id_firm)
        + 5 * np.sin(id_year)
        + rng.uniform(0, 1, n)
    )
    x2 = np.cos(id_indiv) + np.sin(id_firm) + np.sin(id_year) + rng.normal(0, 1, n)
    y = (
        3 * x1
        + 5 * x2
        + np.cos(id_indiv)
        + np.cos(id_firm) ** 2
        + np.sin(id_year)
        + rng.normal(0, 1, n)
    )
    X = np.column_stack([x1, x2, y])
    flist = np.column_stack([id_indiv, id_firm, id_year]).astype(np.uint64)
    weights = rng.uniform(0.5, 2.0, n)
    return X, flist, weights
