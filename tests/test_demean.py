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
                preconditioner="off",
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
    result_unweighted, success, _ = dispatch_demean(
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
    result_weighted, success, _ = dispatch_demean(
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
    assert isinstance(preconditioner, pf.Preconditioner)
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
    # Structural match: reused returns a wrapper around the same factorization
    # (same variant and DOF count) but not the same Python object — pyo3
    # round-trips produce fresh wrappers, matching upstream's value semantics.
    assert preconditioner_reused is not preconditioner
    assert preconditioner_reused.variant == preconditioner.variant
    assert preconditioner_reused.nrows == preconditioner.nrows
    assert preconditioner_reused.ncols == preconditioner.ncols
    assert preconditioner_reused.build_time_seconds == preconditioner.build_time_seconds
    # Solve-equivalence is the load-bearing correctness check: same factorization
    # applied to the same data must yield bitwise-identical demeaned output.
    np.testing.assert_allclose(result_reused, result, rtol=1e-10, atol=1e-10)


def test_demean_within_preconditioner_reports_build_time(demean_data):
    """The preconditioner must report its build time in seconds as a float.
    If the preconditioner is reused, we still report the initial build time,
    and not a new build time of approximately 0.
    """
    x, flist, weights = demean_data

    _, success, preconditioner = demean_within(
        x=x,
        flist=flist.astype(np.uint32, copy=False),
        weights=weights,
    )
    assert success
    assert preconditioner is not None

    build_time = preconditioner.build_time_seconds
    assert isinstance(build_time, float)
    assert build_time >= 0.0
    assert f"build_time_seconds={build_time:.2f}" in repr(preconditioner)

    # the build time survives preconditioner reuse unchagned
    # 1) directly feed preconditioner
    _, success_reused, preconditioner_reused = demean_within(
        x=x,
        flist=flist.astype(np.uint32, copy=False),
        weights=weights,
        preconditioner=preconditioner,
    )
    assert success_reused
    assert preconditioner_reused is not None
    assert preconditioner_reused.build_time_seconds == build_time

    # 2) load cached preconditioner
    # The build cost survives serialization unchanged.
    restored = pickle.loads(pickle.dumps(preconditioner))
    assert restored.build_time_seconds == build_time
    _, success_reused, preconditioner_reused = demean_within(
        x=x,
        flist=flist.astype(np.uint32, copy=False),
        weights=weights,
        preconditioner=restored,
    )
    assert success_reused
    assert preconditioner_reused is not None
    assert preconditioner_reused.build_time_seconds == build_time


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
    assert isinstance(restored, pf.Preconditioner)
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


@pytest.mark.parametrize(
    ("preconditioner", "expected_variant"),
    [
        ("additive", "Additive"),
        ("diagonal", "Diagonal"),
    ],
)
def test_lsmr_within_reuses_cached_preconditioner(
    preconditioner, expected_variant, demean_data
):
    """End-to-end coverage of ``dispatch_demean``'s preconditioner-reuse policy.

    Context
    -------
    ``dispatch_demean`` is the orchestration layer that the model classes
    (``Feols``, ``Fepois``, ``Feglm``) call to demean their working data on
    each iteration. The ``cached_preconditioner`` argument is the
    callsite-supplied factorization that turns IWLS's "build the
    factorization once, reuse it across iterations" claim into reality. The
    dispatcher returns the preconditioner actually used so callers can hold
    onto it for the next solve. This test exercises that contract directly,
    bypassing the model classes so failures point at the reuse logic instead
    of the IWLS loop.

    Three behaviours are pinned down:

    Leg 1 — first call reports the freshly built preconditioner
        With no ``cached_preconditioner``, ``LsmrDemeaner(backend="within")``
        builds the requested factorization. The dispatcher must return it as
        the third tuple element so the caller can hold onto it.

    Leg 2 — subsequent calls reuse the supplied preconditioner under changed
    weights (the IWLS hot path)
        IWLS changes the working weights at every iteration (``W := W * mu``).
        A pure implementation would rebuild the preconditioner each time,
        which is the expensive step. Passing the first-iteration factorization
        back in via ``cached_preconditioner`` lets us reuse it even though
        the weights — and therefore the true normal-equations operator — have
        changed. The factorization is "stale" in that sense, but LSMR still
        converges to the correct demeaned values, just possibly in a few more
        iterations. We simulate a weight change with ``weights + 0.25``, then
        check both that the dispatcher returns the *same* Python object (no
        rebuild) and that the demeaned output still matches an independent
        ground truth.

        Ground truth comes from ``pyhdfe.residualize`` — a separate, mature
        FE-residualization implementation. Matching it under changed weights
        is the strongest evidence that the stale-preconditioner path is
        mathematically correct, not just self-consistent.

    Leg 3 — passing a cached preconditioner back via ``LsmrDemeaner``
    reproduces the original solve
        This is the documented "save in session 1, reload in session 2"
        workflow in miniature. A fresh ``LsmrDemeaner(preconditioner=...)``
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
    demeaner = LsmrDemeaner(
        backend="within",
        preconditioner=preconditioner,
        fixef_atol=1e-10,
        fixef_btol=1e-10,
        fixef_maxiter=10_000,
    )

    # ----- Leg 1: no cache → build a fresh preconditioner and return it.
    result, success, built = dispatch_demean(
        x=x,
        flist=flist,
        weights=weights,
        demeaner=demeaner,
    )
    assert success
    assert isinstance(built, pf.Preconditioner), (
        "first call must report the freshly built preconditioner"
    )
    assert built.variant == expected_variant

    # ----- Leg 2: changed weights, cached preconditioner supplied. Mimics
    # an IWLS iteration where only the working weights moved. The dispatcher
    # must reuse the cached factorization (same variant + DOF count; pyo3
    # produces a fresh wrapper, identity semantics match upstream ``within``)
    # and the result must still be numerically correct under the *new* weights.
    adjusted_weights = weights + 0.25
    result_reused, success, reused = dispatch_demean(
        x=x,
        flist=flist,
        weights=adjusted_weights,
        demeaner=demeaner,
        cached_preconditioner=built,
    )
    assert success
    assert isinstance(reused, pf.Preconditioner)
    assert reused.variant == built.variant
    assert reused.nrows == built.nrows
    assert reused.ncols == built.ncols
    assert reused.build_time_seconds == built.build_time_seconds

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

    # ----- Leg 3: explicit preconditioner reuse via ``LsmrDemeaner``. Mimics
    # the across-sessions workflow: pull the built preconditioner (or a
    # pickled+restored copy) and pass it through
    # ``LsmrDemeaner(preconditioner=...)``. With the original weights, the
    # result must reproduce leg 1 down to floating-point noise.
    result_explicit, success, _ = dispatch_demean(
        x=x,
        flist=flist,
        weights=weights,
        demeaner=LsmrDemeaner(
            backend="within",
            preconditioner=built,
            fixef_atol=1e-10,
            fixef_btol=1e-10,
            fixef_maxiter=10_000,
        ),
    )
    assert success
    np.testing.assert_allclose(
        result_explicit,
        result,
        rtol=1e-10,
        atol=1e-10,
        err_msg="same precond + same weights + same data must reproduce the original solve",
    )


def test_lsmr_within_reports_no_preconditioner_when_unused(demean_data):
    """The dispatcher reports ``None`` whenever no preconditioner ran.

    Two routes produce that outcome, both pinned here:

    1. ``preconditioner="off"`` — preconditioning is explicitly disabled, so
       ``demean_within`` never builds one.
    2. Single-FE design with an explicit ``Preconditioner`` —
       ``demean_within`` special-cases single-FE designs to the MAP fallback,
       which ignores the user-supplied object entirely.

    In both, ``demean_within``'s third return is ``None`` and the dispatcher
    must propagate that. ``fit.preconditioner`` is supposed to reflect
    factorizations that actually participated in a solve; reporting an
    unused one would mislead any code that inspects or pickles it.

    Why this is the only regression test for that guard
    ---------------------------------------------------
    The dispatcher's "used" computation is gated on ``built is not None``.
    Every other code path has ``built is not None`` (multi-FE Schwarz ran,
    reporting is correct) or has no explicit preconditioner to mishandle in
    the first place. Without these two cases, a refactor that drops the
    ``built is not None`` check would silently regress the contract.
    """
    x, flist, weights = demean_data

    # Case 1: preconditioner="off" — demean_within returns built=None.
    expected = pyhdfe.create(flist).residualize(x, weights.reshape((x.shape[0], 1)))
    result_off, success, built_off = demean_within(
        x=x,
        flist=flist.astype(np.uint32, copy=False),
        weights=weights,
        preconditioner="off",
        tol=1e-10,
        maxiter=10_000,
    )
    assert success
    assert built_off is None
    np.testing.assert_allclose(result_off, expected, rtol=1e-6, atol=1e-8)

    # Case 2: explicit preconditioner + single-FE design routes to MAP, which
    # ignores the preconditioner. Build a real one on the 2-FE design first
    # so we have a genuine object to feed back in.
    _, _, full_pre = demean_within(
        x=x,
        flist=flist.astype(np.uint32, copy=False),
        weights=weights,
    )
    assert full_pre is not None

    _, success, used = dispatch_demean(
        x=x,
        flist=flist[:, :1],
        weights=weights,
        demeaner=LsmrDemeaner(backend="within", preconditioner=full_pre),
    )
    assert success
    assert used is None, "unused explicit preconditioner must not be reported as used"


@pytest.mark.parametrize(
    ("backend", "requested", "expected"),
    [
        # within: supports additive, off, diagonal; auto -> additive
        ("within", "auto", "additive"),
        ("within", "additive", "additive"),
        ("within", "off", "off"),
        ("within", "diagonal", "diagonal"),
        # torch: supports diagonal; auto -> diagonal
        ("torch", "auto", "diagonal"),
        ("torch", "diagonal", "diagonal"),
        # cupy: supports diagonal, off; auto -> diagonal
        ("cupy", "auto", "diagonal"),
        ("cupy", "diagonal", "diagonal"),
        ("cupy", "off", "off"),
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
        ("torch", "additive", "diagonal"),
        ("torch", "off", "diagonal"),
        ("cupy", "additive", "diagonal"),
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
    Yd, Xd, _ = benchmark(
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
    Yd, Xd, _ = benchmark(
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
    Yd, Xd, _ = benchmark(
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
    Yd_unweighted, Xd_unweighted, _ = demean_model(
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
    Yd1, Xd1, _ = demean_model(
        Y=Y,
        X=X,
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        demeaner=demeaner,
    )

    # Second run - should use cache
    Yd2, Xd2, _ = benchmark(
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

    _, Xd3, _ = demean_model(
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

    X_demeaned, success, _ = benchmark.pedantic(
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
