from __future__ import annotations

import warnings

from pyfixest.demeaners import AnyDemeaner, LsmrDemeaner, MapDemeaner, WithinDemeaner


def _resolve_demeaner(
    *,
    demeaner: AnyDemeaner | None,
    demeaner_backend: str | None = None,
    fixef_tol: float | None = None,
    fixef_maxiter: int | None = None,
) -> AnyDemeaner:
    legacy_args_used = any(
        arg is not None for arg in (demeaner_backend, fixef_tol, fixef_maxiter)
    )

    if demeaner is not None:
        if legacy_args_used:
            raise ValueError(
                "Please either pass a typed `demeaner` or the deprecated legacy arguments "
                "`demeaner_backend`, `fixef_tol`, and `fixef_maxiter`, not both. Here we prioritize the typed `demeaner` argument and ignore the legacy arguments, but in a future release passing legacy arguments with a typed `demeaner` will raise an error."
                ""
            )
        return demeaner

    if not legacy_args_used:
        return MapDemeaner()

    if fixef_tol is not None:
        if isinstance(fixef_tol, bool) or not isinstance(fixef_tol, float):
            raise TypeError("fixef_tol must be a float")
        if fixef_tol <= 0 or fixef_tol >= 1:
            raise ValueError("fixef_tol must be greater than zero and less than one")

    warnings.warn(
        (
            "The `demeaner_backend`, `fixef_tol`, and `fixef_maxiter` arguments "
            "are deprecated and will be removed in a future release. "
            "Pass a typed `demeaner=` configuration instead."
        ),
        DeprecationWarning,
        stacklevel=3,
    )

    backend = "numba" if demeaner_backend is None else demeaner_backend

    if backend in {"numba", "rust", "jax"}:
        kwargs = {"backend": backend}
        if fixef_tol is not None:
            kwargs["fixef_tol"] = fixef_tol
        if fixef_maxiter is not None:
            kwargs["fixef_maxiter"] = fixef_maxiter
        return MapDemeaner(**kwargs)

    if backend in {"rust-cg", "within"}:
        kwargs = {}
        if fixef_tol is not None:
            kwargs["fixef_tol"] = fixef_tol
        if fixef_maxiter is not None:
            kwargs["fixef_maxiter"] = fixef_maxiter
        return WithinDemeaner(**kwargs)

    lsmr_kwargs = {}
    if fixef_tol is not None:
        lsmr_kwargs["fixef_atol"] = fixef_tol
        lsmr_kwargs["fixef_btol"] = fixef_tol
    if fixef_maxiter is not None:
        lsmr_kwargs["fixef_maxiter"] = fixef_maxiter

    if backend in {"cupy", "cupy64"}:
        return LsmrDemeaner(
            backend="cupy",
            device="cuda",
            precision="float64",
            **lsmr_kwargs,
        )
    if backend == "cupy32":
        return LsmrDemeaner(
            backend="cupy",
            device="cuda",
            precision="float32",
            **lsmr_kwargs,
        )
    if backend == "scipy":
        return LsmrDemeaner(
            backend="cupy",
            device="cpu",
            precision="float64",
            **lsmr_kwargs,
        )
    if backend == "torch":
        return LsmrDemeaner(
            backend="torch",
            device="auto",
            precision="float64",
            **lsmr_kwargs,
        )
    if backend == "torch_cpu":
        return LsmrDemeaner(
            backend="torch",
            device="cpu",
            precision="float64",
            **lsmr_kwargs,
        )
    if backend == "torch_mps":
        return LsmrDemeaner(
            backend="torch",
            device="mps",
            precision="float32",
            **lsmr_kwargs,
        )
    if backend == "torch_cuda":
        return LsmrDemeaner(
            backend="torch",
            device="cuda",
            precision="float64",
            **lsmr_kwargs,
        )
    if backend == "torch_cuda32":
        return LsmrDemeaner(
            backend="torch",
            device="cuda",
            precision="float32",
            **lsmr_kwargs,
        )

    raise ValueError(f"Invalid demeaner backend: {backend}")


def _warn_if_experimental_torch_demeaner(demeaner: object) -> None:
    if isinstance(demeaner, LsmrDemeaner) and demeaner.backend == "torch":
        warnings.warn(
            (
                "The torch LSMR demeaner backend is experimental. "
                "Behavior and performance may change in future releases."
            ),
            UserWarning,
            stacklevel=3,
        )
