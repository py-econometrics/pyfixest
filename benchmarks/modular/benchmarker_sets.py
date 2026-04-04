from __future__ import annotations

from dataclasses import dataclass

from feols_benchmarkers import (
    FixestFeolsBenchmarker,
    JuliaFeolsBenchmarker,
    PyFeolsBenchmarkerFullApi,
    detect_torch_runtime_availability,
)


@dataclass(frozen=True)
class BenchmarkerBundle:
    benchmarkers: list
    figure_backends: list[str]


def build_standard_feols_benchmarkers(
    *,
    fixef_maxiter: int | None = None,
    include_fixest: bool = True,
    include_julia: bool = True,
    include_torch: bool = True,
) -> BenchmarkerBundle:
    """Build the shared feols benchmark runner set used by modular benchmarks."""
    pyfixest_kwargs = {}
    if fixef_maxiter is not None:
        pyfixest_kwargs["fixef_maxiter"] = fixef_maxiter

    pyfixest_benchmarkers = [
        PyFeolsBenchmarkerFullApi("pyfixest (rust-cg)", "rust-cg", **pyfixest_kwargs),
        PyFeolsBenchmarkerFullApi("pyfixest (rust-map)", "rust", **pyfixest_kwargs),
    ]

    if include_torch:
        availability = detect_torch_runtime_availability()
        if not availability.has_torch:
            print(
                "[bench] skipping torch benchmarkers: torch is not installed",
                flush=True,
            )
        else:
            pyfixest_benchmarkers.append(
                PyFeolsBenchmarkerFullApi(
                    "pyfixest (torch-cpu)",
                    "torch_cpu",
                    **pyfixest_kwargs,
                )
            )
            if availability.has_mps:
                pyfixest_benchmarkers.append(
                    PyFeolsBenchmarkerFullApi(
                        "pyfixest (torch-mps)",
                        "torch_mps",
                        **pyfixest_kwargs,
                    )
                )
            else:
                print(
                    "[bench] skipping torch-mps benchmarker: MPS unavailable",
                    flush=True,
                )

            if availability.has_cuda:
                pyfixest_benchmarkers.append(
                    PyFeolsBenchmarkerFullApi(
                        "pyfixest (torch-cuda)",
                        "torch_cuda",
                        **pyfixest_kwargs,
                    )
                )
            else:
                print(
                    "[bench] skipping torch-cuda benchmarker: CUDA unavailable",
                    flush=True,
                )

    benchmarkers = list(pyfixest_benchmarkers)
    if include_fixest:
        benchmarkers.append(FixestFeolsBenchmarker("fixest-map"))
    if include_julia:
        benchmarkers.append(JuliaFeolsBenchmarker("FEM.jl (lsmr)"))

    return BenchmarkerBundle(
        benchmarkers=benchmarkers,
        figure_backends=[b.name for b in pyfixest_benchmarkers],
    )
