"""
Modal-backed remote runner for GPU benchmarks.

Dispatches benchmark work to a remote Modal container with a GPU,
then returns results locally. The returned callable has the same
signature as ``_run_scenario_suite`` in ``run_benchmarks.py``.

Usage (from run_benchmarks.py)::

    from benchmarks.modal_runner import create_runner
    _run_suite = create_runner("T4")
    results = _run_suite(scenarios, n_reps, backends, run_feols)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.bench import BenchmarkResult
    from benchmarks.dgp import DGPConfig


def create_runner(gpu_type: str = "T4"):
    """Return a function that runs benchmarks on a remote Modal GPU.

    Parameters
    ----------
    gpu_type : str
        Modal GPU type, e.g. "T4", "A10G", "L4", "A100", "H100".

    Returns
    -------
    Callable
        A function with the same signature as ``_run_scenario_suite``:
        ``(scenarios, n_reps, backends, run_feols, n_features=1,
        fe_columns=None) -> list[BenchmarkResult]``
    """
    import modal

    app = modal.App("pyfixest-benchmarks")

    benchmarks_path = modal.Mount.from_local_dir(
        local_path="benchmarks",
        remote_path="/root/project/benchmarks",
    )

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "pyfixest",
            "jax[cuda12]",
            "cupy-cuda12x",
            "tabulate",
        )
    )

    @app.function(
        image=image,
        gpu=gpu_type,
        mounts=[benchmarks_path],
        timeout=3600,
    )
    def _remote_run(
        scenario_dicts: dict[str, dict],
        n_reps: int,
        backends: list[str],
        run_feols: bool,
        n_features: int,
        fe_columns: list[str] | None,
    ) -> list[dict]:
        import sys

        sys.path.insert(0, "/root/project")

        from benchmarks.bench import run_benchmark
        from benchmarks.dgp import DGPConfig

        all_results = []
        total = len(scenario_dicts)
        for i, (name, cfg_dict) in enumerate(scenario_dicts.items(), 1):
            print(f"\n{'='*60}")
            print(f"Scenario {i}/{total}: {name}")
            print(f"{'='*60}")
            config = DGPConfig(**cfg_dict)
            results = run_benchmark(
                config,
                name,
                n_repetitions=n_reps,
                backends=backends,
                run_feols=run_feols,
                n_features=n_features,
                fe_columns=fe_columns,
            )
            all_results.extend(results)

        from dataclasses import asdict

        return [asdict(r) for r in all_results]

    def run_suite(
        scenarios: dict[str, DGPConfig],
        n_reps: int,
        backends: list[str],
        run_feols: bool,
        n_features: int = 1,
        fe_columns: list[str] | None = None,
    ) -> list[BenchmarkResult]:
        """Run benchmarks remotely on Modal and return local results."""
        from benchmarks.bench import BenchmarkResult
        from benchmarks.dgp import DGPConfig as DGPConfigCls

        scenario_dicts = {
            name: asdict(config) for name, config in scenarios.items()
        }

        with app.run():
            result_dicts = _remote_run.remote(
                scenario_dicts,
                n_reps,
                backends,
                run_feols,
                n_features,
                fe_columns,
            )

        results = []
        for d in result_dicts:
            config = DGPConfigCls(**d.pop("config"))
            results.append(BenchmarkResult(config=config, **d))
        return results

    return run_suite
