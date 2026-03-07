from __future__ import annotations

import json
import statistics
import subprocess
import tempfile
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    from .interfaces import BenchmarkDataset, FeolsResult, FeolsSpec
except ImportError:
    from interfaces import BenchmarkDataset, FeolsResult, FeolsSpec

_TABLE_HDR = f"{'dgp':<16} {'n_obs':>12} {'n_fe':>4} {'min':>10} {'median':>10} {'max':>10}  status"
_TABLE_SEP = "-" * len(_TABLE_HDR)

SUBSTEP_KEYS = ("model_matrix", "demean", "solve", "vcov")

_SUBSTEP_HDR = f"{'dgp':<16} {'n_obs':>12} {'n_fe':>4} {'matrix':>10} {'demean':>10} {'solve':>10} {'vcov':>10}  status"
_SUBSTEP_SEP = "-" * len(_SUBSTEP_HDR)


def _fmt_time(t: float) -> str:
    if t < 1:
        return f"{t * 1000:.1f}ms"
    return f"{t:.3f}s"


def _print_header(name: str) -> None:
    print(f"\n  {name}", flush=True)
    print(f"  {_TABLE_SEP}", flush=True)
    print(f"  {_TABLE_HDR}", flush=True)
    print(f"  {_TABLE_SEP}", flush=True)


def _print_substep_header(name: str) -> None:
    print(f"\n  {name} [substeps]", flush=True)
    print(f"  {_SUBSTEP_SEP}", flush=True)
    print(f"  {_SUBSTEP_HDR}", flush=True)
    print(f"  {_SUBSTEP_SEP}", flush=True)


# --- A3: Deduplicated print functions ---


def _time_columns(results: list[FeolsResult]) -> tuple[str, str]:
    times = [r.time for r in results if r.success and r.time is not None]
    if times:
        mn, md, mx = min(times), statistics.median(times), max(times)
        columns = f"{_fmt_time(mn):>10} {_fmt_time(md):>10} {_fmt_time(mx):>10}"
        return columns, "ok"
    errs = [r.error for r in results if r.error]
    status = errs[0][:30] if errs else "FAIL"
    columns = f"{'—':>10} {'—':>10} {'—':>10}"
    return columns, status


def _substep_columns(results: list[FeolsResult]) -> tuple[str, str]:
    substep_results = [r for r in results if r.success and r.substeps is not None]
    if substep_results:
        steps = {}
        for key in SUBSTEP_KEYS:
            vals = [r.substeps[key] for r in substep_results if key in r.substeps]
            steps[key] = statistics.median(vals) if vals else 0.0
        columns = (
            f"{_fmt_time(steps['model_matrix']):>10} "
            f"{_fmt_time(steps['demean']):>10} "
            f"{_fmt_time(steps['solve']):>10} "
            f"{_fmt_time(steps['vcov']):>10}"
        )
        return columns, "ok"
    columns = f"{'—':>10} {'—':>10} {'—':>10} {'—':>10}"
    return columns, "FAIL"


def _print_formatted_row(
    results: list[FeolsResult],
    compute_columns: Callable[[list[FeolsResult]], tuple[str, str]],
) -> None:
    r0 = results[0]
    prefix = f"{r0.dgp:<16} {r0.n_obs:>12,} {r0.n_fe:>4}"
    columns, status = compute_columns(results)
    print(f"  {prefix} {columns}  {status}", flush=True)


def _print_row(results: list[FeolsResult]) -> None:
    _print_formatted_row(results, _time_columns)


def _print_substep_row(results: list[FeolsResult]) -> None:
    _print_formatted_row(results, _substep_columns)


def _group_key(r: FeolsResult) -> tuple[str, int, int]:
    return (r.dgp, r.n_obs, r.n_fe)


def _flush_groups(
    results: list[FeolsResult],
    print_fn: Callable[[list[FeolsResult]], None],
    *,
    skip_burnin: bool = True,
) -> None:
    group_buf: list[FeolsResult] = []
    prev_key: tuple | None = None
    for r in results:
        if skip_burnin and r.iter_type == "burnin":
            continue
        key = _group_key(r)
        if prev_key is not None and key != prev_key:
            if group_buf:
                print_fn(group_buf)
            group_buf = []
        group_buf.append(r)
        prev_key = key
    if group_buf:
        print_fn(group_buf)


@dataclass(frozen=True)
class RunOutcome:
    """Outcome of a single benchmark run."""

    elapsed: float | None
    success: bool
    error: str | None = None
    substeps: dict[str, float] | None = None
    n_obs_override: int | None = None


def _result_from_dataset(
    dataset: BenchmarkDataset,
    spec: FeolsSpec,
    *,
    backend: str,
    outcome: RunOutcome,
) -> FeolsResult:
    return FeolsResult(
        dataset_id=dataset.dataset_id,
        iter_type=dataset.iter_type,
        iter_num=dataset.iter_num,
        dgp=dataset.dgp,
        n_obs=outcome.n_obs_override
        if outcome.n_obs_override is not None
        else dataset.n_obs,
        n_fe=spec.n_fe,
        backend=backend,
        time=outcome.elapsed,
        success=outcome.success,
        error=outcome.error,
        substeps=outcome.substeps,
    )


def _normalize_vcov(vcov: str | dict[str, str]) -> str:
    """Normalize vcov spec to a simple string for subprocess backends.

    Returns "iid", "hetero", or "cluster:<colname>".
    """
    if isinstance(vcov, dict) and "CRV1" in vcov:
        return f"cluster:{vcov['CRV1']}"
    return vcov


class PyFeolsBenchmarker:
    """Benchmark the full feols pipeline with a pluggable demean function."""

    def __init__(self, name: str, demean_func: Callable, **demean_kwargs):
        self._name = name
        self._demean_func = demean_func
        self._demean_kwargs = demean_kwargs

    @property
    def name(self) -> str:
        return self._name

    def _build_model_matrix(self, formula, df):
        from pyfixest.estimation.formula.model_matrix import create_model_matrix

        t0 = time.perf_counter()
        mm = create_model_matrix(formula, df)
        Y = mm.dependent
        X = mm.independent
        fe = mm.fixed_effects
        t_mm = time.perf_counter() - t0
        return mm, Y, X, fe, t_mm

    def _demean(self, Y, X, fe):
        t0 = time.perf_counter()
        YX = np.concatenate(
            [Y.to_numpy(dtype=np.float64), X.to_numpy(dtype=np.float64)],
            axis=1,
        )
        weights = np.ones(YX.shape[0], dtype=np.float64)
        if fe is None or fe.shape[1] == 0:
            YX_d = YX
            converged = True
        else:
            fe_arr = fe.to_numpy().astype(np.uint64)
            YX_d, converged = self._demean_func(
                x=YX, flist=fe_arr, weights=weights, **self._demean_kwargs
            )
        t_demean = time.perf_counter() - t0
        return YX_d, converged, t_demean

    def _solve_ols(self, YX_d, n_y):
        from pyfixest.core.collinear import find_collinear_variables

        t0 = time.perf_counter()
        Yd = YX_d[:, :n_y]
        Xd = YX_d[:, n_y:]

        # Drop collinear variables
        tXX = Xd.T @ Xd
        id_excl, n_excl, _ = find_collinear_variables(tXX)
        if n_excl > 0:
            Xd = np.delete(Xd, id_excl[:n_excl], axis=1)
            tXX = Xd.T @ Xd

        tXy = Xd.T @ Yd
        beta_hat = np.linalg.solve(tXX, tXy).flatten()
        u_hat = Yd.flatten() - (Xd @ beta_hat).flatten()
        t_solve = time.perf_counter() - t0
        return Xd, beta_hat, u_hat, tXX, t_solve

    def _compute_vcov(self, spec, df, mm, Xd, u_hat, tXX):
        t0 = time.perf_counter()
        N = Xd.shape[0]
        k = Xd.shape[1]
        bread = np.linalg.inv(tXX)

        if spec.vcov == "iid":
            sigma2 = np.sum(u_hat**2) / (N - k)
            _vcov = sigma2 * bread
        elif spec.vcov == "hetero":
            scores = Xd * u_hat[:, None]
            meat = scores.T @ scores
            _vcov = bread @ meat @ bread
        elif isinstance(spec.vcov, dict) and "CRV1" in spec.vcov:
            cluster_col_name = spec.vcov["CRV1"]
            cluster_col = df[cluster_col_name].to_numpy()
            if mm.na_index:
                keep_mask = np.ones(len(df), dtype=bool)
                keep_mask[np.asarray(mm.na_index)] = False
                cluster_col = cluster_col[keep_mask]
            # A2: cluster alignment assertion
            assert len(cluster_col) == Xd.shape[0], (
                f"cluster_col length {len(cluster_col)} != Xd rows {Xd.shape[0]}"
            )
            scores = Xd * u_hat[:, None]
            # O(n) vectorized cluster aggregation
            cluster_ids, _ = pd.factorize(cluster_col)
            n_clusters = cluster_ids.max() + 1
            cluster_scores = np.zeros((n_clusters, k))
            np.add.at(cluster_scores, cluster_ids, scores)
            meat = cluster_scores.T @ cluster_scores
            _vcov = bread @ meat @ bread
        else:
            # Fallback: iid
            sigma2 = np.sum(u_hat**2) / (N - k)
            _vcov = sigma2 * bread
        t_vcov = time.perf_counter() - t0
        return _vcov, t_vcov

    def run(
        self, datasets: list[BenchmarkDataset], spec: FeolsSpec
    ) -> list[FeolsResult]:
        from pyfixest.estimation.formula.parse import Formula

        parsed_formulas = Formula.parse(spec.formula)
        formula = parsed_formulas[0]

        results: list[FeolsResult] = []

        all_cols = [spec.depvar, *spec.covariates, *spec.fe_cols]
        if isinstance(spec.vcov, dict) and "CRV1" in spec.vcov:
            cluster_col = spec.vcov["CRV1"]
            if cluster_col not in all_cols:
                all_cols.append(cluster_col)

        _print_header(self.name)

        for dataset in datasets:
            n_obs_for_result = dataset.n_obs
            try:
                df = pd.read_parquet(dataset.data_path, columns=all_cols)
                n_obs_for_result = len(df)

                mm, Y, X, fe, t_mm = self._build_model_matrix(formula, df)
                YX_d, converged, t_demean = self._demean(Y, X, fe)
                n_y = Y.shape[1]
                Xd, _beta_hat, u_hat, tXX, t_solve = self._solve_ols(YX_d, n_y)
                _vcov, t_vcov = self._compute_vcov(spec, df, mm, Xd, u_hat, tXX)

                total = t_mm + t_demean + t_solve + t_vcov
                step_times = (t_mm, t_demean, t_solve, t_vcov)
                substeps = dict(zip(SUBSTEP_KEYS, step_times))

                result = _result_from_dataset(
                    dataset,
                    spec,
                    backend=self.name,
                    outcome=RunOutcome(
                        elapsed=total,
                        success=bool(converged),
                        substeps=substeps,
                        n_obs_override=n_obs_for_result,
                    ),
                )
            except Exception as exc:
                result = _result_from_dataset(
                    dataset,
                    spec,
                    backend=self.name,
                    outcome=RunOutcome(
                        elapsed=None,
                        success=False,
                        error=str(exc),
                        n_obs_override=n_obs_for_result,
                    ),
                )

            results.append(result)

        _flush_groups(results, _print_row)

        # Print substep breakdown
        if any(r.substeps for r in results if r.iter_type != "burnin"):
            _print_substep_header(self.name)
            _flush_groups(results, _print_substep_row)

        return results


# ---------------------------------------------------------------------------
# Subprocess-based benchmarkers (R / Julia)
# ---------------------------------------------------------------------------


def _parse_subprocess_output(
    *,
    datasets: list[BenchmarkDataset],
    spec: FeolsSpec,
    backend: str,
    completed_process: subprocess.CompletedProcess[str],
) -> list[FeolsResult]:
    parsed_by_id: dict[str, dict] = {}

    for line in completed_process.stdout.splitlines():
        payload = line.strip()
        if not payload:
            continue
        try:
            entry = json.loads(payload)
        except json.JSONDecodeError:
            continue
        dataset_id = entry.get("dataset_id")
        if isinstance(dataset_id, str):
            parsed_by_id[dataset_id] = entry

    # A4: partial-result warning
    n_missing = len(datasets) - len(parsed_by_id)
    if n_missing > 0 and completed_process.returncode == 0:
        warnings.warn(
            f"Subprocess emitted results for {len(parsed_by_id)}/{len(datasets)} datasets"
        )

    stderr_text = (completed_process.stderr or "").strip()
    if completed_process.returncode != 0:
        default_error = f"Subprocess exited with code {completed_process.returncode}"
        if stderr_text:
            default_error = f"{default_error}: {stderr_text}"
    else:
        default_error = stderr_text or None
    results: list[FeolsResult] = []

    for dataset in datasets:
        entry = parsed_by_id.get(dataset.dataset_id)
        if entry is None:
            missing_error = default_error or "No result emitted by subprocess backend."
            results.append(
                _result_from_dataset(
                    dataset,
                    spec,
                    backend=backend,
                    outcome=RunOutcome(
                        elapsed=None,
                        success=False,
                        error=missing_error,
                    ),
                )
            )
            continue

        elapsed_raw = entry.get("time")
        try:
            elapsed = None if elapsed_raw is None else float(elapsed_raw)
        except (TypeError, ValueError):
            elapsed = None
        n_obs_raw = entry.get("n_obs")
        try:
            n_obs_override = None if n_obs_raw is None else int(n_obs_raw)
        except (TypeError, ValueError):
            n_obs_override = None

        results.append(
            _result_from_dataset(
                dataset,
                spec,
                backend=backend,
                outcome=RunOutcome(
                    elapsed=elapsed,
                    success=bool(entry.get("success", elapsed is not None)),
                    error=entry.get("error"),
                    n_obs_override=n_obs_override,
                ),
            )
        )

    return results


class SubprocessFeolsBenchmarker:
    """Generic subprocess backend for feols (R/Julia)."""

    def __init__(
        self,
        *,
        name: str,
        command_prefix: Sequence[str],
        script_path: Path,
    ):
        self._name = name
        self._command_prefix = tuple(command_prefix)
        self._script_path = script_path.resolve()

    @property
    def name(self) -> str:
        return self._name

    def run(
        self, datasets: list[BenchmarkDataset], spec: FeolsSpec
    ) -> list[FeolsResult]:
        manifest = [
            {
                "dataset_id": dataset.dataset_id,
                "data_path": str(dataset.data_path.resolve()),
                "dgp": dataset.dgp,
                "n_obs": dataset.n_obs,
                "iter_type": dataset.iter_type,
                "iter_num": dataset.iter_num,
            }
            for dataset in datasets
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "manifest": manifest,
                        "formula": spec.formula,
                        "depvar": spec.depvar,
                        "covariates": spec.covariates,
                        "fe_cols": spec.fe_cols,
                        "vcov": spec.vcov,
                        "vcov_type": _normalize_vcov(spec.vcov),
                    }
                ),
                encoding="utf-8",
            )

            command = [
                *self._command_prefix,
                str(self._script_path),
                str(config_path),
            ]

            try:
                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except Exception as exc:
                return [
                    _result_from_dataset(
                        dataset,
                        spec,
                        backend=self.name,
                        outcome=RunOutcome(
                            elapsed=None,
                            success=False,
                            error=str(exc),
                        ),
                    )
                    for dataset in datasets
                ]

        return _parse_subprocess_output(
            datasets=datasets,
            spec=spec,
            backend=self.name,
            completed_process=proc,
        )


class FixestFeolsBenchmarker:
    def __init__(self, script_path: Path | None = None):
        resolved_script = script_path or Path(__file__).with_name("feols_r.R")
        self._delegate = SubprocessFeolsBenchmarker(
            name="r.fixest",
            command_prefix=["Rscript"],
            script_path=resolved_script,
        )

    @property
    def name(self) -> str:
        return self._delegate.name

    def run(
        self, datasets: list[BenchmarkDataset], spec: FeolsSpec
    ) -> list[FeolsResult]:
        return self._delegate.run(datasets, spec)


class JuliaFeolsBenchmarker:
    def __init__(self, script_path: Path | None = None):
        resolved_script = script_path or Path(__file__).with_name("feols_julia.jl")
        self._delegate = SubprocessFeolsBenchmarker(
            name="julia.FixedEffectModels",
            command_prefix=["julia"],
            script_path=resolved_script,
        )

    @property
    def name(self) -> str:
        return self._delegate.name

    def run(
        self, datasets: list[BenchmarkDataset], spec: FeolsSpec
    ) -> list[FeolsResult]:
        return self._delegate.run(datasets, spec)
