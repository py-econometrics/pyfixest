from __future__ import annotations

import ctypes
import gc
import json
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from .interfaces import BenchmarkDataset, FeolsResult, FeolsSpec
except ImportError:
    from interfaces import BenchmarkDataset, FeolsResult, FeolsSpec

_MIN_DGP_WIDTH = 16


def _trim_process_memory(demeaner_backend: str) -> None:
    """Return unused Python and native allocator memory after large benchmark cases."""
    gc.collect()

    if demeaner_backend.startswith("torch"):
        try:
            import torch
        except ImportError:
            pass
        else:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


@dataclass(frozen=True)
class TorchRuntimeAvailability:
    """Runtime availability of optional torch benchmark targets."""

    has_torch: bool
    has_mps: bool
    has_cuda: bool


@dataclass(frozen=True)
class CupyRuntimeAvailability:
    """Runtime availability of optional cupy benchmark targets."""

    has_cupy: bool
    has_cuda: bool


@dataclass(frozen=True)
class JaxRuntimeAvailability:
    """Runtime availability of optional jax benchmark targets."""

    has_jax: bool
    has_gpu: bool


def detect_torch_runtime_availability() -> TorchRuntimeAvailability:
    """Detect whether torch and optional accelerator backends are available."""
    try:
        import torch
    except ImportError:
        return TorchRuntimeAvailability(
            has_torch=False,
            has_mps=False,
            has_cuda=False,
        )

    has_mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    has_cuda = bool(torch.cuda.is_available())
    return TorchRuntimeAvailability(
        has_torch=True,
        has_mps=has_mps,
        has_cuda=has_cuda,
    )


def detect_jax_runtime_availability() -> JaxRuntimeAvailability:
    """Detect whether jax and a GPU runtime are available."""
    try:
        import jax
    except ImportError:
        return JaxRuntimeAvailability(
            has_jax=False,
            has_gpu=False,
        )

    try:
        gpu_platforms = {"gpu", "cuda", "rocm", "metal"}
        has_gpu = any(
            getattr(device, "platform", "").lower() in gpu_platforms
            for device in jax.devices()
        )
    except Exception:
        has_gpu = False

    return JaxRuntimeAvailability(
        has_jax=True,
        has_gpu=has_gpu,
    )


def detect_cupy_runtime_availability() -> CupyRuntimeAvailability:
    """Detect whether cupy and a CUDA runtime are available."""
    try:
        import cupy
    except ImportError:
        return CupyRuntimeAvailability(
            has_cupy=False,
            has_cuda=False,
        )

    try:
        has_cuda = bool(cupy.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        has_cuda = False

    return CupyRuntimeAvailability(
        has_cupy=True,
        has_cuda=has_cuda,
    )


def _fmt_time(t: float) -> str:
    if t < 1:
        return f"{t * 1000:.1f}ms"
    return f"{t:.3f}s"


def _dgp_width(datasets: list[BenchmarkDataset]) -> int:
    return max(
        _MIN_DGP_WIDTH, max((len(d.dgp) for d in datasets), default=_MIN_DGP_WIDTH)
    )


class _TablePrinter:
    """Formats benchmark tables with dynamic DGP column width."""

    def __init__(self, dgp_w: int):
        self._w = dgp_w
        self._hdr = (
            f"{'dgp':<{dgp_w}} {'k':>3} {'n_obs':>12} {'n_fe':>4} "
            f"{'min':>10} {'median':>10} {'max':>10}  status"
        )
        self._sep = "-" * len(self._hdr)

    def print_header(self, name: str) -> None:
        print(f"\n  {name}", flush=True)
        print(f"  {self._sep}", flush=True)
        print(f"  {self._hdr}", flush=True)
        print(f"  {self._sep}", flush=True)

    def _row_prefix(self, r: FeolsResult) -> str:
        return f"{r.dgp:<{self._w}} {r.model_k:>3} {r.n_obs:>12,} {r.n_fe:>4}"

    def print_row(self, results: list[FeolsResult]) -> None:
        columns, status = _time_columns(results)
        print(f"  {self._row_prefix(results[0])} {columns}  {status}", flush=True)


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


def _group_key(r: FeolsResult) -> tuple[str, int, int, int]:
    return (r.dgp, r.model_k, r.n_obs, r.n_fe)


def _result_from_dataset(
    dataset: BenchmarkDataset,
    spec: FeolsSpec,
    *,
    backend: str,
    elapsed: float | None,
    success: bool,
    error: str | None = None,
    substeps: dict[str, float] | None = None,
    n_obs_override: int | None = None,
) -> FeolsResult:
    return FeolsResult(
        source_dataset_id=dataset.dataset_id,
        source_k=dataset.k,
        iter_type=dataset.iter_type,
        iter_num=dataset.iter_num,
        dgp=dataset.dgp,
        model_k=spec.k,
        n_obs=n_obs_override if n_obs_override is not None else dataset.n_obs,
        n_fe=spec.n_fe,
        backend=backend,
        time=elapsed,
        success=success,
        error=error,
        substeps=substeps,
    )


def _safe_cast(val, type_fn):
    if val is None:
        return None
    try:
        return type_fn(val)
    except (TypeError, ValueError):
        return None


def _normalize_vcov(vcov: str | dict[str, str]) -> str:
    """Normalize vcov spec to a simple string for subprocess backends.

    Returns "iid", "hetero", or "cluster:<colname>".
    """
    if isinstance(vcov, dict) and "CRV1" in vcov:
        return f"cluster:{vcov['CRV1']}"
    return vcov


class PyFeolsBenchmarkerFullApi:
    """Benchmark pf.feols() end-to-end using one configured demeaner backend."""

    def __init__(self, name: str, demeaner_backend: str, **feols_kwargs):
        self._name = name
        self._demeaner_backend = demeaner_backend
        self._feols_kwargs = feols_kwargs

    @property
    def name(self) -> str:
        return self._name

    def run(
        self, datasets: list[BenchmarkDataset], spec: FeolsSpec
    ) -> list[FeolsResult]:
        import pyfixest as pf

        results: list[FeolsResult] = []

        all_cols = [spec.depvar, *spec.covariates, *spec.fe_cols]
        if isinstance(spec.vcov, dict) and "CRV1" in spec.vcov:
            cluster_col = spec.vcov["CRV1"]
            if cluster_col not in all_cols:
                all_cols.append(cluster_col)

        tbl = _TablePrinter(_dgp_width(datasets))
        tbl.print_header(self.name)

        group_buf: list[FeolsResult] = []
        prev_key: tuple | None = None

        for dataset in datasets:
            n_obs_for_result = dataset.n_obs
            df = None
            try:
                df = pd.read_parquet(dataset.data_path, columns=all_cols)
                n_obs_for_result = len(df)

                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"\d+ singleton fixed effect\(s\) dropped from the model\.",
                        category=UserWarning,
                    )
                    pf.feols(
                        fml=spec.formula,
                        data=df,
                        vcov=spec.vcov,
                        copy_data=False,
                        store_data=False,
                        demeaner_backend=self._demeaner_backend,
                        **self._feols_kwargs,
                    )
                elapsed = time.perf_counter() - t0

                result = _result_from_dataset(
                    dataset,
                    spec,
                    backend=self.name,
                    elapsed=elapsed,
                    success=True,
                    n_obs_override=n_obs_for_result,
                )
            except Exception as exc:
                result = _result_from_dataset(
                    dataset,
                    spec,
                    backend=self.name,
                    elapsed=None,
                    success=False,
                    error=str(exc),
                    n_obs_override=n_obs_for_result,
                )
            finally:
                del df
                _trim_process_memory(self._demeaner_backend)

            results.append(result)

            if result.iter_type != "burnin":
                key = _group_key(result)
                if prev_key is not None and key != prev_key and group_buf:
                    tbl.print_row(group_buf)
                    group_buf = []
                group_buf.append(result)
                prev_key = key

        if group_buf:
            tbl.print_row(group_buf)

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
                    elapsed=None,
                    success=False,
                    error=missing_error,
                )
            )
            continue

        elapsed = _safe_cast(entry.get("time"), float)
        n_obs_override = _safe_cast(entry.get("n_obs"), int)

        results.append(
            _result_from_dataset(
                dataset,
                spec,
                backend=backend,
                elapsed=elapsed,
                success=bool(entry.get("success", elapsed is not None)),
                error=entry.get("error"),
                n_obs_override=n_obs_override,
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
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout_lines: list[str] = []
                # Stream stderr line-by-line for real-time console output
                # while collecting stdout for JSON parsing.
                assert proc.stderr is not None
                assert proc.stdout is not None
                for line in proc.stderr:
                    sys.stderr.write(line)
                    sys.stderr.flush()
                stdout_lines = proc.stdout.readlines()
                proc.wait()
            except Exception as exc:
                return [
                    _result_from_dataset(
                        dataset,
                        spec,
                        backend=self.name,
                        elapsed=None,
                        success=False,
                        error=str(exc),
                    )
                    for dataset in datasets
                ]

        completed = subprocess.CompletedProcess(
            args=command,
            returncode=proc.returncode,
            stdout="".join(stdout_lines),
            stderr=None,
        )
        return _parse_subprocess_output(
            datasets=datasets,
            spec=spec,
            backend=self.name,
            completed_process=completed,
        )


_SCRIPT_DIR = Path(__file__).parent


class FixestFeolsBenchmarker(SubprocessFeolsBenchmarker):
    def __init__(self, name: str | Path | None = None, script_path: Path | None = None):
        if isinstance(name, Path):
            if script_path is not None:
                raise TypeError(
                    "script_path must not be provided twice for FixestFeolsBenchmarker."
                )
            script_path = name
            name = None
        super().__init__(
            name=name or "r.fixest",
            command_prefix=["Rscript"],
            script_path=(script_path or _SCRIPT_DIR / "feols_r.R"),
        )


class JuliaFeolsBenchmarker(SubprocessFeolsBenchmarker):
    def __init__(self, name: str | Path | None = None, script_path: Path | None = None):
        if isinstance(name, Path):
            if script_path is not None:
                raise TypeError(
                    "script_path must not be provided twice for JuliaFeolsBenchmarker."
                )
            script_path = name
            name = None
        super().__init__(
            name=name or "julia.FixedEffectModels",
            command_prefix=["julia"],
            script_path=(script_path or _SCRIPT_DIR / "feols_julia.jl"),
        )
