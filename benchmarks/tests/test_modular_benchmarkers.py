from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

MODULAR_DIR = Path(__file__).resolve().parents[1] / "modular"
if str(MODULAR_DIR) not in sys.path:
    sys.path.insert(0, str(MODULAR_DIR))

import benchmarker_sets as bs  # noqa: E402
import feols_benchmarkers as fb  # noqa: E402
from interfaces import BenchmarkDataset, FeolsSpec  # noqa: E402


def _backends(benchmarkers):
    return [
        benchmarker._demeaner_backend
        for benchmarker in benchmarkers
        if hasattr(benchmarker, "_demeaner_backend")
    ]


def _names(benchmarkers):
    return [benchmarker.name for benchmarker in benchmarkers]


def test_detect_torch_runtime_availability_without_torch(monkeypatch):
    """Test that detect_torch_runtime_availability reports no torch when import fails."""
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    availability = fb.detect_torch_runtime_availability()

    assert availability == fb.TorchRuntimeAvailability(
        has_torch=False,
        has_mps=False,
        has_cuda=False,
    )


def test_build_standard_feols_benchmarkers_skips_torch_when_unavailable(
    monkeypatch, capsys
):
    """Test that torch benchmarkers are excluded when torch is unavailable."""
    availability = fb.TorchRuntimeAvailability(
        has_torch=False,
        has_mps=False,
        has_cuda=False,
    )
    monkeypatch.setattr(
        bs,
        "detect_torch_runtime_availability",
        lambda: availability,
    )

    bundle = bs.build_standard_feols_benchmarkers()

    assert all("torch" not in name for name in _names(bundle.benchmarkers))
    assert "torch is not installed" in capsys.readouterr().out


def test_build_standard_feols_benchmarkers_can_exclude_base_pyfixest(
    monkeypatch,
) -> None:
    """Test that fixest and Julia remain when base pyfixest backends are disabled."""
    monkeypatch.setattr(
        bs,
        "detect_torch_runtime_availability",
        lambda: fb.TorchRuntimeAvailability(
            has_torch=False,
            has_mps=False,
            has_cuda=False,
        ),
    )

    bundle = bs.build_standard_feols_benchmarkers(
        include_pyfixest=False,
        include_fixest=True,
        include_julia=True,
    )

    assert all(
        name not in _names(bundle.benchmarkers)
        for name in [
            "pyfixest (within)",
            "pyfixest (rust-map)",
        ]
    )
    assert "fixest-map" in _names(bundle.benchmarkers)
    assert "FEM.jl (lsmr)" in _names(bundle.benchmarkers)
    assert bundle.figure_backends == ["fixest-map", "FEM.jl (lsmr)"]


def test_build_standard_feols_benchmarkers_includes_explicit_torch_benchmarkers(
    monkeypatch, capsys
):
    """Test that torch-cpu benchmarker is included when torch is available."""
    availability = fb.TorchRuntimeAvailability(
        has_torch=True,
        has_mps=True,
        has_cuda=False,
    )
    monkeypatch.setattr(
        bs,
        "detect_torch_runtime_availability",
        lambda: availability,
    )

    bundle = bs.build_standard_feols_benchmarkers(fixef_maxiter=321)

    assert "pyfixest (torch-cpu)" in _names(bundle.benchmarkers)
    assert "pyfixest (torch-mps)" in _names(bundle.benchmarkers)
    assert "pyfixest (torch-cuda)" not in _names(bundle.benchmarkers)
    assert "CUDA unavailable" in capsys.readouterr().out
    assert bundle.figure_backends == _names(bundle.benchmarkers)
    assert all(
        benchmarker._feols_kwargs.get("fixef_maxiter") == 321
        for benchmarker in bundle.benchmarkers
        if hasattr(benchmarker, "_feols_kwargs")
    )


def test_build_standard_feols_benchmarkers_includes_cuda_when_available(monkeypatch):
    """Test that torch-cuda benchmarker is included when CUDA is available."""
    availability = fb.TorchRuntimeAvailability(
        has_torch=True,
        has_mps=False,
        has_cuda=True,
    )
    monkeypatch.setattr(
        bs,
        "detect_torch_runtime_availability",
        lambda: availability,
    )

    bundle = bs.build_standard_feols_benchmarkers()

    assert "pyfixest (torch-cuda)" in _names(bundle.benchmarkers)
    assert _backends(bundle.benchmarkers).count("torch_cuda") == 1
    assert "fixest-map" in bundle.figure_backends
    assert "FEM.jl (lsmr)" in bundle.figure_backends


def test_build_standard_feols_benchmarkers_keeps_torch_when_base_pyfixest_disabled(
    monkeypatch,
) -> None:
    """Test that optional torch benchmarkers still work without base pyfixest ones."""
    monkeypatch.setattr(
        bs,
        "detect_torch_runtime_availability",
        lambda: fb.TorchRuntimeAvailability(
            has_torch=True,
            has_mps=True,
            has_cuda=False,
        ),
    )

    bundle = bs.build_standard_feols_benchmarkers(
        include_pyfixest=False,
        include_fixest=False,
        include_julia=False,
        include_torch=True,
    )

    assert _names(bundle.benchmarkers) == [
        "pyfixest (torch-cpu)",
        "pyfixest (torch-mps)",
    ]
    assert bundle.figure_backends == _names(bundle.benchmarkers)


def test_build_standard_feols_benchmarkers_raises_when_none_available(
    monkeypatch,
) -> None:
    """Test that an empty benchmarker selection raises a clear error."""
    monkeypatch.setattr(
        bs,
        "detect_torch_runtime_availability",
        lambda: fb.TorchRuntimeAvailability(
            has_torch=False,
            has_mps=False,
            has_cuda=False,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"No benchmarkers available after applying include flags and runtime "
            r"availability checks\."
        ),
    ):
        bs.build_standard_feols_benchmarkers(
            include_pyfixest=False,
            include_fixest=False,
            include_julia=False,
            include_torch=False,
        )


def test_parse_subprocess_output_skips_non_object_json_lines() -> None:
    """Test that non-object JSON lines are skipped instead of crashing."""
    datasets = [
        BenchmarkDataset(
            dataset_id="d1",
            data_path=Path("d1.parquet"),
            dgp="base",
            k=1,
            n_obs=100,
            iter_type="timed",
            iter_num=1,
        ),
        BenchmarkDataset(
            dataset_id="d2",
            data_path=Path("d2.parquet"),
            dgp="base",
            k=1,
            n_obs=100,
            iter_type="timed",
            iter_num=2,
        ),
    ]
    spec = FeolsSpec(depvar="Y", covariates=["X1"], fe_cols=[], vcov="iid")
    completed = subprocess.CompletedProcess(
        args=["Rscript", "feols_r.R"],
        returncode=0,
        stdout='42\n[1, 2]\nnull\n{"dataset_id": "d1", "time": 0.25, "success": true}\n',
        stderr="",
    )

    with pytest.warns(
        UserWarning,
        match=r"emitted no results for 1 datasets, including \['d2'\]",
    ):
        results = fb._parse_subprocess_output(
            datasets=datasets,
            spec=spec,
            backend="r.fixest",
            completed_process=completed,
        )

    assert len(results) == 2
    assert results[0].source_dataset_id == "d1"
    assert results[0].success is True
    assert results[0].time == pytest.approx(0.25)
    assert results[1].source_dataset_id == "d2"
    assert results[1].success is False
    assert results[1].time is None
    assert results[1].error is not None
    assert "d2" in results[1].error
