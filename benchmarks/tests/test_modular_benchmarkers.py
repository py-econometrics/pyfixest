from __future__ import annotations

import sys
from pathlib import Path

import pytest

MODULAR_DIR = Path(__file__).resolve().parents[1] / "modular"
if str(MODULAR_DIR) not in sys.path:
    sys.path.insert(0, str(MODULAR_DIR))

import benchmarker_sets as bs  # noqa: E402
import feols_benchmarkers as fb  # noqa: E402


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


def test_detect_cupy_runtime_availability_without_cupy(monkeypatch):
    """Test that detect_cupy_runtime_availability reports no cupy when import fails."""
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("cupy missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    availability = fb.detect_cupy_runtime_availability()

    assert availability == fb.CupyRuntimeAvailability(
        has_cupy=False,
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
    monkeypatch.setattr(
        bs,
        "detect_cupy_runtime_availability",
        lambda: fb.CupyRuntimeAvailability(has_cupy=False, has_cuda=False),
    )

    bundle = bs.build_standard_feols_benchmarkers()

    assert all("torch" not in name for name in _names(bundle.benchmarkers))
    assert "torch is not installed" in capsys.readouterr().out


def test_build_standard_feols_benchmarkers_can_exclude_base_pyfixest(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        bs,
        "detect_torch_runtime_availability",
        lambda: fb.TorchRuntimeAvailability(
            has_torch=False,
            has_mps=False,
            has_cuda=False,
        ),
    )
    monkeypatch.setattr(
        bs,
        "detect_cupy_runtime_availability",
        lambda: fb.CupyRuntimeAvailability(has_cupy=False, has_cuda=False),
    )

    bundle = bs.build_standard_feols_benchmarkers(
        include_pyfixest=False,
        include_fixest=True,
        include_julia=True,
    )

    assert all(
        name not in _names(bundle.benchmarkers)
        for name in [
            "pyfixest (rust-cg)",
            "pyfixest (rust-map)",
            "pyfixest (scipy-lsmr)",
        ]
    )
    assert "fixest-map" in _names(bundle.benchmarkers)
    assert "FEM.jl (lsmr)" in _names(bundle.benchmarkers)
    assert bundle.figure_backends == ["fixest-map", "FEM.jl (lsmr)"]


def test_build_standard_feols_benchmarkers_includes_explicit_torch_benchmarkers(
    monkeypatch, capsys
):
    """Test that torch-cpu and torch-mps benchmarkers are included when available."""
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
    monkeypatch.setattr(
        bs,
        "detect_cupy_runtime_availability",
        lambda: fb.CupyRuntimeAvailability(has_cupy=False, has_cuda=False),
    )

    bundle = bs.build_standard_feols_benchmarkers(fixef_maxiter=321)

    assert "pyfixest (torch-cpu)" in _names(bundle.benchmarkers)
    assert "pyfixest (torch-mps)" in _names(bundle.benchmarkers)
    assert "pyfixest (torch-cuda)" not in _names(bundle.benchmarkers)
    assert "CUDA unavailable" in capsys.readouterr().out
    assert bundle.figure_backends == [
        name for name in _names(bundle.benchmarkers) if name.startswith("pyfixest")
    ]
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
    monkeypatch.setattr(
        bs,
        "detect_cupy_runtime_availability",
        lambda: fb.CupyRuntimeAvailability(has_cupy=False, has_cuda=False),
    )

    bundle = bs.build_standard_feols_benchmarkers()

    assert "pyfixest (torch-cuda)" in _names(bundle.benchmarkers)
    assert _backends(bundle.benchmarkers).count("torch_cuda") == 1


def test_build_standard_feols_benchmarkers_includes_cupy_when_available(monkeypatch):
    """Test that cupy32 benchmarker is included when cupy CUDA is available."""
    monkeypatch.setattr(
        bs,
        "detect_torch_runtime_availability",
        lambda: fb.TorchRuntimeAvailability(
            has_torch=False,
            has_mps=False,
            has_cuda=False,
        ),
    )
    monkeypatch.setattr(
        bs,
        "detect_cupy_runtime_availability",
        lambda: fb.CupyRuntimeAvailability(has_cupy=True, has_cuda=True),
    )

    bundle = bs.build_standard_feols_benchmarkers()

    assert "pyfixest (cupy32)" in _names(bundle.benchmarkers)
    assert _backends(bundle.benchmarkers).count("cupy32") == 1


def test_build_standard_feols_benchmarkers_keeps_torch_when_base_pyfixest_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        bs,
        "detect_torch_runtime_availability",
        lambda: fb.TorchRuntimeAvailability(
            has_torch=True,
            has_mps=True,
            has_cuda=False,
        ),
    )
    monkeypatch.setattr(
        bs,
        "detect_cupy_runtime_availability",
        lambda: fb.CupyRuntimeAvailability(has_cupy=False, has_cuda=False),
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
    monkeypatch.setattr(
        bs,
        "detect_torch_runtime_availability",
        lambda: fb.TorchRuntimeAvailability(
            has_torch=False,
            has_mps=False,
            has_cuda=False,
        ),
    )
    monkeypatch.setattr(
        bs,
        "detect_cupy_runtime_availability",
        lambda: fb.CupyRuntimeAvailability(has_cupy=False, has_cuda=False),
    )

    with pytest.raises(
        ValueError,
        match="No benchmarkers available after applying include flags and runtime availability checks.",
    ):
        bs.build_standard_feols_benchmarkers(
            include_pyfixest=False,
            include_fixest=False,
            include_julia=False,
            include_torch=False,
            include_cupy=False,
        )
