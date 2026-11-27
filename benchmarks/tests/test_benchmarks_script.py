from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import pyfixest as pf

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks.py"
_spec = importlib.util.spec_from_file_location("benchmarks_script", SCRIPT)
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)


def _scheduled_optional_backends(monkeypatch):
    monkeypatch.setattr(bm, "HAS_TORCH", True)
    monkeypatch.setattr(bm, "HAS_MPS", True)
    monkeypatch.setattr(bm, "HAS_CUDA", True)
    estimators = []
    bm._append_optional_backends(estimators, "pyfixest.feols", None, "feols")
    return [backend for _, backend, *_ in estimators]


def test_every_scheduled_optional_backend_has_a_demeaner(monkeypatch):
    """Test that each torch backend the script schedules maps to a typed demeaner."""
    backends = _scheduled_optional_backends(monkeypatch)

    assert backends == ["torch_cpu", "torch_mps", "torch_cuda", "torch_cuda32"]
    assert set(backends) <= bm._PYFIXEST_BACKENDS
    for backend in backends:
        assert isinstance(bm._make_demeaner(backend), pf.LsmrDemeaner)


@pytest.mark.parametrize(
    ("backend", "device", "precision"),
    [
        ("torch_cpu", "cpu", "float64"),
        ("torch_mps", "mps", "float32"),
        ("torch_cuda", "cuda", "float64"),
        ("torch_cuda32", "cuda", "float32"),
    ],
)
def test_torch_backend_demeaner_configuration(backend, device, precision):
    """Test that torch backend names keep their legacy device and precision."""
    demeaner = bm._make_demeaner(backend)

    assert demeaner.backend == "torch"
    assert demeaner.device == device
    assert demeaner.precision == precision


def test_make_demeaner_rejects_unmapped_backend():
    """Test that an unmapped backend name raises instead of picking a default."""
    with pytest.raises(ValueError, match="no demeaner for backend 'cupy'"):
        bm._make_demeaner("cupy")
