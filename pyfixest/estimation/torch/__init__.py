from __future__ import annotations


def _raise_torch_import_error(*args, **kwargs):
    raise ModuleNotFoundError(
        "torch is required to use pyfixest.estimation.torch. "
        "Install the optional torch dependency first."
    )


try:
    from pyfixest.estimation.torch.demean_torch_ import (
        demean_torch,
        demean_torch_cpu,
        demean_torch_cuda,
        demean_torch_cuda32,
        demean_torch_mps,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    demean_torch = _raise_torch_import_error
    demean_torch_cpu = _raise_torch_import_error
    demean_torch_cuda = _raise_torch_import_error
    demean_torch_cuda32 = _raise_torch_import_error
    demean_torch_mps = _raise_torch_import_error

__all__ = [
    "demean_torch",
    "demean_torch_cpu",
    "demean_torch_cuda",
    "demean_torch_cuda32",
    "demean_torch_mps",
]
