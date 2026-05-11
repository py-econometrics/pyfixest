from __future__ import annotations

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - environment dependent
    torch = None
    HAS_TORCH = False

HAS_MPS = (
    HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
)
HAS_CUDA = HAS_TORCH and torch.cuda.is_available()


def torch_param(*values, id: str, require: str | None = None):
    """Create a pytest parameter with torch/device-aware skip marks."""
    marks: list[object] = []

    if len(values) == 1 and isinstance(values[0], tuple):
        values = tuple(values[0])

    if not HAS_TORCH:
        marks.append(pytest.mark.skip(reason="torch not available"))
    elif require == "mps" and not HAS_MPS:
        marks.append(pytest.mark.skip(reason="MPS not available"))
    elif require == "cuda" and not HAS_CUDA:
        marks.append(pytest.mark.skip(reason="CUDA not available"))

    return pytest.param(*values, id=id, marks=marks)
