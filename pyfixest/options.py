from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Union
from contextlib import contextmanager

import pandas as pd

from pyfixest.utils.utils import ssc as ssc_func

__all__ = ["get_option", "option_context", "options", "set_option"]


@dataclass
class _Options:
    data: Optional[pd.DataFrame] = None
    vcov: Optional[Union[str, Mapping[str, str]]] = None
    weights: Optional[str] = None
    ssc: Optional[dict[str, Union[str, bool]]] = field(default_factory=ssc_func)
    fixef_rm: str = "none"
    fixef_tol: float = 1e-8
    collin_tol: float = 1e-10
    drop_intercept: bool = False
    i_ref1: Optional[str] = None
    copy_data: bool = True
    store_data: bool = True
    lean: bool = False
    weights_type: str = "aweights"
    solver: str = "scipy.linalg.solve"
    demeaner_backend: str = "numba"
    use_compression: bool = False
    reps: int = 100
    context: Optional[Union[int, Mapping[str, Any]]] = None
    seed: Optional[int] = None
    split: Optional[str] = None
    fsplit: Optional[str] = None

    separation_check: list[str] = field(default_factory=lambda: ["fe"])
    iwls_tol: Optional[float] = 1e-6
    iwls_maxiter: Optional[int] = 25

    # helpers ------------
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise KeyError(f"Unknown option '{k}'")
            setattr(self, k, v)

    def to_dict(self):
        return asdict(self)


options = _Options()


def set_option(**kwargs):
    """Globally set default options (except `fml`)."""
    options.update(**kwargs)


def get_option(name: str):
    return getattr(options, name)


@contextmanager
def option_context(**kwargs):
    "Temporarily override options inside a `with` block."
    old = options.to_dict()
    try:
        options.update(**kwargs)
        yield
    finally:
        options.__dict__.update(old)
