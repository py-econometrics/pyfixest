# pyfixest/_options.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Mapping, Optional, Union

__all__ = ["options", "set_option", "get_option", "option_context"]

@dataclass
class _Options:

    data: Optional[pd.DataFrame] = None
    vcov: Optional[Union[str, Mapping[str, str]]] = None
    weights: Optional[str] = None
    ssc: Optional[dict[str, Union[str, bool]]] = None
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

    # helpers ------------
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise KeyError(f"Unknown option '{k}'")
            setattr(self, k, v)

    def to_dict(self):
        return asdict(self)


# a module-level singleton
options = _Options()


def set_option(**kwargs):
    """Globally set default options (except `fml`)."""
    options.update(**kwargs)


def get_option(name: str):
    return getattr(options, name)


from contextlib import contextmanager

@contextmanager
def option_context(**kwargs):
    """
    Temporarily override options inside a `with` block.
    Usage:
        with option_context(weights="my_w", reps=500):
            feols("y~x", df)   # uses the overrides
    """
    old = options.to_dict()
    try:
        options.update(**kwargs)
        yield
    finally:
        options.__dict__.update(old)
