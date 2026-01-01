import importlib as _importlib
from importlib.metadata import PackageNotFoundError, version

# Version handling (keep eager - it's cheap)
try:
    __version__ = version("pyfixest")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "SaturatedEventStudy",
    "bonferroni",
    "coefplot",
    "did",
    "did2s",
    "dtable",
    "errors",
    "estimation",
    "etable",
    "event_study",
    "feglm",
    "feols",
    "fepois",
    "get_data",
    "get_ssc",
    "iplot",
    "lpdid",
    "make_table",
    "panelview",
    "qplot",
    "quantreg",
    "report",
    "rwolf",
    "ssc",
    "summary",
    "utils",
    "wyoung",
]

# Submodules loaded lazily
_submodules = ["did", "errors", "estimation", "report", "utils"]

# Map function/class names to their module prefix
# For direct module imports: import_module(f"{prefix}.{name}")
_lazy_imports = {
    # estimation - each function in its own module
    "feols": "pyfixest.estimation.api",
    "fepois": "pyfixest.estimation.api",
    "feglm": "pyfixest.estimation.api",
    "quantreg": "pyfixest.estimation.api",
    # estimation - other functions (still use parent module + getattr)
    "bonferroni": "pyfixest.estimation",
    "rwolf": "pyfixest.estimation",
    "wyoung": "pyfixest.estimation",
    # did
    "did2s": "pyfixest.did",
    "event_study": "pyfixest.did",
    "lpdid": "pyfixest.did",
    "panelview": "pyfixest.did",
    "SaturatedEventStudy": "pyfixest.did",
    # report
    "etable": "pyfixest.report",
    "dtable": "pyfixest.report",
    "summary": "pyfixest.report",
    "coefplot": "pyfixest.report",
    "iplot": "pyfixest.report",
    "qplot": "pyfixest.report",
    "make_table": "pyfixest.report",
    # utils
    "get_data": "pyfixest.utils",
    "ssc": "pyfixest.utils",
    "get_ssc": "pyfixest.utils",
}

# Functions that have their own dedicated module (can be imported directly)
_direct_module_imports = {"feols", "fepois", "feglm", "quantreg"}


def __getattr__(name: str):
    if name in _submodules:
        return _importlib.import_module(f"pyfixest.{name}")
    if name in _lazy_imports:
        if name in _direct_module_imports:
            # Direct module import: import pyfixest.estimation.api.feols
            module = _importlib.import_module(f"{_lazy_imports[name]}.{name}")
            return getattr(module, name)
        else:
            # Fallback: import parent module and get attribute
            module = _importlib.import_module(_lazy_imports[name])
            return getattr(module, name)
    raise AttributeError(f"module 'pyfixest' has no attribute {name!r}")


def __dir__():
    return __all__
