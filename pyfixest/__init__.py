"""
PyFixest: fast high-dimensional fixed-effects regression in Python.

PyFixest fits OLS, IV, Poisson, GLM and quantile regressions with many fixed
effects on pandas or polars DataFrames, from a formula such as
``"Y ~ X1 | f1 + f2"``, and provides standard errors, post-estimation tools and
regression tables.

Example::

    import pyfixest as pf

    data = pf.get_data()
    fit = pf.feols("Y ~ X1 | f1 + f2", data=data)
    fit.summary()

Resources:

- Documentation: https://pyfixest.org
- Documentation index for LLMs: https://pyfixest.org/llms.txt
- Agent skill: ``skills/pyfixest/SKILL.md`` inside this package, a guide for
  coding agents to the current API and its common pitfalls
- Migrating code written for PyFixest 0.60:
  https://pyfixest.org/how-to/migrate-fit-attributes.html
- Source code: https://github.com/py-econometrics/pyfixest
"""

import importlib as _importlib
from importlib.metadata import PackageNotFoundError, version

# Version handling (keep eager - it's cheap)
try:
    __version__ = version("pyfixest")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "BaseDemeaner",
    "LsmrDemeaner",
    "MapDemeaner",
    "Preconditioner",
    "SaturatedEventStudy",
    "bonferroni",
    "coefplot",
    "did",
    "did2s",
    "errors",
    "estimation",
    "etable",
    "event_study",
    "feglm",
    "feols",
    "fepois",
    "get_bartik_data",
    "get_data",
    "get_encouragement_data",
    "get_ivf_data",
    "get_motherhood_event_study_data",
    "get_ssc",
    "get_twin_data",
    "get_worker_panel",
    "iplot",
    "lpdid",
    "optimal_mixture_precision",
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

_submodules = ["did", "errors", "estimation", "report", "utils"]

# Map function/class names to their module prefix
# For direct module imports: import_module(f"{prefix}.{name}")
_lazy_imports = {
    "feols": "pyfixest.estimation.api",
    "fepois": "pyfixest.estimation.api",
    "feglm": "pyfixest.estimation.api",
    "quantreg": "pyfixest.estimation.api",
    "BaseDemeaner": "pyfixest.demeaners",
    "MapDemeaner": "pyfixest.demeaners",
    "LsmrDemeaner": "pyfixest.demeaners",
    "Preconditioner": "pyfixest.core.demean",
    "bonferroni": "pyfixest.estimation",
    "optimal_mixture_precision": "pyfixest.estimation",
    "rwolf": "pyfixest.estimation",
    "wyoung": "pyfixest.estimation",
    "did2s": "pyfixest.did",
    "event_study": "pyfixest.did",
    "lpdid": "pyfixest.did",
    "panelview": "pyfixest.did",
    "SaturatedEventStudy": "pyfixest.did",
    "etable": "pyfixest.report",
    "summary": "pyfixest.report",
    "coefplot": "pyfixest.report",
    "iplot": "pyfixest.report",
    "qplot": "pyfixest.report",
    "get_bartik_data": "pyfixest.utils",
    "get_data": "pyfixest.utils",
    "get_encouragement_data": "pyfixest.utils",
    "get_ivf_data": "pyfixest.utils",
    "get_motherhood_event_study_data": "pyfixest.utils",
    "get_twin_data": "pyfixest.utils",
    "get_worker_panel": "pyfixest.utils",
    "ssc": "pyfixest.utils",
    "get_ssc": "pyfixest.utils",
}

# Functions that have their own dedicated module (can be imported directly)
_direct_module_imports = {
    "feglm",
    "feols",
    "fepois",
    "quantreg",
}


def __getattr__(name: str):
    if name in _submodules:
        return _importlib.import_module(f"pyfixest.{name}")
    if name in _lazy_imports:
        if name in _direct_module_imports:
            module = _importlib.import_module(f"{_lazy_imports[name]}.{name}")
            return getattr(module, name)
        else:
            module = _importlib.import_module(_lazy_imports[name])
            return getattr(module, name)
    raise AttributeError(f"module 'pyfixest' has no attribute {name!r}")


def __dir__():
    return __all__
