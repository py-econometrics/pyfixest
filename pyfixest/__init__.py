import importlib as _importlib
from importlib.metadata import PackageNotFoundError, version

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
    "make_table": "pyfixest.report",
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
