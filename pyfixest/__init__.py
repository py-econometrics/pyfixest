"""Public lazy-loading namespace for pyfixest estimators, reporting, and data helpers."""

import importlib as _importlib
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

if _TYPE_CHECKING:
    from pyfixest import did as did
    from pyfixest import errors as errors
    from pyfixest import estimation as estimation
    from pyfixest import report as report
    from pyfixest import typing as typing
    from pyfixest import utils as utils
    from pyfixest.core.demean import Preconditioner as Preconditioner
    from pyfixest.demeaners import BaseDemeaner as BaseDemeaner
    from pyfixest.demeaners import LsmrDemeaner as LsmrDemeaner
    from pyfixest.demeaners import MapDemeaner as MapDemeaner
    from pyfixest.did import SaturatedEventStudy as SaturatedEventStudy
    from pyfixest.did import did2s as did2s
    from pyfixest.did import event_study as event_study
    from pyfixest.did import lpdid as lpdid
    from pyfixest.did import panelview as panelview
    from pyfixest.estimation import bonferroni as bonferroni
    from pyfixest.estimation import feglm as feglm
    from pyfixest.estimation import feols as feols
    from pyfixest.estimation import fepois as fepois
    from pyfixest.estimation import quantreg as quantreg
    from pyfixest.estimation import rwolf as rwolf
    from pyfixest.estimation import wyoung as wyoung
    from pyfixest.report import coefplot as coefplot
    from pyfixest.report import dtable as dtable
    from pyfixest.report import etable as etable
    from pyfixest.report import iplot as iplot
    from pyfixest.report import qplot as qplot
    from pyfixest.report import summary as summary
    from pyfixest.utils import get_bartik_data as get_bartik_data
    from pyfixest.utils import get_data as get_data
    from pyfixest.utils import get_encouragement_data as get_encouragement_data
    from pyfixest.utils import get_ivf_data as get_ivf_data
    from pyfixest.utils import (
        get_motherhood_event_study_data as get_motherhood_event_study_data,
    )
    from pyfixest.utils import get_ssc as get_ssc
    from pyfixest.utils import get_twin_data as get_twin_data
    from pyfixest.utils import get_worker_panel as get_worker_panel
    from pyfixest.utils import ssc as ssc

# Version handling (keep eager - it's cheap)
try:
    __version__ = _version("pyfixest")
except _PackageNotFoundError:
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
    "dtable",
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
    "panelview",
    "qplot",
    "quantreg",
    "report",
    "rwolf",
    "ssc",
    "summary",
    "typing",
    "utils",
    "wyoung",
]

# Submodules loaded lazily
_submodules = ["did", "errors", "estimation", "report", "typing", "utils"]

# Map function/class names to their module prefix
# For direct module imports: import_module(f"{prefix}.{name}")
_lazy_imports = {
    # estimation - each function in its own module
    "feols": "pyfixest.estimation.api",
    "fepois": "pyfixest.estimation.api",
    "feglm": "pyfixest.estimation.api",
    "quantreg": "pyfixest.estimation.api",
    # demeaner configs
    "BaseDemeaner": "pyfixest.demeaners",
    "MapDemeaner": "pyfixest.demeaners",
    "LsmrDemeaner": "pyfixest.demeaners",
    "Preconditioner": "pyfixest.core.demean",
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
    # utils
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
_direct_module_imports = {"feols", "fepois", "feglm", "quantreg"}


def __getattr__(name: str) -> _Any:
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


def __dir__() -> list[str]:
    """List lazy public exports alongside normal module metadata."""
    return sorted(set(__all__) | set(globals()))
