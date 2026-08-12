import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyfixest.report.summarize import etable, summary
    from pyfixest.report.visualize import coefplot, iplot, qplot

__all__ = [
    "coefplot",
    "etable",
    "iplot",
    "qplot",
    "summary",
]

_REPORT_MODULES = {
    "etable": "pyfixest.report.summarize",
    "summary": "pyfixest.report.summarize",
    "coefplot": "pyfixest.report.visualize",
    "iplot": "pyfixest.report.visualize",
    "qplot": "pyfixest.report.visualize",
}


def __getattr__(name: str):
    if name in _REPORT_MODULES:
        module = importlib.import_module(_REPORT_MODULES[name])
        return getattr(module, name)
    raise AttributeError(f"module 'pyfixest.report' has no attribute {name!r}")


def __dir__():
    return __all__
