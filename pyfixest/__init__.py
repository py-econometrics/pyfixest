# Import modules
from pyfixest import (
    did,
    errors,
    estimation,
    report,
    utils,
)
from pyfixest.did import did2s, event_study, lpdid, panelview

# Import frequently used functions and classes
from pyfixest.estimation import (
    bonferroni,
    feglm,
    feols,
    fepois,
    rwolf,
    wyoung,
)
from pyfixest.report import coefplot, dtable, etable, iplot, make_table, summary
from pyfixest.utils import (
    get_data,
    get_ssc,
    ssc,
)

__all__ = [
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
    "report",
    "rwolf",
    "ssc",
    "summary",
    "utils",
    "wyoung",
]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyfixest")
except PackageNotFoundError:
    __version__ = "unknown"
