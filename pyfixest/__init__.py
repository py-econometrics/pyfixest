# Import modules
from pyfixest import (
    did,
    errors,
    estimation,
    report,
    utils,
)

# Import frequently used functions and classes
from pyfixest.estimation import (
    bonferroni,
    feols,
    fepois,
    rwolf,
)
from pyfixest.report import (
    coefplot,
    etable,
    iplot,
    summary,
)
from pyfixest.utils import (
    get_data,
    get_ssc,
    ssc,
)

__all__ = [
    "feols",
    "fepois",
    "etable",
    "summary",
    "iplot",
    "coefplot",
    "bonferroni",
    "rwolf",
    "get_data",
    "ssc",
    "get_ssc",
    "did",
    "errors",
    "estimation",
    "report",
    "utils",
]

from importlib.metadata import PackageNotFoundError, version

try:
    # Replace 'foo' with the name of your package
    __version__ = version("pyfixest")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"
