"""Public model summaries, tables, and coefficient plots.

Use the explicit reporting functions and result methods here rather than private
result state. The installed reporting tutorial and reference pages under
``pyfixest/docs/pages/`` describe supported single-model and ``FixestMulti``
workflows.
"""

from pyfixest.report.summarize import (
    dtable,
    etable,
    summary,
)
from pyfixest.report.visualize import (
    coefplot,
    iplot,
    qplot,
)

__all__ = [
    "coefplot",
    "dtable",
    "etable",
    "iplot",
    "qplot",
    "summary",
]
