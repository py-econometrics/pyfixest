"""Public difference-in-differences estimators and visualization helpers.

Use ``event_study``, ``did2s``, and ``lpdid`` through this namespace. Consult the
installed DiD tutorial and reference pages under ``pyfixest/docs/pages/`` before
combining an estimator with nonstandard inference or reporting.
"""

from pyfixest.did.estimation import (
    did2s,
    event_study,
    lpdid,
)
from pyfixest.did.saturated_twfe import SaturatedEventStudy
from pyfixest.did.visualize import (
    panelview,
)

__all__ = ["SaturatedEventStudy", "did2s", "event_study", "lpdid", "panelview"]
