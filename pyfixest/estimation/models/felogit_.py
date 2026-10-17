from __future__ import annotations

from typing import ClassVar

from pyfixest.estimation.internals.families import LOGIT, GlmFamily
from pyfixest.estimation.models.feglm_ import Feglm


class Felogit(Feglm):
    """Fitted logit GLM result returned by `pf.feglm()`.

    Access as ``fit = pf.feglm(formula, data, family="logit")``.
    Public operations and state are inherited from `Feglm`.

    See Also
    --------
    [feglm()](/reference/estimation.api.feglm.feglm.qmd),
    [Feglm](/reference/estimation.models.feglm_.Feglm.qmd),
    [model support](/reference-guide/model-support.qmd).
    """

    _family: ClassVar[GlmFamily] = LOGIT
