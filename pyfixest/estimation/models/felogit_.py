from __future__ import annotations

from typing import ClassVar

from pyfixest.estimation.internals.families import LOGIT, GlmFamily
from pyfixest.estimation.models.feglm_ import Feglm


class Felogit(Feglm):
    "Class for the estimation of a fixed-effects logit model."

    _family: ClassVar[GlmFamily] = LOGIT
