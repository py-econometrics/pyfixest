from __future__ import annotations

from typing import ClassVar

from pyfixest.estimation.internals.families import PROBIT, GlmFamily
from pyfixest.estimation.models.feglm_ import Feglm


class Feprobit(Feglm):
    "Class for the estimation of a fixed-effects probit model."

    _family: ClassVar[GlmFamily] = PROBIT
