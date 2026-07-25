from pyfixest.estimation.internals.families import LOGIT
from pyfixest.estimation.models._model_init import ModelInit
from pyfixest.estimation.models.feglm_ import Feglm


class Felogit(Feglm):
    "Class for the estimation of a fixed-effects logit model."

    def __init__(
        self,
        init: ModelInit,
        *,
        tol: float,
        maxiter: int,
        separation_check: list[str] | None = None,
        accelerate: bool = True,
    ):
        super().__init__(
            init,
            tol=tol,
            maxiter=maxiter,
            separation_check=separation_check,
            accelerate=accelerate,
            family=LOGIT,
        )

        self._method = "feglm-logit"
