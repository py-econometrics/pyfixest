from pyfixest.estimation.internals.families import PROBIT
from pyfixest.estimation.models._model_init import ModelInit
from pyfixest.estimation.models.feglm_ import Feglm


class Feprobit(Feglm):
    "Class for the estimation of a fixed-effects probit model."

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
            family=PROBIT,
        )

        self._method = "feglm-probit"
