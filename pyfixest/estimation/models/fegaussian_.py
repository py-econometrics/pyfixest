from pyfixest.estimation.internals.families import GAUSSIAN
from pyfixest.estimation.internals.vcov_ import vcov_iid_ols
from pyfixest.estimation.models._model_init import ModelInit
from pyfixest.estimation.models.feglm_ import Feglm


class Fegaussian(Feglm):
    "Class for the estimation of a fixed-effects GLM with normal errors."

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
            family=GAUSSIAN,
        )

        self._method = "feglm-gaussian"

    def _vcov_iid(self):
        # we set gaussian glms to match pf.feols exactly
        return vcov_iid_ols(residuals=self._u_hat, bread=self._bread, N=self._N)
