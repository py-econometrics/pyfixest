from pyfixest.estimation.internals.vcov_ import vcov_iid_ols
from pyfixest.estimation.models.feglm_ import Feglm


class Fegaussian(Feglm):
    "Class for the estimation of a fixed-effects GLM with normal errors."

    def __init__(self, *, method: str = "feglm-gaussian", **kwargs):
        super().__init__(method=method, **kwargs)

    def _vcov_iid(self):
        return vcov_iid_ols(residuals=self._u_hat, bread=self._bread, N=self._N)
