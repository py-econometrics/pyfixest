from pyfixest.estimation.models.feglm_ import Feglm


class Felogit(Feglm):
    "Class for the estimation of a fixed-effects logit model."

    def __init__(self, *, method: str = "feglm-logit", **kwargs):
        super().__init__(method=method, **kwargs)
