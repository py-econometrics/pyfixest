from pyfixest.estimation.models.feglm_ import Feglm


class Feprobit(Feglm):
    "Class for the estimation of a fixed-effects probit model."

    def __init__(self, *, method: str = "feglm-probit", **kwargs):
        super().__init__(method=method, **kwargs)
