import pandas as pd

from pyfixest.core.demean import Preconditioner
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.families import PROBIT
from pyfixest.estimation.internals.model_state import GlmEstimationOptions
from pyfixest.estimation.models.feglm_ import Feglm


class Feprobit(Feglm):
    "Class for the estimation of a fixed-effects probit model."

    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        *,
        options: GlmEstimationOptions,
        lookup_demeaned_data: dict[frozenset[int], DemeanedData],
        lookup_preconditioner: dict[frozenset[int], Preconditioner] | None = None,
        sample_split_var: str | None = None,
        sample_split_value: str | int | None = None,
    ):
        super().__init__(
            FixestFormula=FixestFormula,
            data=data,
            options=options,
            lookup_demeaned_data=lookup_demeaned_data,
            lookup_preconditioner=lookup_preconditioner,
            sample_split_var=sample_split_var,
            sample_split_value=sample_split_value,
            family=PROBIT,
        )

        self._method = "feglm-probit"
