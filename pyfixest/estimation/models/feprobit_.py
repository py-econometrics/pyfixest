from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.families import FAMILY_REGISTRY
from pyfixest.estimation.models.feglm_ import Feglm


class Feprobit(Feglm):
    "Class for the estimation of a fixed-effects probit model."

    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        ssc_dict: dict[str, str | bool],
        drop_singletons: bool,
        drop_intercept: bool,
        weights: str | None,
        weights_type: str | None,
        collin_tol: float,
        lookup_demeaned_data: dict[frozenset[int], pd.DataFrame],
        tol: float,
        maxiter: int,
        solver: Literal[
            "np.linalg.lstsq",
            "np.linalg.solve",
            "scipy.linalg.solve",
            "scipy.sparse.linalg.lsqr",
        ],
        demeaner: AnyDemeaner | None = None,
        lookup_preconditioner: dict[frozenset[int], Preconditioner] | None = None,
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        sample_split_var: str | None = None,
        sample_split_value: str | int | None = None,
        separation_check: list[str] | None = None,
        context: int | Mapping[str, Any] = 0,
        accelerate: bool = True,
    ):
        super().__init__(
            FixestFormula=FixestFormula,
            data=data,
            ssc_dict=ssc_dict,
            drop_singletons=drop_singletons,
            drop_intercept=drop_intercept,
            weights=weights,
            weights_type=weights_type,
            collin_tol=collin_tol,
            lookup_demeaned_data=lookup_demeaned_data,
            tol=tol,
            maxiter=maxiter,
            solver=solver,
            demeaner=demeaner,
            lookup_preconditioner=lookup_preconditioner,
            store_data=store_data,
            copy_data=copy_data,
            lean=lean,
            sample_split_var=sample_split_var,
            sample_split_value=sample_split_value,
            separation_check=separation_check,
            context=context,
            accelerate=accelerate,
            family=FAMILY_REGISTRY["probit"],
        )

        self._method = "feglm-probit"
