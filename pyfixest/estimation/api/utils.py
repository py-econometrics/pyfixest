from collections.abc import Mapping
from typing import Any, Optional, Union

import pandas as pd

from pyfixest.estimation.fixest_multi import FixestMulti
from pyfixest.estimation.internals.literals import (
    DemeanerBackendOptions,
    QuantregMethodOptions,
    QuantregMultiOptions,
    SolverOptions,
)
from pyfixest.utils.dev_utils import DataFrameType, _narwhals_to_pandas
from pyfixest.utils.utils import capture_context
from pyfixest.utils.utils import ssc as ssc_func


def _estimation_input_checks(
    fml: str,
    data: DataFrameType,
    vcov: Optional[Union[str, dict[str, str]]],
    vcov_kwargs: Optional[dict[str, Union[str, int]]],
    weights: Union[None, str],
    ssc: dict[str, Union[str, bool]],
    fixef_rm: str,
    collin_tol: float,
    copy_data: bool,
    store_data: bool,
    lean: bool,
    fixef_tol: float,
    fixef_maxiter: int,
    weights_type: str,
    use_compression: bool,
    reps: Optional[int],
    seed: Optional[int],
    split: Optional[str],
    fsplit: Optional[str],
    separation_check: Optional[list[str]] = None,
):
    if not isinstance(fml, str):
        raise TypeError("fml must be a string")
    if not isinstance(data, pd.DataFrame):
        data = _narwhals_to_pandas(data)
    if not isinstance(vcov, (str, dict, type(None))):
        raise TypeError("vcov must be a string, dictionary, or None")
    if not isinstance(fixef_rm, str):
        raise TypeError("fixef_rm must be a string")
    if not isinstance(collin_tol, float):
        raise TypeError("collin_tol must be a float")

    if fixef_rm not in ["none", "singleton"]:
        raise ValueError("fixef_rm must be either 'none' or 'singleton'.")
    if collin_tol <= 0:
        raise ValueError("collin_tol must be greater than zero")
    if collin_tol >= 1:
        raise ValueError("collin_tol must be less than one")

    if not (isinstance(weights, str) or weights is None):
        raise ValueError(
            f"weights must be a string or None but you provided weights = {weights}."
        )
    if weights is not None:
        assert weights in data.columns, "weights must be a column in data"

    bool_args = [copy_data, store_data, lean]
    for arg in bool_args:
        if not isinstance(arg, bool):
            raise TypeError(f"The function argument {arg} must be of type bool.")

    if not isinstance(fixef_tol, float):
        raise TypeError(
            """The function argument `fixef_tol` needs to be of
            type float.
            """
        )
    if fixef_tol <= 0:
        raise ValueError(
            """
            The function argument `fixef_tol` needs to be of
            strictly larger than 0.
            """
        )
    if fixef_tol >= 1:
        raise ValueError(
            """
            The function argument `fixef_tol` needs to be of
            strictly smaller than 1.
            """
        )

    if not isinstance(fixef_maxiter, int):
        raise TypeError(
            """The function argument `fixef_maxiter` needs to be of
            type int.
            """
        )
    if fixef_maxiter <= 0:
        raise ValueError(
            """
            The function argument `fixef_maxiter` needs to be of
            strictly larger than 0.
            """
        )

    if weights_type not in ["aweights", "fweights"]:
        raise ValueError(
            f"""
            The `weights_type` argument must be of type `aweights`
            (for analytical / precision weights) or `fweights`
            (for frequency weights) but it is {weights_type}.
            """
        )

    if not isinstance(use_compression, bool):
        raise TypeError("The function argument `use_compression` must be of type bool.")
    if use_compression and weights is not None:
        raise NotImplementedError(
            "Compressed regression is not supported with weights."
        )

    if reps is not None:
        if not isinstance(reps, int):
            raise TypeError("The function argument `reps` must be of type int.")

        if reps <= 0:
            raise ValueError("The function argument `reps` must be strictly positive.")

    if seed is not None and not isinstance(seed, int):
        raise TypeError("The function argument `seed` must be of type int.")

    if split is not None and not isinstance(split, str):
        raise TypeError("The function argument split needs to be of type str.")

    if fsplit is not None and not isinstance(fsplit, str):
        raise TypeError("The function argument fsplit needs to be of type str.")

    if split is not None and fsplit is not None and split != fsplit:
        raise ValueError(
            f"""
                        Arguments split and fsplit are both specified, but not identical.
                        split is specified as {split}, while fsplit is specified as {fsplit}.
                        """
        )

    if isinstance(split, str) and split not in data.columns:
        raise KeyError(f"Column '{split}' not found in data.")

    if isinstance(fsplit, str) and fsplit not in data.columns:
        raise KeyError(f"Column '{fsplit}' not found in data.")

    if separation_check is not None:
        if not isinstance(separation_check, list):
            raise TypeError(
                "The function argument `separation_check` must be of type list."
            )

        if not all(x in ["fe", "ir"] for x in separation_check):
            raise ValueError(
                "The function argument `separation_check` must be a list of strings containing 'fe' and/or 'ir'."
            )

    if vcov_kwargs is not None:
        # check that dict keys are either "lag", "time_id", or "panel_id"
        if not all(key in ["lag", "time_id", "panel_id"] for key in vcov_kwargs):
            raise ValueError(
                "The function argument `vcov_kwargs` must be a dictionary with keys 'lag', 'time_id', or 'panel_id'."
            )

        # if lag provided, check that it is an int
        if "lag" in vcov_kwargs and not isinstance(vcov_kwargs["lag"], int):
            raise ValueError(
                "The function argument `vcov_kwargs` must be a dictionary with integer values for 'lag' if explicitly provided."
            )

        if "time_id" in vcov_kwargs:
            if not isinstance(vcov_kwargs["time_id"], str):
                raise ValueError(
                    "The function argument `vcov_kwargs` must be a dictionary with string values for 'time_id' if explicitly provided."
                )
            if vcov_kwargs["time_id"] not in data.columns:
                raise ValueError(
                    f"The variable '{vcov_kwargs['time_id']}' is not in the data."
                )

        if "panel_id" in vcov_kwargs:
            if not isinstance(vcov_kwargs["panel_id"], str):
                raise ValueError(
                    "The function argument `vcov_kwargs` must be a dictionary with string values for 'panel_id' if explicitly provided."
                )
            if vcov_kwargs["panel_id"] not in data.columns:
                raise ValueError(
                    f"The variable '{vcov_kwargs['panel_id']}' is not in the data."
                )


def _run_estimation(
    *,
    estimation: str,
    fml: str,
    data: DataFrameType,
    vcov: Optional[Union[str, dict[str, str]]],
    vcov_kwargs: Optional[dict[str, Union[str, int]]],
    weights: Union[None, str],
    ssc: Optional[dict[str, Union[str, bool]]],
    fixef_rm: str,
    collin_tol: float,
    copy_data: bool,
    store_data: bool,
    lean: bool,
    fixef_tol: float,
    fixef_maxiter: int,
    weights_type: str,
    use_compression: bool,
    reps: Optional[int],
    seed: Optional[int],
    split: Optional[str],
    fsplit: Optional[str],
    context: Optional[Union[int, Mapping[str, Any]]],
    separation_check: Optional[list[str]] = None,
    drop_intercept: bool = False,
    solver: SolverOptions = "scipy.linalg.solve",
    demeaner_backend: DemeanerBackendOptions = "numba",
    iwls_tol: float = 1e-08,
    iwls_maxiter: int = 25,
    accelerate: bool = True,
    quantile: Optional[Union[float, list[float]]] = None,
    quantile_tol: float = 1e-06,
    quantile_maxiter: Optional[int] = None,
    quantreg_method: QuantregMethodOptions = "fn",
    quantreg_multi_method: QuantregMultiOptions = "cfm1",
    iv_error_message: Optional[str] = None,
):
    if ssc is None:
        ssc = ssc_func()
    context = {} if context is None else capture_context(context)

    _estimation_input_checks(
        fml=fml,
        data=data,
        vcov=vcov,
        vcov_kwargs=vcov_kwargs,
        weights=weights,
        ssc=ssc,
        fixef_rm=fixef_rm,
        collin_tol=collin_tol,
        copy_data=copy_data,
        store_data=store_data,
        lean=lean,
        fixef_tol=fixef_tol,
        fixef_maxiter=fixef_maxiter,
        weights_type=weights_type,
        use_compression=use_compression,
        reps=reps,
        seed=seed,
        split=split,
        fsplit=fsplit,
        separation_check=separation_check,
    )

    fixest = FixestMulti(
        data=data,
        copy_data=copy_data,
        store_data=store_data,
        lean=lean,
        fixef_tol=fixef_tol,
        fixef_maxiter=fixef_maxiter,
        weights_type=weights_type,
        use_compression=use_compression,
        reps=reps,
        seed=seed,
        split=split,
        fsplit=fsplit,
        context=context,
        quantreg_method=quantreg_method,
        quantreg_multi_method=quantreg_multi_method,
    )

    fixest._prepare_estimation(
        estimation=estimation,
        fml=fml,
        vcov=vcov,
        vcov_kwargs=vcov_kwargs,
        weights=weights,
        ssc=ssc,
        fixef_rm=fixef_rm,
        drop_intercept=drop_intercept,
        quantile=quantile,
        quantile_tol=quantile_tol,
        quantile_maxiter=quantile_maxiter,
    )

    if iv_error_message is not None and fixest._is_iv:
        raise NotImplementedError(iv_error_message)

    fixest._estimate_all_models(
        vcov=vcov,
        solver=solver,
        vcov_kwargs=vcov_kwargs,
        demeaner_backend=demeaner_backend,
        collin_tol=collin_tol,
        iwls_maxiter=iwls_maxiter,
        iwls_tol=iwls_tol,
        separation_check=separation_check,
        accelerate=accelerate,
    )

    if fixest._is_multiple_estimation:
        return fixest
    else:
        return fixest.fetch_model(0, print_fml=False)
