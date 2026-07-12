"""Fit one or more linear quantile regressions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from pyfixest.estimation.api.utils import _estimation_input_checks
from pyfixest.estimation.config import EstimationConfig
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.internals.literals import (
    QuantregMethodOptions,
    QuantregMultiOptions,
    SolverOptions,
)
from pyfixest.estimation.plan_ import parse_formula
from pyfixest.estimation.quantreg.quantreg_ import Quantreg
from pyfixest.estimation.runner import run_estimation
from pyfixest.typing import DataFrameType, QuantregVcovType, SscConfig, WeightsType
from pyfixest.utils.utils import capture_context
from pyfixest.utils.utils import ssc as ssc_func


def _quantreg_input_checks(
    quantile: float | list[float], tol: float, maxiter: int | None
) -> None:
    "Run custom input checks for quantreg."
    if isinstance(quantile, list):
        if not all(isinstance(q, float) for q in quantile):
            raise ValueError("quantile must be a list of floats")

        if not all(0.0 < q < 1.0 for q in quantile):
            raise ValueError("quantile must be between 0 and 1")
    else:
        # single quantile provided
        if not isinstance(quantile, float):
            raise TypeError("quantile must be a float")
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be between 0 and 1")

    # tol must always be in (0, 1)
    if not 0.0 < tol < 1.0:
        raise ValueError("tol must be in (0, 1)")

    if maxiter is not None and maxiter <= 0:
        raise ValueError("maxiter must be greater than 0")


def quantreg(
    fml: str,
    data: DataFrameType,  # type: ignore
    vcov: QuantregVcovType | dict[str, str] | None = "nid",
    quantile: float | list[float] = 0.5,
    method: QuantregMethodOptions = "fn",
    multi_method: QuantregMultiOptions = "cfm1",
    tol: float = 1e-06,
    maxiter: int | None = None,
    ssc: SscConfig | None = None,
    collin_tol: float = 1e-09,
    separation_check: list[str] | None = None,
    drop_intercept: bool = False,
    copy_data: bool = True,
    store_data: bool = True,
    lean: bool = False,
    context: int | Mapping[str, Any] | None = None,
    split: str | None = None,
    fsplit: str | None = None,
    seed: int | None = None,
) -> Quantreg | FixestMulti:
    """
    Fit a quantile regression model using the interior point algorithm from Portnoy and Koenker (1997).
    Note that the interior point algorithm assumes independent observations.

    Parameters
    ----------
    fml : str
        A two-sided formula string using fixest formula syntax.
        In contrast to `feols()` and `feglm()`, no fixed effects formula syntax is supported.

    data : DataFrameType
        A pandas or polars dataframe containing the variables in the formula.

    vcov : QuantregVcovType or dict[str, str], optional
        Variance-covariance estimator. Defaults to `"nid"`. String options are
        `"iid"`, `"nid"`, `"hetero"`, `"HC1"`, `"HC2"`, and `"HC3"`;
        HC1-HC3 are aliases for `"hetero"`. One-way clustering uses a dictionary
        such as `{"CRV1": "cluster"}`. HAC estimators are not supported.

    quantile : float or list[float], optional
        Quantile or quantiles to estimate, strictly between zero and one. A list
        returns a `FixestMulti` containing one `Quantreg` result per quantile.
        Defaults to `0.5`.

    method : QuantregMethodOptions, optional
        Fitting algorithm. `"fn"` implements the Frisch-Newton interior-point algorithm
        described in Portnoy and Koenker (1997).
        `"pfn"` implements a variant with preprocessing steps from the same paper.
        The preprocessing can accelerate large samples and uses `seed` for its
        random number generator.

    multi_method : QuantregMultiOptions, optional
        Algorithm for a list of quantiles. `"cfm1"` (default) implements
        algorithm 2 and `"cfm2"` implements algorithm 3 from Chernozhukov,
        Fernández-Val, and Melly (2019).

    tol : float, optional
        The tolerance for the algorithm. Defaults to 1e-06. As in R's quantreg package, the
        algorithm stops when the relative change in the duality gap is less than tol.

    maxiter : int, optional
        The maximum number of iterations. If None, maxiter = the number of observations in the model
        (as in R's quantreg package via nit(3) = n).

    ssc : SscConfig, optional
        Small-sample correction created by `ssc()`. `None` uses
        `ssc(k_adj=True, k_fixef="nonnested", G_adj=True, G_df="min")`. To
        match software that applies no small-sample correction, use
        `ssc(k_adj=False, G_adj=False)`.

    collin_tol : float, optional
        Tolerance for the collinearity check. Defaults to `1e-9`.

    separation_check : list[str], optional
        Methods to identify and drop separated observations. Not used in quantile regression.

    drop_intercept : bool, optional
        Whether to drop the intercept from the model, by default False.

    copy_data : bool, optional
        Whether to copy the data before estimation, by default True.
        If set to False, the data is not copied, which can save memory but
        may lead to unintended changes in the input data outside of `quantreg`.

    store_data : bool, optional
        Whether to store the data in the model object, by default True.
        If set to False, the data is not stored in the model object, which can
        improve performance and save memory. However, it will no longer be possible
        to access the data via the `data` attribute of the model object.

    lean : bool, optional
        False by default. If True, then all large objects are removed from the
        returned result: this will save memory but will block the possibility
        to use many methods. It is recommended to use the argument vcov
        to obtain the appropriate standard-errors at estimation time,
        since obtaining different SEs won't be possible afterwards.

    context : int or Mapping[str, Any], optional
        A dictionary containing additional context variables to be used by
        formulaic during the creation of the model matrix. This can include
        custom factorization functions, transformations, or any other
        variables that need to be available in the formula environment.

    split : str, optional
        A character string, i.e. 'split = var'. If provided, the sample is split according to the
        variable and one estimation is performed for each value of that variable. If you also want
        to include the estimation for the full sample, use the argument fsplit instead.

    fsplit : str, optional
        This argument is the same as split but also includes the full sample as the first estimation.

    seed: int, optional
        A random seed for reproducibility. If None, no seed is set.
        Only relevant for the "pfn" method.
        The "fn" method is deterministic and does not require a seed.

    Returns
    -------
    Quantreg or FixestMulti
        A [Quantreg](/reference/estimation.quantreg.quantreg_.Quantreg.qmd), or
        [FixestMulti](/reference/estimation.FixestMulti_.FixestMulti.qmd) when
        quantiles, formula syntax, or split options produce multiple models.

    Examples
    --------
    The following example regresses `Y` on `X1` and `X2` at the median (0.5 quantile):

    ```{python}
    import pyfixest as pf

    data = pf.get_data()

    fit = pf.quantreg("Y ~ X1 + X2", data, quantile=0.5)
    fit.summary()
    ```

    To fit multiple quantiles in one call:

    ```{python}
    fits = pf.quantreg("Y ~ X1 + X2", data, quantile=[0.1, 0.5, 0.9])
    pf.qplot(fits)
    ```

    Arguments such as `split`, `fsplit`, `context`, `lean`, and `copy_data`
    behave as in `feols()`, but quantile regression does not support fixed-effects
    formula syntax. For details around inference, fast fitting, and visualization
    of the full quantile regression process, see the
    [quantile regression tutorial](/tutorials/quantile-regression.html).
    """
    # WLS currently not supported for quantile regression
    weights = None
    weights_type: WeightsType = "aweights"
    solver: SolverOptions = "np.linalg.solve"

    if ssc is None:
        ssc = ssc_func()

    context = {} if context is None else capture_context(context)

    if isinstance(vcov, str) and vcov in ["HC1", "HC2", "HC3"]:
        vcov = "hetero"

    _quantreg_input_checks(quantile, tol, maxiter)

    _estimation_input_checks(
        fml=fml,
        data=data,
        vcov=vcov,
        vcov_kwargs=None,
        weights=weights,
        ssc=ssc,
        fixef_rm="none",  # arbitrary, not supported
        collin_tol=collin_tol,
        copy_data=copy_data,
        store_data=store_data,
        lean=lean,
        weights_type=weights_type,
        reps=None,
        seed=None,
        split=split,
        fsplit=fsplit,
        separation_check=separation_check,
    )

    estimation = "quantreg" if not isinstance(quantile, list) else "quantreg_multi"
    config = EstimationConfig(
        method=estimation,
        data=data,
        fml=fml,
        copy_data=copy_data,
        store_data=store_data,
        lean=lean,
        drop_intercept=drop_intercept,
        vcov=vcov,
        vcov_kwargs=None,
        ssc_dict=ssc,
        solver=solver,
        collin_tol=collin_tol,
        context=context,
        weights=weights,
        weights_type=weights_type,
        split=split,
        fsplit=fsplit,
        seed=seed,
        quantile=quantile,
        quantreg_method=method,
        quantile_tol=tol,
        quantile_maxiter=maxiter,
        quantreg_multi_method=multi_method,
    )

    parsed = parse_formula(config)
    if parsed.is_iv:
        raise NotImplementedError(
            "IV Estimation is not supported for Quantile Regression"
        )

    return cast(Quantreg | FixestMulti, run_estimation(config, parsed))
