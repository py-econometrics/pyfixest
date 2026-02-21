from collections.abc import Mapping
from typing import Any, Optional, Union

from pyfixest.estimation.api.utils import _estimation_input_checks
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.internals.literals import (
    QuantregMethodOptions,
    QuantregMultiOptions,
    SolverOptions,
    VcovTypeOptions,
)
from pyfixest.utils.dev_utils import DataFrameType
from pyfixest.utils.utils import capture_context
from pyfixest.utils.utils import ssc as ssc_func


def _quantreg_input_checks(quantile: float, tol: float, maxiter: Optional[int]):
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
    vcov: Optional[Union[VcovTypeOptions, dict[str, str]]] = "nid",
    quantile: float = 0.5,
    method: QuantregMethodOptions = "fn",
    multi_method: QuantregMultiOptions = "cfm1",
    tol: float = 1e-06,
    maxiter: Optional[int] = None,
    ssc: Optional[dict[str, Union[str, bool]]] = None,
    collin_tol: float = 1e-09,
    separation_check: Optional[list[str]] = None,
    drop_intercept: bool = False,
    copy_data: bool = True,
    store_data: bool = True,
    lean: bool = False,
    context: Optional[Union[int, Mapping[str, Any]]] = None,
    split: Optional[str] = None,
    fsplit: Optional[str] = None,
    seed: Optional[int] = None,
):
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

    quantile : float
        The quantile to estimate. Must be between 0 and 1.

    method : QuantregMethodOptions, optional
        The method to use for the quantile regression. Currently, only "fn" is supported.
        In the future, will be either "fn" or "pfn".
        "fn" implements the Frisch-Newton interior point algorithm
        described in Portnoy and Koenker (1997).
        The "pfn" method implements a variant of the
        algorithm proposed by Portnoy and Koenker (1997) including preprocessing steps, which
        a) can speed up the algorithm if N is very large but b) assumes independent observations.
        For details, you can either take a look at the Portnoy and Koenker paper, or "Fast Algorithms for the Quantile Regression Process"
        by Chernozhukov, Fernández-Val, and Melly (2019).

    multi_method: QuantregMultiMethodOpitons, optional
        Controls the algorithm for running the quantile regression process.
        Only relevant if more than one quantile regression is fit in one `quantreg` call.
        Options are 'cmf1', which is the default and implements algorithm 2 from Chernozhukov et al,
        'cmf2', which implements their algorithm 3, and 'none', which just loops over separate model
        calls.

    tol : float, optional
        The tolerance for the algorithm. Defaults to 1e-06. As in R's quantreg package, the
        algorithm stops when the relative change in the duality gap is less than tol.

    maxiter : int, optional
        The maximum number of iterations. If None, maxiter = the number of observations in the model
        (as in R's quantreg package via nit(3) = n).

    vcov : Union[VcovTypeOptions, dict[str, str]]
        Type of variance-covariance matrix for inference. Currently supported are
        "iid", "nid", and cluster robust errors, "iid" by default.
        All of "iid", "hetero"and "cluster" robust error are based on a kernel-based estimator as in Powell (1991).
        The "nid" method implements the robust sandwich estimator proposed in Hendricks and Koenker (1993).
        Any of "HC1 / HC2 / HC3 also works and is equivalent to "hetero".
        Cluster robust inference
        following Parente and Santos Silva (2016) can be specified via a dictionary with the keys "type" and "cluster".
        Only one-way clustering is supported.

    ssc : dict[str, Union[str, bool]], optional
        A dictionary specifying the small sample correction for inference.
        If None, uses default settings from `ssc_func()`. Note that by default, R's quantreg and Stata's qreg2 do not use
        small sample corrections. To match their behavior, set `ssc = pf.ssc(adj = False, cluster_adj = False)`.

    collin_tol : float, optional
        Tolerance for collinearity check, by default 1e-10.

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
    object
        An instance of the [Quantreg](/reference/estimation.quantreg.quantreg_.Quantreg.qmd) class or [FixestMulti](/reference/estimation.FixestMulti_.FixestMulti.qmd) class for multiple models specified via `fml`.

    Examples
    --------
    The following example regresses `Y` on `X1` and `X2` at the median (0.5 quantile):

    ```{python}
    import pyfixest as pf
    import pandas as pd
    import numpy as np

    data = pf.get_data()

    fit = pf.quantreg("Y ~ X1 + X2", data, quantile=0.5)
    fit.summary()
    ```

    For details around inference, estimation techniques, (fast) fitting and visualizing the full quantile regression
    process, please take a look at the dedicated [vignette](https://py-econometrics.github.io/pyfixest/quantile-regression.html).
    """
    # WLS currently not supported for quantile regression
    weights = None
    weights_type = "aweights"
    solver: SolverOptions = "np.linalg.solve"

    if ssc is None:
        ssc = ssc_func()

    context = {} if context is None else capture_context(context)

    fixef_rm = "none"
    fixef_tol = 1e-06
    fixef_maxiter = 100_000
    iwls_tol = 1e-08
    iwls_maxiter = 25

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
        fixef_rm=fixef_rm,
        fixef_maxiter=fixef_maxiter,
        collin_tol=collin_tol,
        copy_data=copy_data,
        store_data=store_data,
        lean=lean,
        fixef_tol=fixef_tol,
        weights_type=weights_type,
        use_compression=False,
        reps=None,
        seed=None,
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
        use_compression=False,
        reps=None,
        seed=seed,
        split=split,
        fsplit=fsplit,
        context=context,
        quantreg_method=method,
        quantreg_multi_method=multi_method,
    )

    # same checks as for Poisson regression
    fixest._prepare_estimation(
        estimation="quantreg" if not isinstance(quantile, list) else "quantreg_multi",
        fml=fml,
        vcov=vcov,
        vcov_kwargs=None,
        weights=weights,
        ssc=ssc,
        fixef_rm=fixef_rm,
        drop_intercept=drop_intercept,
        quantile=quantile,
        quantile_tol=tol,
        quantile_maxiter=maxiter,
    )
    if fixest._is_iv:
        raise NotImplementedError(
            "IV Estimation is not supported for Quantile Regression"
        )

    fixest._estimate_all_models(
        vcov=vcov,
        solver=solver,
        vcov_kwargs=None,
        iwls_tol=iwls_tol,
        iwls_maxiter=iwls_maxiter,
        collin_tol=collin_tol,
        separation_check=separation_check,
    )

    if fixest._is_multiple_estimation:
        return fixest
    else:
        return fixest.fetch_model(0, print_fml=False)
