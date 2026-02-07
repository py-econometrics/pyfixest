from collections.abc import Mapping
from typing import Any, Optional, Union

from pyfixest.estimation.api.utils import _estimation_input_checks
from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.fepois_ import Fepois
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.literals import (
    DemeanerBackendOptions,
    FixedRmOptions,
    SolverOptions,
    VcovTypeOptions,
    WeightsTypeOptions,
)
from pyfixest.utils.dev_utils import DataFrameType
from pyfixest.utils.utils import capture_context
from pyfixest.utils.utils import ssc as ssc_func


def fepois(
    fml: str,
    data: DataFrameType,  # type: ignore
    vcov: Optional[Union[VcovTypeOptions, dict[str, str]]] = None,
    vcov_kwargs: Optional[dict[str, Union[str, int]]] = None,
    weights: Union[None, str] = None,
    weights_type: WeightsTypeOptions = "aweights",
    ssc: Optional[dict[str, Union[str, bool]]] = None,
    fixef_rm: FixedRmOptions = "singleton",
    fixef_tol: float = 1e-06,
    fixef_maxiter: int = 10_000,
    iwls_tol: float = 1e-08,
    iwls_maxiter: int = 25,
    collin_tol: float = 1e-09,
    collin_tol_var: Optional[float] = None,
    separation_check: Optional[list[str]] = None,
    solver: SolverOptions = "scipy.linalg.solve",
    demeaner_backend: DemeanerBackendOptions = "numba",
    drop_intercept: bool = False,
    copy_data: bool = True,
    store_data: bool = True,
    lean: bool = False,
    context: Optional[Union[int, Mapping[str, Any]]] = None,
    split: Optional[str] = None,
    fsplit: Optional[str] = None,
) -> Union[Feols, Fepois, FixestMulti]:
    """
    Estimate Poisson regression model with fixed effects using the `ppmlhdfe` algorithm.

    Parameters
    ----------
    fml : str
        A two-sided formula string using fixest formula syntax.
        Syntax: "Y ~ X1 + X2 | FE1 + FE2". "|" separates left-hand side and fixed
        effects.
        Special syntax includes:
        - Stepwise regressions (sw, sw0)
        - Cumulative stepwise regression (csw, csw0)
        - Multiple dependent variables (Y1 + Y2 ~ X)
        - Interaction of variables (i(X1,X2))
        - Interacted fixed effects (fe1^fe2)
        Compatible with formula parsing via the formulaic module.

    data : DataFrameType
        A pandas or polars dataframe containing the variables in the formula.

    vcov : Union[VcovTypeOptions, dict[str, str]]
        Type of variance-covariance matrix for inference. Options include "iid",
          "hetero", "HC1", "HC2", "HC3", "NW" for Newey-West HAC standard errors,
        "DK" for Driscoll-Kraay HAC standard errors, or a dictionary for CRV1/CRV3 inference.
        Note that NW and DK require to pass additional keyword arguments via the `vcov_kwargs` argument.
        For time-series HAC, you need to pass the 'time_id' column. For panel-HAC, you need to add
        pass both 'time_id' and 'panel_id'. See `vcov_kwargs` for details.

    vcov_kwargs : Optional[dict[str, any]]
         Additional keyword arguments to pass to the vcov function. These keywoards include
        "lag" for the number of lag to use in the Newey-West (NW) and Driscoll-Kraay (DK) HAC standard errors.
        "time_id" for the time ID used for NW and DK standard errors, and "panel_id" for the panel
         identifier used for NW and DK standard errors. Currently, the the time difference between consecutive time
         periods is always treated as 1. More flexible time-step selection is work in progress.

    weights : Union[None, str], optional.
        Default is None. Weights for weighted Poisson regression. If None, all observations
        are weighted equally. If a string, the name of the column in `data` that
        contains the weights.

    weights_type: WeightsTypeOptions, optional
        Options include `aweights` or `fweights`. `aweights` implement analytic or
        precision weights, while `fweights` implement frequency weights. Frequency weights
        are useful for compressed count data where identical observations are aggregated.
        For details see this blog post: https://notstatschat.rbind.io/2020/08/04/weights-in-statistics/.

    ssc : str
        A ssc object specifying the small sample correction for inference.

    fixef_rm : FixedRmOptions
        Specifies whether to drop singleton fixed effects.
        Can be equal to "singletons" (default) or "none".
        "singletons" will drop singleton fixed effects. This will not impact point
        estimates but it will impact standard errors.

    fixef_tol: float, optional
        Tolerance for the fixed effects demeaning algorithm. Defaults to 1e-06.
        For LSMR-based backends (cupy, cupy32, cupy64, scipy), the tolerance is
        passed directly as LSMR's atol and btol parameters.

    fixef_maxiter: int, optional
         Maximum number of iterations for the demeaning algorithm. Defaults to 100,000.

    iwls_tol : Optional[float], optional
        Tolerance for IWLS convergence, by default 1e-08.

    iwls_maxiter : Optional[float], optional
        Maximum number of iterations for IWLS convergence, by default 25.

    collin_tol : float, optional
        Tolerance for collinearity check, by default 1e-10.

    collin_tol_var : float, optional
        Tolerance for the variance ratio collinearity check.
        Default is None: auto-enabled with threshold 1e-6 for LSMR
        backends (cupy, cupy32, cupy64, scipy), disabled for
        MAP backends (numba, rust, jax). Set to 0 to disable explicitly.

    separation_check: list[str], optional
        Methods to identify and drop separated observations.
        Either "fe" or "ir". Executes "fe" by default (when None).

    solver : SolverOptions, optional.
        The solver to use for the regression. Can be "np.linalg.lstsq",
        "np.linalg.solve", "scipy.linalg.solve", "scipy.sparse.linalg.lsqr" and "jax".
        Defaults to "scipy.linalg.solve".

    demeaner_backend: DemeanerBackendOptions, optional
        The backend to use for demeaning. Options include:
        - "numba" (default): CPU-based demeaning using Numba JIT via the Alternating Projections Algorithm.
        - "rust": CPU-based demeaning implemented in Rust via the Alternating Projections Algorithm.
        - "jax": CPU or GPU-accelerated using JAX (requires jax/jaxlib) via the Alternating Projections Algorithm.
        - "cupy" or "cupy64": GPU-accelerated using CuPy with float64 precision via direct application of the Frisch-Waugh-Lovell Theorem on sparse
          matrices (requires cupy & GPU, defaults to scipy/CPU if no GPU available)
        - "cupy32": GPU-accelerated using CuPy with float32 precision via direct application of the Frisch-Waugh-Lovell Theorem on sparse
          matrices (requires cupy & GPU, defaults to scipy/CPU and float64 if no GPU available)
        - "scipy": Direct application of the Frisch-Waugh-Lovell Theorem on sparse matrice.
          Forces to use a scipy-sparse backend even when cupy is installed and GPU is available.
        Defaults to "numba".

    drop_intercept : bool, optional
        Whether to drop the intercept from the model, by default False.

    copy_data : bool, optional
        Whether to copy the data before estimation, by default True.
        If set to False, the data is not copied, which can save memory but
        may lead to unintended changes in the input data outside of `fepois`.
        For example, the input data set is re-index within the function.
        As far as I know, the only other relevant case is
        when using interacted fixed effects, in which case you'll find
        a column with interacted fixed effects in the data set.

    store_data : bool, optional
        Whether to store the data in the model object, by default True.
        If set to False, the data is not stored in the model object, which can
        improve performance and save memory. However, it will no longer be possible
        to access the data via the `data` attribute of the model object. This has
        impact on post-estimation capabilities that rely on the data, e.g. `predict()`
        or `vcov()`.

    lean: bool, optional
        False by default. If True, then all large objects are removed from the
        returned result: this will save memory but will block the possibility
        to use many methods. It is recommended to use the argument vcov
        to obtain the appropriate standard-errors at estimation time,
        since obtaining different SEs won't be possible afterwards.

    context : int or Mapping[str, Any]
        A dictionary containing additional context variables to be used by
        formulaic during the creation of the model matrix. This can include
        custom factorization functions, transformations, or any other
        variables that need to be available in the formula environment.

    split: Optional[str]
        A character string, i.e. 'split = var'. If provided, the sample is split according to the
        variable and one estimation is performed for each value of that variable. If you also want
        to include the estimation for the full sample, use the argument fsplit instead.

    fsplit: Optional[str]
        This argument is the same as split but also includes the full sample as the first estimation.

    Returns
    -------
    object
        An instance of the [Fepois](/reference/estimation.fepois_.Fepois.qmd) class
        or an instance of class [FixestMulti](/reference/estimation.FixestMulti_.FixestMulti.qmd) for multiple models specified via `fml`.

    Examples
    --------
    The `fepois()` function can be used to estimate a simple Poisson regression
    model with fixed effects.
    The following example regresses `Y` on `X1` and `X2` with fixed effects for
    `f1` and `f2`: fixed effects are specified after the `|` symbol.

    ```{python}
    import pyfixest as pf

    data = pf.get_data(model = "Fepois")
    fit = pf.fepois("Y ~ X1 + X2 | f1 + f2", data)
    fit.summary()
    ```

    For more examples on the use of other function arguments, please take a look at the documentation of the [feols()](https://py-econometrics.github.io/pyfixest/reference/estimation.api.feols.html#pyfixest.estimation.api.feols) function.
    """
    if separation_check is None:
        separation_check = ["fe"]
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
        seed=None,
        split=split,
        fsplit=fsplit,
        context=context,
    )

    fixest._prepare_estimation(
        estimation="fepois",
        fml=fml,
        vcov=vcov,
        vcov_kwargs=vcov_kwargs,
        weights=weights,
        ssc=ssc,
        fixef_rm=fixef_rm,
        drop_intercept=drop_intercept,
    )
    if fixest._is_iv:
        raise NotImplementedError(
            "IV Estimation is not supported for Poisson Regression"
        )

    fixest._estimate_all_models(
        vcov=vcov,
        solver=solver,
        vcov_kwargs=vcov_kwargs,
        iwls_tol=iwls_tol,
        iwls_maxiter=iwls_maxiter,
        collin_tol=collin_tol,
        separation_check=separation_check,
        demeaner_backend=demeaner_backend,
        collin_tol_var=collin_tol_var,
    )

    if fixest._is_multiple_estimation:
        return fixest
    else:
        return fixest.fetch_model(0, print_fml=False)
