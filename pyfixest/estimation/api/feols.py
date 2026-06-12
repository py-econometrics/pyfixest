from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.api.utils import _estimation_input_checks
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.internals.demeaner_options import (
    _resolve_demeaner,
    _warn_if_deprecated_demeaner_backend,
    _warn_if_experimental_torch_demeaner,
)
from pyfixest.estimation.internals.literals import (
    FixedRmOptions,
    SolverOptions,
    VcovTypeOptions,
    WeightsTypeOptions,
)
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.utils.dev_utils import DataFrameType
from pyfixest.utils.utils import capture_context
from pyfixest.utils.utils import ssc as ssc_func


def feols(
    fml: str,
    data: DataFrameType,  # type: ignore
    vcov: VcovTypeOptions | dict[str, str] | None = None,
    vcov_kwargs: dict[str, str | int] | None = None,
    weights: None | str = None,
    ssc: dict[str, str | bool] | None = None,
    fixef_rm: FixedRmOptions = "singleton",
    collin_tol: float = 1e-09,
    drop_intercept: bool = False,
    copy_data: bool = True,
    store_data: bool = True,
    lean: bool = False,
    weights_type: WeightsTypeOptions = "aweights",
    solver: SolverOptions = "scipy.linalg.solve",
    demeaner: AnyDemeaner | None = None,
    use_compression: bool = False,
    reps: int = 100,
    context: int | Mapping[str, Any] | None = None,
    seed: int | None = None,
    split: str | None = None,
    fsplit: str | None = None,
) -> Feols | FixestMulti:
    """
    Estimate a linear regression model with fixed effects using fixest formula syntax.

    Returns an object of type [Feols](/reference/estimation.models.feols_.Feols.qmd) or
    [Feiv](/reference/estimation.models.feiv_.Feiv.qmd) (when using instrumental variables).

    Parameters
    ----------
    fml : str
        A three-sided formula string using fixest formula syntax.
        Syntax: "Y ~ X1 + X2 | FE1 + FE2 | X1 ~ Z1". "|" separates dependent variable,
        fixed effects, and instruments. Special syntax includes stepwise regressions,
        cumulative stepwise regression, multiple dependent variables,
        interaction of variables (i(X1,X2)), and interacted fixed effects (fe1^fe2).

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
        Default is None. Weights for WLS estimation. If None, all observations
        are weighted equally. If a string, the name of the column in `data` that
        contains the weights.

    ssc : str
        A ssc object specifying the small sample correction for inference.

    fixef_rm : FixedRmOptions
        Specifies whether to drop singleton fixed effects.
        Can be equal to "singleton" (default),
        or "none".
        "singletons" will drop singleton fixed effects. This will not impact point
        estimates but it will impact standard errors.

    collin_tol : float, optional
        Tolerance for collinearity check, by default 1e-10.

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

    weights_type: WeightsTypeOptions, optional
        Options include `aweights` or `fweights`. `aweights` implement analytic or
        precision weights, while `fweights` implement frequency weights. For details
        see this blog post: https://notstatschat.rbind.io/2020/08/04/weights-in-statistics/.

    solver : SolverOptions, optional.
        The solver to use for the regression. Can be "np.linalg.lstsq",
        "np.linalg.solve", "scipy.linalg.solve" and "scipy.sparse.linalg.lsqr".
        Defaults to "scipy.linalg.solve".

    demeaner : AnyDemeaner | None, optional
        Typed demeaner configuration. Controls the fixed-effects demeaning
        backend, tolerance, and iteration limits. Accepts a `MapDemeaner`
        or `LsmrDemeaner` instance. Defaults to
        `MapDemeaner()` (Rust MAP algorithm, tol=1e-6, maxiter=10_000).
        For other options - including the optional Numba backend and the
        torch-based LSMR backends - see the
        [Demeaner Backends vignette](../../how-to/demeaner-backends.qmd).

        .. deprecated::
            The ``cupy`` / ``scipy`` LSMR backends are deprecated and will
            be removed in a future release. Replacements:

            - cupy LSMR on GPU →
              ``LsmrDemeaner(backend="torch", device="cuda")``.
            - Scipy / cupy LSMR on CPU → ``LsmrDemeaner()``
              (the default within backend).

    use_compression: bool
        .. deprecated::
            ``use_compression`` is deprecated and will be removed in a future release.
            For out-of-memory regression on large datasets, consider using the
            `duckreg <https://github.com/py-econometrics/duckreg>`_ package instead.

        Whether to use sufficient statistics to losslessly fit the regression model
        on compressed data. False by default. If True, the model is estimated on
        compressed data, which can lead to a significant speed-up for large data sets.
        See the paper by Wong et al (2021) for more details https://arxiv.org/abs/2102.11297.
        Note that if `use_compression = True`, inference is lossless. If standard errors are
        clustered, a wild cluster bootstrap is employed. Parameters for the wild bootstrap
        can be specified via the `reps` and `seed` arguments. Additionally, note that for one-way
        fixed effects, the estimation method uses a Mundlak transform to "control" for the
        fixed effects. For two-way fixed effects, a two-way Mundlak transform is employed.
        For two-way fixed effects, the Mundlak transform is only identical to a two-way
        fixed effects model if the data set is a panel. We do not provide any checks for the
        panel status of the data set.

    reps: int
        Number of bootstrap repetitions. Only relevant for boostrap inference applied to
        compute cluster robust errors when `use_compression = True`.

    context : int or Mapping[str, Any]
        A dictionary containing additional context variables to be used by
        formulaic during the creation of the model matrix. This can include
        custom factorization functions, transformations, or any other
        variables that need to be available in the formula environment.

    seed: Optional[int]
        Seed for the random number generator. Only relevant for boostrap inference applied to
        compute cluster robust errors when `use_compression = True`.

    split: Optional[str]
        A character string, i.e. 'split = var'. If provided, the sample is split according to the
        variable and one estimation is performed for each value of that variable. If you also want
        to include the estimation for the full sample, use the argument fsplit instead.

    fsplit: Optional[str]
        This argument is the same as split but also includes the full sample as the first estimation.

    Returns
    -------
    object
        An instance of the [Feols](/reference/estimation.models.feols_.Feols.qmd) class,
        [Feiv](/reference/estimation.models.feiv_.Feiv.qmd) class (when using instrumental variables), or
        [FixestMulti](/reference/estimation.FixestMulti_.FixestMulti.qmd) class for multiple models specified via `fml`.

    Examples
    --------
    As in `fixest`, the [feols()](/reference/estimation.api.feols.feols.qmd) function can be used to
    estimate a simple linear regression model with fixed effects.
    The following example regresses `Y` on `X1` and `X2` with fixed effects for
    `f1` and `f2`: fixed effects are specified after the `|` symbol.

    ```{python}
    import pyfixest as pf

    data = pf.get_data()

    fit = pf.feols("Y ~ X1 + X2 | f1 + f2", data)
    fit.summary()
    ```

    Calling `feols()` returns an instance of the [Feols](/reference/estimation.models.feols_.Feols.qmd)
    class, whose `summary()`, `tidy()`, `coef()`, `se()`, `tstat()`, `pvalue()`,
    and `confint()` methods give access to the results.

    Inference is controlled via the `vcov` argument; clustered standard errors are
    specified via a dictionary, e.g. `{"CRV1": "f1"}`:

    ```{python}
    fit_crv = pf.feols("Y ~ X1 + X2 | f1 + f2", data, vcov={"CRV1": "f1"})
    fit_crv.tidy()
    ```

    Multiple models can be estimated in one call via the stepwise syntax
    (`sw()`, `sw0()`, `csw()`, `csw0()`), multiple dependent variables, and the
    `split`/`fsplit` arguments; the result is a `FixestMulti` object:

    ```{python}
    fits = pf.feols("Y ~ X1 + X2 | sw(f1, f2)", data)
    pf.etable(fits)
    ```

    IV estimation is supported via three-part formulas, where the last part
    specifies the first stage, e.g. `"Y ~ X2 | f1 + f2 | X1 ~ Z1"`.

    For the full tour — inference options, multiple estimation, IV diagnostics,
    custom transforms via `context`, prediction, and post-estimation methods
    (`wildboottest()`, `ritest()`, `ccv()`, `decompose()`) — see the
    [feols() by Example guide](/how-to/feols-examples.html) and the
    [formula syntax tutorial](/tutorials/formula-syntax.html).
    """
    if ssc is None:
        ssc = ssc_func()
    context = {} if context is None else capture_context(context)
    demeaner = _resolve_demeaner(demeaner)
    _warn_if_experimental_torch_demeaner(demeaner)
    _warn_if_deprecated_demeaner_backend(demeaner)

    if use_compression:
        warnings.warn(
            (
                "The `use_compression` argument is deprecated and will be removed in a future release. "
                "For out-of-memory regression on large datasets, consider using the "
                "`duckreg` package (https://github.com/py-econometrics/duckreg) instead. "
                "See https://github.com/py-econometrics/pyfixest/issues/1302 for context."
            ),
            DeprecationWarning,
            stacklevel=2,
        )

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
        weights_type=weights_type,
        use_compression=use_compression,
        reps=reps,
        seed=seed,
        split=split,
        fsplit=fsplit,
    )

    fixest = FixestMulti(
        data=data,
        copy_data=copy_data,
        store_data=store_data,
        lean=lean,
        weights_type=weights_type,
        use_compression=use_compression,
        reps=reps,
        seed=seed,
        split=split,
        fsplit=fsplit,
        context=context,
    )

    estimation = "feols" if not use_compression else "compression"

    fixest._prepare_estimation(
        estimation=estimation,
        fml=fml,
        vcov=vcov,
        vcov_kwargs=vcov_kwargs,
        weights=weights,
        ssc=ssc,
        fixef_rm=fixef_rm,
        drop_intercept=drop_intercept,
    )

    # demean all models: based on fixed effects x split x missing value combinations
    fixest._estimate_all_models(
        vcov=vcov,
        solver=solver,
        vcov_kwargs=vcov_kwargs,
        collin_tol=collin_tol,
        demeaner=demeaner,
    )

    if fixest._is_multiple_estimation:
        return fixest
    else:
        return fixest.fetch_model(0, print_fml=False)
