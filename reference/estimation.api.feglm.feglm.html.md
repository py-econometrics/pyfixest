# estimation.api.feglm.feglm { #pyfixest.estimation.api.feglm.feglm }

```python
estimation.api.feglm.feglm(
    fml,
    data,
    family,
    vcov=None,
    vcov_kwargs=None,
    ssc=None,
    fixef_rm='singleton',
    iwls_tol=1e-08,
    iwls_maxiter=25,
    collin_tol=1e-09,
    separation_check=None,
    solver='scipy.linalg.solve',
    demeaner=None,
    demeaner_backend=None,
    fixef_tol=None,
    fixef_maxiter=None,
    drop_intercept=False,
    copy_data=True,
    store_data=True,
    lean=False,
    context=None,
    split=None,
    fsplit=None,
    accelerate=True,
)
```

Estimate GLM regression models with fixed effects.

Supported families: [logit](/reference/estimation.models.felogit_.Felogit.qmd),
[probit](/reference/estimation.models.feprobit_.Feprobit.qmd),
[gaussian](/reference/estimation.models.fegaussian_.Fegaussian.qmd).

## References {.doc-section .doc-section-references}

- Bergé, L. (2018). Efficient estimation of maximum likelihood models with
  multiple fixed-effects: the R package FENmlm.
  [CREA Discussion Paper](https://ideas.repec.org/p/luc/wpaper/18-13.html).
- Correia, S., Guimaraes, P., & Zylkin, T. (2019). ppmlhdfe: Fast Poisson
  Estimation with High-Dimensional Fixed Effects.
  [The Stata Journal](https://journals.sagepub.com/doi/pdf/10.1177/1536867X20909691).
- Stammann, A. (2018). Fast and Feasible Estimation of Generalized Linear
  Models with High-Dimensional k-way Fixed Effects.
  [arXiv:1707.01815](https://arxiv.org/pdf/1707.01815).

## Parameters {.doc-section .doc-section-parameters}

| Name             | Type                                       | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Default                |
|------------------|--------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| fml              | str                                        | A two-sided formula string using fixest formula syntax. Syntax: "Y ~ X1 + X2 \| FE1 + FE2". "\|" separates left-hand side and fixed effects. Special syntax includes: - Stepwise regressions (sw, sw0) - Cumulative stepwise regression (csw, csw0) - Multiple dependent variables (Y1 + Y2 ~ X) - Interaction of variables (i(X1,X2)) - Interacted fixed effects (fe1^fe2) Compatible with formula parsing via the formulaic module.                                                                   | _required_             |
| data             | DataFrameType                              | A pandas or polars dataframe containing the variables in the formula.                                                                                                                                                                                                                                                                                                                                                                                                                                   | _required_             |
| family           | str                                        | The family of the GLM model. Options include "gaussian", "logit" and "probit".                                                                                                                                                                                                                                                                                                                                                                                                                          | _required_             |
| vcov             | Union\[VcovTypeOptions, dict\[str, str\]\] | Type of variance-covariance matrix for inference. Options include "iid",   "hetero", "HC1", "HC2", "HC3", "NW" for Newey-West HAC standard errors, "DK" for Driscoll-Kraay HAC standard errors, or a dictionary for CRV1/CRV3 inference. Note that NW and DK require to pass additional keyword arguments via the `vcov_kwargs` argument. For time-series HAC, you need to pass the 'time_id' column. For panel-HAC, you need to add pass both 'time_id' and 'panel_id'. See `vcov_kwargs` for details. | `None`                 |
| vcov_kwargs      | Optional\[dict\[str, any\]\]               | Additional keyword arguments to pass to the vcov function. These keywoards include "lag" for the number of lag to use in the Newey-West (NW) and Driscoll-Kraay (DK) HAC standard errors. "time_id" for the time ID used for NW and DK standard errors, and "panel_id" for the panel  identifier used for NW and DK standard errors. Currently, the the time difference between consecutive time  periods is always treated as 1. More flexible time-step selection is work in progress.                | `None`                 |
| ssc              | str                                        | A ssc object specifying the small sample correction for inference.                                                                                                                                                                                                                                                                                                                                                                                                                                      | `None`                 |
| fixef_rm         | FixedRmOptions                             | Specifies whether to drop singleton fixed effects. Can be equal to "singleton" (default), or "none". "singletons" will drop singleton fixed effects. This will not impact point estimates but it will impact standard errors.                                                                                                                                                                                                                                                                           | `'singleton'`          |
| iwls_tol         | Optional\[float\]                          | Tolerance for IWLS convergence, by default 1e-08.                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `1e-08`                |
| iwls_maxiter     | Optional\[float\]                          | Maximum number of iterations for IWLS convergence, by default 25.                                                                                                                                                                                                                                                                                                                                                                                                                                       | `25`                   |
| collin_tol       | float                                      | Tolerance for collinearity check, by default 1e-10.                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `1e-09`                |
| separation_check | list\[str\] \| None                        | Methods to identify and drop separated observations. Either "fe" or "ir". Executes "fe" by default (when None).                                                                                                                                                                                                                                                                                                                                                                                         | `None`                 |
| solver           | SolverOptions, optional.                   | The solver to use for the regression. Can be "np.linalg.lstsq", "np.linalg.solve", "scipy.linalg.solve", "scipy.sparse.linalg.lsqr" and "jax". Defaults to "scipy.linalg.solve".                                                                                                                                                                                                                                                                                                                        | `'scipy.linalg.solve'` |
| demeaner         | AnyDemeaner \| None                        | Typed demeaner configuration. Controls the fixed-effects demeaning backend, tolerance, and iteration limits. Accepts a `MapDemeaner`, `WithinDemeaner`, or `LsmrDemeaner` instance. Defaults to `MapDemeaner()` (numba MAP algorithm, tol=1e-6, maxiter=10_000).                                                                                                                                                                                                                                        | `None`                 |
| drop_intercept   | bool                                       | Whether to drop the intercept from the model, by default False.                                                                                                                                                                                                                                                                                                                                                                                                                                         | `False`                |
| copy_data        | bool                                       | Whether to copy the data before estimation, by default True. If set to False, the data is not copied, which can save memory but may lead to unintended changes in the input data outside of `fepois`. For example, the input data set is re-index within the function. As far as I know, the only other relevant case is when using interacted fixed effects, in which case you'll find a column with interacted fixed effects in the data set.                                                         | `True`                 |
| store_data       | bool                                       | Whether to store the data in the model object, by default True. If set to False, the data is not stored in the model object, which can improve performance and save memory. However, it will no longer be possible to access the data via the `data` attribute of the model object. This has impact on post-estimation capabilities that rely on the data, e.g. `predict()` or `vcov()`.                                                                                                                | `True`                 |
| lean             | bool                                       | False by default. If True, then all large objects are removed from the returned result: this will save memory but will block the possibility to use many methods. It is recommended to use the argument vcov to obtain the appropriate standard-errors at estimation time, since obtaining different SEs won't be possible afterwards.                                                                                                                                                                  | `False`                |
| context          | int or Mapping\[str, Any\]                 | A dictionary containing additional context variables to be used by formulaic during the creation of the model matrix. This can include custom factorization functions, transformations, or any other variables that need to be available in the formula environment.                                                                                                                                                                                                                                    | `None`                 |
| split            | str \| None                                | A character string, i.e. 'split = var'. If provided, the sample is split according to the variable and one estimation is performed for each value of that variable. If you also want to include the estimation for the full sample, use the argument fsplit instead.                                                                                                                                                                                                                                    | `None`                 |
| fsplit           | str \| None                                | This argument is the same as split but also includes the full sample as the first estimation.                                                                                                                                                                                                                                                                                                                                                                                                           | `None`                 |
| accelerate       | bool                                       | Whether to use acceleration tricks developed in the ppmlhdfe paper (warm start and adaptive fixed effects tolerance) for models with fixed effects. Produces numerically identical results faster, so we recommend to always set it to True.                                                                                                                                                                                                                                                            | `True`                 |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|--------|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        | object | An instance of the [Feglm](/reference/estimation.models.feglm_.Feglm.qmd) class (or one of its subclasses: [Felogit](/reference/estimation.models.felogit_.Felogit.qmd), [Feprobit](/reference/estimation.models.feprobit_.Feprobit.qmd), [Fegaussian](/reference/estimation.models.fegaussian_.Fegaussian.qmd)) or an instance of class [FixestMulti](/reference/estimation.FixestMulti_.FixestMulti.qmd) for multiple models specified via `fml`. |

## Examples {.doc-section .doc-section-examples}

The example below fits a logit model with fixed effects. As in `feols()`,
fixed effects are specified after the `|` symbol.

```{python}
import pyfixest as pf
import numpy as np

data = pf.get_data()
data["Y_bin"] = np.where(data["Y"] > 0, 1, 0)

fit_logit = pf.feglm("Y_bin ~ X1 + X2 | f1", data, family="logit")
fit_logit.summary()
```

To compare families with the same specification:

```{python}
fit_probit = pf.feglm("Y_bin ~ X1 + X2 | f1", data, family="probit")
fit_gaussian = pf.feglm("Y_bin ~ X1 + X2 | f1", data, family="gaussian")
pf.etable([fit_logit, fit_probit, fit_gaussian])
```

`PyFixest` also integrates with the [marginaleffects](https://marginaleffects.com/bonus/python.html) package.
To compute average marginal effects for the logit model above:

```{python}
from marginaleffects import avg_slopes
avg_slopes(fit_logit, variables="X1")
```

We can also compute marginal effects by group (group average marginal effects):

```{python}
avg_slopes(fit_logit, variables="X1", by="f1")
```

Shared arguments such as `vcov`, `ssc`, `split`, `fsplit`, `context`, and typed demeaners
work the same way as in `feols()`. See the [feols() reference](/reference/estimation.api.feols.feols.html)
for those details, and the [Marginal Effects guide](/how-to/marginaleffects.html)
for a compact post-estimation workflow.