# Feols

``` python
Feols(
    FixestFormula,
    data,
    ssc_dict,
    drop_singletons,
    drop_intercept,
    weights,
    weights_type,
    collin_tol,
    lookup_demeaned_data,
    solver='np.linalg.solve',
    demeaner=None,
    lookup_preconditioner=None,
    store_data=True,
    copy_data=True,
    lean=False,
    context=0,
    sample_split_var=None,
    sample_split_value=None,
)
```

Non user-facing class to estimate a linear regression via OLS.

Users should not directly instantiate this class, but rather use the [feols()](../reference/estimation.api.feols.feols.llms.md) function. Note that no demeaning is performed in this class: demeaning is performed in the FixestMulti class (to allow for caching of demeaned variables for multiple estimation).

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| Y | np.ndarray | Dependent variable, a two-dimensional numpy array. | *required* |
| X | np.ndarray | Independent variables, a two-dimensional numpy array. | *required* |
| weights | np.ndarray | Weights, a one-dimensional numpy array. | *required* |
| collin_tol | float | Tolerance level for collinearity checks. | *required* |
| coefnames | list\[str\] | Names of the coefficients (of the design matrix X). | *required* |
| weights_name | Optional\[str\] | Name of the weights variable. | *required* |
| weights_type | Optional\[str\] | Type of the weights variable. Either “aweights” for analytic weights or “fweights” for frequency weights. | *required* |
| solver | str, optional. | The solver to use for the regression. Can be “np.linalg.lstsq”, “np.linalg.solve”, “scipy.linalg.solve” and “scipy.sparse.linalg.lsqr”. Defaults to “scipy.linalg.solve”. | `'np.linalg.solve'` |
| context | int or Mapping\[str, Any\] | A dictionary containing additional context variables to be used by formulaic during the creation of the model matrix. This can include custom factorization functions, transformations, or any other variables that need to be available in the formula environment. | `0` |

## Attributes

| Name | Type | Description |
|----|----|----|
| \_method | str | Specifies the method used for regression, set to “feols”. |
| \_is_iv | bool | Indicates whether instrumental variables are used, initialized as False. |
| \_Y | np.ndarray | The demeaned dependent variable, a two-dimensional numpy array. |
| \_X | np.ndarray | The demeaned independent variables, a two-dimensional numpy array. |
| \_X_is_empty | bool | Indicates whether the X array is empty. |
| \_collin_tol | float | Tolerance level for collinearity checks. |
| \_coefnames | list | Names of the coefficients (of the design matrix X). |
| \_collin_vars | list | Variables identified as collinear. |
| \_collin_index | list | Indices of collinear variables. |
| \_Z | np.ndarray | Alias for the \_X array, used for calculations. |
| \_solver | str | The solver used for the regression. |
| \_weights | np.ndarray | Array of weights for each observation. |
| \_N | int | Number of observations. |
| \_k | int | Number of independent variables (or features). |
| \_support_crv3_inference | bool | Indicates support for CRV3 inference. |
| \_data | Any | Data used in the regression, to be enriched outside of the class. |
| \_fml | Any | Formula used in the regression, to be enriched outside of the class. |
| \_has_fixef | bool | Indicates whether fixed effects are used. |
| \_fixef | Any | Fixed effects used in the regression. |
| \_icovars | Any | Internal covariates, to be enriched outside of the class. |
| \_ssc_dict | dict | dictionary for sum of squares and cross products matrices. |
| \_tZX | np.ndarray | Transpose of Z multiplied by X, set in get_fit(). |
| \_tXZ | np.ndarray | Transpose of X multiplied by Z, set in get_fit(). |
| \_tZy | np.ndarray | Transpose of Z multiplied by Y, set in get_fit(). |
| \_tZZinv | np.ndarray | Inverse of the transpose of Z multiplied by Z, set in get_fit(). |
| \_beta_hat | np.ndarray | Estimated regression coefficients. |
| \_Y_hat_link | np.ndarray | Prediction at the level of the explanatory variable, i.e., the linear predictor X @ beta. |
| \_Y_hat_response | np.ndarray | Prediction at the level of the response variable, i.e., the expected predictor E(Y\|X). |
| \_u_hat | np.ndarray | Residuals of the regression model. |
| \_scores | np.ndarray | Scores used in the regression analysis. |
| \_hessian | np.ndarray | Hessian matrix used in the regression. |
| \_bread | np.ndarray | Bread matrix, used in calculating the variance-covariance matrix. |
| \_vcov_type | Any | Type of variance-covariance matrix used. |
| \_vcov_type_detail | Any | Detailed specification of the variance-covariance matrix type. |
| \_is_clustered | bool | Indicates if clustering is used in the variance-covariance calculation. |
| \_clustervar | Any | Variable used for clustering in the variance-covariance calculation. |
| \_G | Any | Group information used in clustering. |
| \_ssc | Any | Sum of squares and cross products matrix. |
| \_vcov | np.ndarray | Variance-covariance matrix of the estimated coefficients. |
| \_se | np.ndarray | Standard errors of the estimated coefficients. |
| \_tstat | np.ndarray | T-statistics of the estimated coefficients. |
| \_pvalue | np.ndarray | P-values associated with the t-statistics. |
| \_conf_int | np.ndarray | Confidence intervals for the estimated coefficients. |
| \_F_stat | Any | F-statistic for the model, set in get_Ftest(). |
| \_fixef_dict | dict | dictionary containing fixed effects estimates. |
| \_alpha | pd.DataFrame | A DataFrame with the estimated fixed effects. |
| \_sumFE | np.ndarray | Sum of all fixed effects for each observation. |
| \_rmse | float | Root mean squared error of the model. |
| \_r2 | float | R-squared value of the model. |
| \_r2_within | float | R-squared value computed on demeaned dependent variable. |
| \_adj_r2 | float | Adjusted R-squared value of the model. |
| \_adj_r2_within | float | Adjusted R-squared value computed on demeaned dependent variable. |
| \_solver | Literal\["np.linalg.lstsq", "np.linalg.solve", "scipy.linalg.solve", | “scipy.sparse.linalg.lsqr”\], default is “scipy.linalg.solve”. Solver to use for the estimation. |
| \_data | pd.DataFrame | The data frame used in the estimation. None if arguments `lean = True` or `store_data = False`. |
| \_model_name | str | The name of the model. Usually just the formula string. If split estimation is used, the model name will include the split variable and value. |
| \_model_name_plot | str | The name of the model used when plotting and summarizing models. Usually identical to `_model_name`. This might be different when pf.summary() or pf.coefplot() are called and models with identical \_model_name attributes are passed. In this case, the \_model_name_plot attribute will be modified. |
| \_quantile | Optional\[float\] | The quantile used for quantile regression. None if not a quantile regression. |
| \# special for did |  |  |
| \_res_cohort_eventtime_dict | Optional\[dict\[str, Any\]\] |  |
| \_yname | Optional\[str\] |  |
| \_gname | Optional\[str\] |  |
| \_tname | Optional\[str\] |  |
| \_idname | Optional\[str\] |  |
| \_att | Optional\[Any\] |  |
| test_treatment_heterogeneity | Callable\[…, Any\] |  |
| aggregate | Callable\[…, Any\] |  |
| iplot_aggregate | Callable\[…, Any\] |  |

## Methods

| Name | Description |
|----|----|
| [Feols.ccv](#pyfixest.estimation.models.feols_.Feols.ccv) | Compute the Causal Cluster Variance following Abadie et al (QJE 2023). |
| [Feols.decompose](#pyfixest.estimation.models.feols_.Feols.decompose) | Implement the Gelbach (2016) decomposition method for mediation analysis. |
| [Feols.fixef](#pyfixest.estimation.models.feols_.Feols.fixef) | Compute the coefficients of (swept out) fixed effects for a regression model. |
| [Feols.plot_ritest](#pyfixest.estimation.models.feols_.Feols.plot_ritest) | Plot the distribution of the Randomization Inference Statistics. |
| [Feols.predict](#pyfixest.estimation.models.feols_.Feols.predict) | Predict values of the model on new data. |
| [Feols.ritest](#pyfixest.estimation.models.feols_.Feols.ritest) | Conduct Randomization Inference (RI) test against a null hypothesis of |
| [Feols.update](#pyfixest.estimation.models.feols_.Feols.update) | Update coefficients for new observations using Sherman-Morrison formula. |
| [Feols.vcov](#pyfixest.estimation.models.feols_.Feols.vcov) | Compute covariance matrices for an estimated regression model. |
| [Feols.wald_test](#pyfixest.estimation.models.feols_.Feols.wald_test) | Conduct Wald test. |
| [Feols.wildboottest](#pyfixest.estimation.models.feols_.Feols.wildboottest) | Run a wild cluster bootstrap based on an object of type “Feols”. |

### Feols.ccv

``` python
ccv(treatment, cluster=None, seed=None, n_splits=8, pk=1, qk=1)
```

Compute the Causal Cluster Variance following Abadie et al (QJE 2023).

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| treatment |  | The name of the treatment variable. | *required* |
| cluster | str | The name of the cluster variable. None by default. If None, uses the cluster variable from the model fit. | `None` |
| seed | int | An integer to set the random seed. Defaults to None. | `None` |
| n_splits | int | The number of splits to use in the cross-fitting procedure. Defaults to 8. | `8` |
| pk | float | The proportion of sampled clusters. Defaults to 1, which corresponds to all clusters of the population being sampled. | `1` |
| qk | float | The proportion of sampled observations within each cluster. Defaults to 1, which corresponds to all observations within each cluster being sampled. | `1` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | pd.DataFrame | A DataFrame with inference based on the “Causal Cluster Variance” and “regular” CRV1 inference. |

#### Examples

``` python
import pyfixest as pf
import numpy as np

data = pf.get_data()
data["D"] = np.random.choice([0, 1], size=data.shape[0])

fit = pf.feols("Y ~ D", data=data, vcov={"CRV1": "group_id"})
fit.ccv(treatment="D", pk=0.05, qk=0.5, n_splits=8, seed=123).head()
```

|      | Estimate           | Std. Error | t value  | Pr(\>\|t\|) | 2.5%      | 97.5%    |
|------|--------------------|------------|----------|-------------|-----------|----------|
| CCV  | 0.2924377693929018 | 0.231115   | 1.265334 | 0.221887    | -0.193117 | 0.777993 |
| CRV1 | 0.292438           | 0.120141   | 2.43412  | 0.025568    | 0.040031  | 0.544845 |

### Feols.decompose

``` python
decompose(
    param=None,
    x1_vars=None,
    decomp_var=None,
    type='gelbach',
    cluster=None,
    combine_covariates=None,
    reps=1000,
    seed=None,
    nthreads=None,
    agg_first=None,
    only_coef=False,
    digits=4,
)
```

Implement the Gelbach (2016) decomposition method for mediation analysis.

Compares a short model `depvar on param` with the long model specified in the original feols() call.

For details, take a look at “When do covariates matter?” by Gelbach (2016, JoLe). You can find an ungated version of the paper on SSRN under the following link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1425737 .

When the initial regression is weighted, weights are interpreted as frequency weights. Inference is not yet supported for weighted models.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| param | str | The name of the focal covariate whose effect is to be decomposed into direct and indirect components with respect to the rest of the right-hand side. | `None` |
| x1_vars | list\[str\] | A list of covariates that are included in both the baseline and the full regressions. | `None` |
| decomp_var | str | The name of the focal covariate whose effect is to be decomposed into direct and indirect components with respect to the rest of the right-hand side. | `None` |
| type | str | The type of decomposition method to use. Defaults to “gelbach”, which currently is the only supported option. | `'gelbach'` |
| cluster | str \| None | The name of the cluster variable. If None, uses the cluster variable from the model fit. Defaults to None. | `None` |
| combine_covariates | dict\[str, list\[str\]\] \| None | A dictionary that specifies which covariates to combine into groups. See the example for how to use this argument. Defaults to None. | `None` |
| reps | int | The number of bootstrap iterations to run. Defaults to 1000. | `1000` |
| seed | int | An integer to set the random seed. Defaults to None. | `None` |
| nthreads | int | The number of threads to use for the bootstrap. Defaults to None. If None, uses all available threads minus one. | `None` |
| agg_first | bool | If True, use the ‘aggregate first’ algorithm described in Gelbach (2016). False by default, unless combine_covariates is provided. Recommended to set to True if combine_covariates is argument is provided. As a rule of thumb, the more covariates are combined, the larger the performance improvement. | `None` |
| only_coef | bool | Indicates whether to compute inference for the decomposition. Defaults to False. If True, skips the inference step and only returns the decomposition results. | `False` |
| digits | int | The number of digits to round the results to. Defaults to 4. | `4` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | GelbachDecomposition | A GelbachDecomposition object with the decomposition results. Use `tidy()` and `etable()` to access the estimation results. |

#### Examples

``` python
import re
import pyfixest as pf
from pyfixest.utils.dgps import gelbach_data

data = gelbach_data(nobs = 1000)
fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

# simple decomposition
gb = fit.decompose(decomp_var = "x1", reps = 10, nthreads = 1)
type(gb)

gb.tidy()
gb = fit.decompose(decomp_var = "x1", reps = 10, nthreads = 1, x1_vars = ["x21"])
# combine covariates
gb = fit.decompose(decomp_var = "x1", reps = 10, nthreads = 1, combine_covariates = {"g1": ["x21", "x22"], "g2": ["x23"]})
# supress inference
gb = fit.decompose(decomp_var = "x1", reps = 10, nthreads = 1, combine_covariates = {"g1": ["x21", "x22"], "g2": ["x23"]}, only_coef = True)
# print results
gb.etable()

# group covariates via regex
res = fit.decompose(decomp_var="x1", combine_covariates={"g1": re.compile("x2[1-2]"), "g2": re.compile("x23")})
```

### Feols.fixef

``` python
fixef(atol=1e-06, btol=1e-06)
```

Compute the coefficients of (swept out) fixed effects for a regression model.

This method creates the following attributes: - `_alpha` (pd.DataFrame): A DataFrame with the estimated fixed effects. - `_sumFE` (np.array): An array with the sum of fixed effects for each observation (i = 1, …, N).

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| atol | Float | Stopping tolerance for scipy.sparse.linalg.lsqr(). See https://docs.scipy.org/doc/ scipy/reference/generated/scipy.sparse.linalg.lsqr.html | `1e-6` |
| btol | Float | Another stopping tolerance for scipy.sparse.linalg.lsqr(). See https://docs.scipy.org/doc/ scipy/reference/generated/scipy.sparse.linalg.lsqr.html | `1e-6` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | dict\[str, dict\[str, float\]\] | A dictionary with the estimated fixed effects. |

#### Examples

Fixed effects are swept out during estimation and are not part of the coefficient table. `fixef()` computes them. The result is keyed by fixed effect term, then by level.

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
fe = fit.fixef()

fe.keys()
```

    dict_keys(['C(f1)'])

``` python
list(fe["C(f1)"].items())[:5]
```

    [('0.0', np.float64(0.4837574151832394)),
     ('1.0', np.float64(3.0661419612921605)),
     ('2.0', np.float64(-1.0947871507593956)),
     ('3.0', np.float64(0.33109523121756435)),
     ('4.0', np.float64(-1.2872533793542074))]

### Feols.plot_ritest

``` python
plot_ritest(plot_backend='lets_plot')
```

Plot the distribution of the Randomization Inference Statistics.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| plot_backend | str | The plotting backend to use. Defaults to “lets_plot”. Alternatively, “matplotlib” is available. | `'lets_plot'` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | A lets_plot or matplotlib figure with the distribution of the Randomization |  |
|  | Inference Statistics. |  |

### Feols.predict

``` python
predict(
    newdata=None,
    atol=1e-06,
    btol=1e-06,
    type='link',
    se_fit=False,
    interval=None,
    alpha=0.05,
)
```

Predict values of the model on new data.

Return a flat np.array with predicted values of the regression model. If new fixed effect levels are introduced in `newdata`, predicted values for such observations will be set to NaN.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| newdata | DataFrameType | A narwhals compatible DataFrame (polars, pandas, duckdb, etc). If None (default), the data used for fitting the model is used. | `None` |
| type | str | The type of prediction to be computed. Can be either “response” (default) or “link”. For linear models, both are identical. | `'link'` |
| atol | Float | Stopping tolerance for scipy.sparse.linalg.lsqr(). See https://docs.scipy.org/doc/ scipy/reference/generated/scipy.sparse.linalg.lsqr.html | `1e-6` |
| btol | Float | Another stopping tolerance for scipy.sparse.linalg.lsqr(). See https://docs.scipy.org/doc/ scipy/reference/generated/scipy.sparse.linalg.lsqr.html | `1e-6` |
| type | PredictionType | The type of prediction to be made. Can be either ‘link’ or ‘response’. Defaults to ‘link’. ‘link’ and ‘response’ lead to identical results for linear models. | `'link'` |
| se_fit | bool \| None | If True, the standard error of the prediction is computed. Only feasible for models without fixed effects. GLMs are not supported. Defaults to False. | `False` |
| interval | PredictionErrorOptions \| None | The type of interval to compute. Can be either ‘prediction’ or None. | `None` |
| alpha | float | The alpha level for the confidence interval. Defaults to 0.05. Only used if interval = “prediction” is not None. | `0.05` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | Union\[np.ndarray, pd.DataFrame\] | Returns a pd.Dataframe with columns “fit”, “se_fit” and CIs if argument “interval=prediction”. Otherwise, returns a np.ndarray with the predicted values of the model or the prediction standard errors if argument “se_fit=True”. |

#### Examples

In-sample predictions:

``` python
import pyfixest as pf

data = pf.get_data()
fit = pf.feols("Y ~ X1 + X2 | f1", data)
fit.predict()[:5]
```

    array([ 1.84475454, -0.17106206,  0.46970178, -0.74191438, -1.52651336])

Pass `newdata` to predict out of sample. Fixed effect levels that do not appear in the estimation sample return missing values.

``` python
fit.predict(newdata=data.head())
```

    array([ 1.80731416,         nan,         nan,  1.84475484, -0.17106194])

### Feols.ritest

``` python
ritest(
    resampvar,
    cluster=None,
    reps=100,
    type='randomization-c',
    rng=None,
    choose_algorithm='auto',
    store_ritest_statistics=False,
    level=0.95,
)
```

Conduct Randomization Inference (RI) test against a null hypothesis of `resampvar = 0`.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| resampvar | str | The name of the variable to be resampled. | *required* |
| cluster | str | The name of the cluster variable in case of cluster random assignment. If provided, `resampvar` is held constant within each `cluster`. Defaults to None. | `None` |
| reps | int | The number of randomization iterations. Defaults to 100. | `100` |
| type | str | The type of the randomization inference test. Can be “randomization-c” or “randomization-t”. Note that the “randomization-c” is much faster, while the “randomization-t” is recommended by Wu & Ding (JASA, 2021). | `'randomization-c'` |
| rng | np.random.Generator | A random number generator. Defaults to None. | `None` |
| choose_algorithm | str | The algorithm to use for the computation. Defaults to “auto”. The alternatives are “fast” and “slow”. The fast algorithm requires the optional `numba` extra (install via `pip install pyfixest[numba]`); without it, the fast path raises an `ImportError`. The slow path does not require numba. | `'auto'` |
| include_plot |  | Whether to include a plot of the distribution p-values. Defaults to False. | *required* |
| store_ritest_statistics | bool | Whether to store the simulated statistics of the RI procedure. Defaults to False. If True, stores the simulated statistics in the model object via the `ritest_statistics` attribute as a numpy array. | `False` |
| level | float | The level for the confidence interval of the randomization inference p-value. Defaults to 0.95. | `0.95` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | A pd.Series with the regression coefficient of `resampvar` and the p-value |  |
|  | of the RI test. Additionally, reports the standard error and the confidence |  |
|  | interval of the p-value. |  |

#### Examples

``` python
import pyfixest as pf
data = pf.get_data()
fit = pf.feols("Y ~ X1 + X2", data=data)

# Conduct a randomization inference test for the coefficient of X1
fit.ritest("X1", reps=1000)

# use randomization-t instead of randomization-c
fit.ritest("X1", reps=1000, type="randomization-t")

# store statistics for plotting
fit.ritest("X1", reps=1000, store_ritest_statistics=True)
```

    H0                                      X1=0
    ri-type                      randomization-c
    Estimate                 -0.9929357698186863
    Pr(>|t|)                                 0.0
    Std. Error (Pr(>|t|))                    0.0
    2.5% (Pr(>|t|))                          0.0
    97.5% (Pr(>|t|))                         0.0
    dtype: object

### Feols.update

``` python
update(X_new, y_new, inplace=False)
```

Update coefficients for new observations using Sherman-Morrison formula.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| X_new | np.ndarray | Covariates for new data points. Users expected to ensure conformability with existing data. | *required* |
| y_new | np.ndarray | Outcome values for new data points. | *required* |
| inplace | bool | Whether to update the model object in place. Defaults to False. | `False` |

#### Returns

| Name | Type       | Description           |
|------|------------|-----------------------|
|      | np.ndarray | Updated coefficients. |

#### Notes

Updates the coefficients in closed form via the Sherman-Morrison identity instead of refitting on the full sample. `X_new` has to include the intercept column. Models with fixed effects are not supported.

#### Examples

Fit on all but the last observation, then add it:

``` python
import numpy as np
import pyfixest as pf

data = pf.get_data().dropna()
fit = pf.feols("Y ~ X1 + X2", data.iloc[:-1])

last = data.iloc[[-1]]
X_new = np.column_stack(
    [np.ones(1), last["X1"].to_numpy(), last["X2"].to_numpy()]
)
y_new = last["Y"].to_numpy()

fit.update(X_new, y_new)
```

    array([ 0.88955689, -0.99519687, -0.17661729])

### Feols.vcov

``` python
vcov(vcov, vcov_kwargs=None, data=None)
```

Compute covariance matrices for an estimated regression model.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| vcov | Union\[str, dict\[str, str\]\] | A string or dictionary specifying the type of variance-covariance matrix to use for inference. If a string, it can be one of “iid”, “hetero”, “HC1”, “HC2”, “HC3”, “NW”, “DK”. If a dictionary, it should have the format {“CRV1”: “clustervar”} for CRV1 inference or {“CRV3”: “clustervar”} for CRV3 inference. Note that CRV3 inference is currently not supported for IV estimation. | *required* |
| vcov_kwargs | Optional\[dict\[str, any\]\] | Additional keyword arguments for the variance-covariance matrix. | `None` |
| data | DataFrameType \| None | The data used for estimation. If None, tries to fetch the data from the model object. Defaults to None. | `None` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | Feols | An instance of class [Feols](../reference/estimation.models.feols_.Feols.llms.md) with updated inference. |

#### Examples

Updates the variance estimator of a fitted model without refitting it. The model is modified in place and returned.

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
fit.vcov("iid").tidy()
```

|             | Estimate  | Std. Error | t value    | Pr(\>\|t\|) | 2.5%      | 97.5%     |
|-------------|-----------|------------|------------|-------------|-----------|-----------|
| Coefficient |           |            |            |             |           |           |
| X1          | -0.949526 | 0.066373   | -14.305943 | 0.0         | -1.079777 | -0.819274 |
| X2          | -0.174225 | 0.017596   | -9.901590  | 0.0         | -0.208755 | -0.139695 |

``` python
# switch to cluster-robust inference
fit.vcov({"CRV1": "f1"}).tidy()
```

|             | Estimate  | Std. Error | t value    | Pr(\>\|t\|)  | 2.5%      | 97.5%     |
|-------------|-----------|------------|------------|--------------|-----------|-----------|
| Coefficient |           |            |            |              |           |           |
| X1          | -0.949526 | 0.066557   | -14.266314 | 1.221245e-14 | -1.085650 | -0.813401 |
| X2          | -0.174225 | 0.018409   | -9.464130  | 2.267890e-10 | -0.211876 | -0.136575 |

See [On Small Sample Corrections](../explanation/ssc.llms.md) for how the `ssc` adjustments interact with each estimator.

### Feols.wald_test

``` python
wald_test(R=None, q=None, distribution='F')
```

Conduct Wald test.

Compute a Wald test for a linear hypothesis of the form R \* beta = q. where R is m x k matrix, beta is a k x 1 vector of coefficients, and q is m x 1 vector. By default, tests the joint null hypothesis that all coefficients are zero.

This method producues the following attriutes

\_dfd : int degree of freedom in denominator \_dfn : int degree of freedom in numerator \_wald_statistic : scalar Wald-statistics computed for hypothesis testing \_f_statistic : scalar Wald-statistics(when R is an indentity matrix, and q being zero vector) computed for hypothesis testing \_p_value : scalar corresponding p-value for statistics

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| R | array - like | The matrix R of the linear hypothesis. If None, defaults to an identity matrix. | `None` |
| q | array - like | The vector q of the linear hypothesis. If None, defaults to a vector of zeros. | `None` |
| distribution | str | The distribution to use for the p-value. Can be either “F” or “chi2”. Defaults to “F”. | `'F'` |

#### Returns

| Name | Type      | Description                                      |
|------|-----------|--------------------------------------------------|
|      | pd.Series | A pd.Series with the Wald statistic and p-value. |

#### Examples

``` python
import numpy as np
import pandas as pd
import pyfixest as pf

data = pf.get_data()
fit = pf.feols("Y ~ X1 + X2| f1", data, vcov={"CRV1": "f1"}, ssc=pf.ssc(k_adj=False))

R = np.array([[1,-1]] )
q = np.array([0.0])

# Wald test
fit.wald_test(R=R, q=q, distribution = "chi2")
f_stat = fit._f_statistic
p_stat = fit._p_value

print(f"Python f_stat: {f_stat}")
print(f"Python p_stat: {p_stat}")
```

    Python f_stat: 126.40650474043508
    Python p_stat: 2.505309282813844e-29

### Feols.wildboottest

``` python
wildboottest(
    reps,
    cluster=None,
    param=None,
    weights_type='rademacher',
    impose_null=True,
    bootstrap_type='11',
    seed=None,
    k_adj=True,
    G_adj=True,
    parallel=False,
    return_bootstrapped_t_stats=False,
)
```

Run a wild cluster bootstrap based on an object of type “Feols”.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| reps | int | The number of bootstrap iterations to run. | *required* |
| cluster | Union\[str, None\] | The variable used for clustering. Defaults to None. If None, then uses the variable specified in the model’s `clustervar` attribute. If no `_clustervar` attribute is found, runs a heteroskedasticity- robust bootstrap. | `None` |
| param | Union\[str, None\] | A string of length one, containing the test parameter of interest. Defaults to None. | `None` |
| weights_type | str | The type of bootstrap weights. Options are ‘rademacher’, ‘mammen’, ‘webb’, or ‘normal’. Defaults to ‘rademacher’. | `'rademacher'` |
| impose_null | bool | Indicates whether to impose the null hypothesis on the bootstrap DGP. Defaults to True. | `True` |
| bootstrap_type | str | A string of length one to choose the bootstrap type. Options are ‘11’, ‘31’, ‘13’, or ‘33’. Defaults to ‘11’. | `'11'` |
| seed | Union\[int, None\] | An option to provide a random seed. Defaults to None. | `None` |
| k_adj | bool | Indicates whether to apply a small sample adjustment for the number of observations and covariates. Defaults to True. | `True` |
| G_adj | bool | Indicates whether to apply a small sample adjustment for the number of clusters. Defaults to True. | `True` |
| parallel | bool | Indicates whether to run the bootstrap in parallel. Defaults to False. | `False` |
| seed | Union\[str, None\] | An option to provide a random seed. Defaults to None. | `None` |
| return_bootstrapped_t_stats | bool, optional: | If True, the method returns a tuple of the regular output and the bootstrapped t-stats. Defaults to False. | `False` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | pd.DataFrame | A DataFrame with the original, non-bootstrapped t-statistic and bootstrapped p-value, along with the bootstrap type, inference type (HC vs CRV), and whether the null hypothesis was imposed on the bootstrap DGP. If `return_bootstrapped_t_stats` is True, the method returns a tuple of the regular output and the bootstrapped t-stats. |

#### Examples

``` python
import re
import pyfixest as pf

data = pf.get_data()
fit = pf.feols("Y ~ X1 + X2 | f1", data)

fit.wildboottest(
    param = "X1",
    reps=1000,
    seed = 822
)

fit.wildboottest(
    param = "X1",
    reps=1000,
    seed = 822,
    bootstrap_type = "31"
)
```

    param                    X1
    t value          -14.843741
    Pr(>|t|)                0.0
    bootstrap_type           31
    inference                HC
    impose_null            True
    ssc                       1
    dtype: object
