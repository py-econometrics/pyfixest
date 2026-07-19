# Feols.decompose

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

## Parameters

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

## Returns

| Name | Type | Description |
|----|----|----|
|  | GelbachDecomposition | A GelbachDecomposition object with the decomposition results. Use `tidy()` and `etable()` to access the estimation results. |

## Examples

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
