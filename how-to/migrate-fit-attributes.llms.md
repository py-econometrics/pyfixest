# Migrating Fitted-Model Attributes from PyFixest 0.60

migration

fitted models

How to migrate fitted-model attributes to PyFixest 0.70.

PyFixest 0.70 exposes fitted-model state through typed components. Private attributes from 0.60 are not supported, except for the compatibility names listed below. Accessing a removed private name raises an `AttributeError` with a link to this guide. Some private names are still used internally; their presence does not make them part of the supported API.

## Find the replacement

Use the component that owns the information your code needs:

| Information | Component or method |
|----|----|
| Formula, dependent variable and fixed effects | `fit.model` |
| Estimation options | `fit.options` |
| Supported operations (`_support_*`, `_supports_*`) | `fit.capabilities` |
| Observation counts and dropped rows | `fit.sample_info` |
| Fixed-effect counts (`_k_fe`, `_n_fe`) | `fit.fixef_counts` |
| Original model inputs | `fit.model_matrix` |
| Observation weights (`_weights`) | `fit.observation_weights` |
| Demeaned inputs for linear and IV models | `fit.within_data` |
| Within preconditioner (`_preconditioner`) | `fit.preconditioner` |
| GLM working values | `fit.working_state` |
| GLM convergence (`_convergence`) | `fit.convergence` |
| Quantile solver state (`_it`, `_x_final`, etc.) | `fit.solution` |
| Collinear regressors or instruments (`_collin_*`) | `fit.collinearity`, `fit.collinearity_instruments` |
| Coefficients, standard errors and confidence intervals | `fit.coeftable` |
| Covariance matrix and inference settings | `fit.variance_covariance` |
| Fit statistics and fitted values | `fit.fitstat`, `fit.fitted_values` |
| Sandwich components, when available | `fit.sandwich` |
| IV first stage and diagnostics | `fit.first_stage` |
| Wald test results (`_dfn`, `_dfd`, `_f_statistic`, etc.) | `fit.wald` |
| Randomization inference results (`_ritest_*`) | `fit.ritest_statistics` |
| Saturated event-study design (`_yname`, `_att`, etc.) | `fit.event_study_design` |
| Fixed-effect estimates (`_fixef_dict`) | `fit.fixef()`, `fit.fixef_estimates` |

Components differ between estimators, and some data may be unavailable when storage options such as `lean=True` are used. Their shapes, weighting, and contents may also differ from the 0.60 private attributes. See the [API reference](../reference/index.llms.md) for each component’s contract. In particular, `fit.resid()` returns response residuals; the old `_u_hat` on weighted linear and IV fits was multiplied by the square root of the observation weights. `fit.fixef_estimates` is available after calling `fit.fixef()`.

## Compatibility names

Names without a leading underscore and three documented IV names remain as deprecated aliases or raise a specific removal message. Names read by released versions of [marginaleffects](https://github.com/vincentarelbundock/pymarginaleffects) and [maketables](https://github.com/py-econometrics/maketables) remain available without a warning. These aliases can reflect estimation or inference changes since 0.60; see the [changelog](../changelog.llms.md).

After compatible releases of both packages use typed state, the plan is to turn on warnings for their aliases. No removal version is scheduled yet.

Other retained aliases emit `PyfixestDeprecationWarning` (a `FutureWarning`). To find their use in your test suite, turn the warning into an error:

``` python
import warnings
from pyfixest.errors import PyfixestDeprecationWarning

warnings.simplefilter("error", PyfixestDeprecationWarning)
```

With pytest, add `"error::pyfixest.errors.PyfixestDeprecationWarning"` to the `filterwarnings` setting. Silent aliases do not emit a warning, while removed names raise `AttributeError`.

The table lists the retained compatibility names. Entries for `Feols` also apply to IV, GLM, and quantile models, unless a more specific entry is listed.

| Old attribute | Model classes | Use instead | Status |
|----|----|----|----|
| `FixestFormula` | `Feols` | `fit.model.parsed_formula` | removed: the parsed formula interface changed, including its stages and fixed effects |
| `_fml` | `Feols` | `fit.model.formula` | silent alias |
| `_method` | `Feols` | `fit.model.method` | silent alias |
| `_fixef` | `Feols` | `fit.model.fixef` | silent alias |
| `_has_fixef` | `Feols` | `fit.model.has_fixef` | silent alias |
| `_depvar` | `Feols` | `fit.model.depvar` | silent alias |
| `_sample_split_value` | `Feols` | `fit.model.sample_split.value` | silent alias |
| `_sample_split_var` | `Feols` | `fit.model.sample_split.var` | silent alias |
| `maxiter` | `Feglm` | `fit.options.maxiter` | deprecated alias |
| `tol` | `Feglm` | `fit.options.tol` | deprecated alias |
| `separation_check` | `Feglm` | `fit.options.separation_check` | deprecated alias |
| `_N` | `Feols` | `fit.sample_info.n_obs` | silent alias |
| `n_separation_na` | `Feols` | `fit.sample_info.dropped_by_stage.separation` | deprecated alias |
| `na_index` | `Feols` | `fit.sample_info.dropped_row_index` | removed: it only listed the rows a GLM dropped for separation, by position in the already filtered data |
| `_vcov` | `Feols` | `fit.variance_covariance.vcov` | silent alias |
| `_vcov_type` | `Feols` | `fit.variance_covariance.spec.vcov_type` | silent alias |
| `_clustervar` | `Feols` | `fit.variance_covariance.spec.clustervar` | silent alias |
| `_G` | `Feols` | `fit.variance_covariance.G` | silent alias |
| `_pvalue` | `Feols` | `fit.coeftable.pvalue` | silent alias |
| `get_performance` | `Feols` | `fit.fitstat` | deprecated alias |
| `_rmse` | `Feols` | `fit.fitstat.rmse` | silent alias |
| `_r2` | `Feols` | `fit.fitstat.r2` | silent alias |
| `_r2_within` | `Feols` | `fit.fitstat.r2_within` | silent alias |
| `_adj_r2` | `Feols` | `fit.fitstat.adj_r2` | silent alias |
| `_r2_adj` | `Feols` | `fit.fitstat.adj_r2` | silent alias |
| `_adj_r2_within` | `Feols` | `fit.fitstat.adj_r2_within` | silent alias |
| `deviance` | `Feols` | `fit.fitstat.deviance` | silent alias |
| `deviance` | `Feglm` | `fit.fitstat.deviance` | silent alias |
| `_model_1st_stage` | `Feiv` | `fit.first_stage.model` | deprecated alias |
| `_f_stat_1st_stage` | `Feiv` | `fit.first_stage.diagnostics.f_stat` | silent alias |
| `_eff_F` | `Feiv` | `fit.first_stage.diagnostics.eff_f` | deprecated alias |
