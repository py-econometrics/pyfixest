# PyFixest Function Reference

## Estimation

User-facing estimation functions. Everything in this reference is available under the `pf.` namespace after `import pyfixest as pf`.

|  |  |
|----|----|
| [feols](../reference/estimation.api.feols.feols.llms.md#pyfixest.estimation.api.feols.feols) | Estimate a linear regression model with fixed effects using fixest formula syntax. |
| [fepois](../reference/estimation.api.fepois.fepois.llms.md#pyfixest.estimation.api.fepois.fepois) | Estimate Poisson regression model with fixed effects using the `ppmlhdfe` algorithm. |
| [feglm](../reference/estimation.api.feglm.feglm.llms.md#pyfixest.estimation.api.feglm.feglm) | Estimate GLM regression models with fixed effects. |
| [quantreg](../reference/estimation.api.quantreg.quantreg.llms.md#pyfixest.estimation.api.quantreg.quantreg) | Fit a quantile regression model using the interior point algorithm from Portnoy and Koenker (1997). |

## Difference-in-Differences

Difference-in-differences and event-study estimators, plus panel treatment visualization.

|  |  |
|----|----|
| [event_study](../reference/did.estimation.event_study.llms.md#pyfixest.did.estimation.event_study) | Estimate Event Study Model. |
| [did2s](../reference/did.estimation.did2s.llms.md#pyfixest.did.estimation.did2s) | Estimate a Difference-in-Differences model using Gardner’s two-step DID2S estimator. |
| [lpdid](../reference/did.estimation.lpdid.llms.md#pyfixest.did.estimation.lpdid) | Local projections approach to estimation. |
| [SaturatedEventStudy](../reference/did.saturated_twfe.SaturatedEventStudy.llms.md#pyfixest.did.saturated_twfe.SaturatedEventStudy) | Saturated event study with cohort-specific effect curves. |
| [panelview](../reference/did.visualize.panelview.llms.md#pyfixest.did.visualize.panelview) | Generate a panel view of the treatment variable over time for each unit. |

## Multiple Hypothesis Testing

Family-wise error-rate corrections for multiple hypothesis testing.

|  |  |
|----|----|
| [bonferroni](../reference/estimation.post_estimation.multcomp.bonferroni.llms.md#pyfixest.estimation.post_estimation.multcomp.bonferroni) | Compute Bonferroni adjusted p-values for multiple hypothesis testing. |
| [rwolf](../reference/estimation.post_estimation.multcomp.rwolf.llms.md#pyfixest.estimation.post_estimation.multcomp.rwolf) | Compute Romano-Wolf adjusted p-values for multiple hypothesis testing. |
| [wyoung](../reference/estimation.post_estimation.multcomp.wyoung.llms.md#pyfixest.estimation.post_estimation.multcomp.wyoung) | Compute the Westfall-Young adjusted p-values for multiple hypothesis testing. |

## Estimation Classes

Fitted-model classes returned by the estimation functions. Users do not construct these directly; they are the objects `feols()`, `fepois()`, `feglm()`, and `quantreg()` return.

|  |  |
|----|----|
| [Feols](../reference/estimation.models.feols_.Feols.llms.md#pyfixest.estimation.models.feols_.Feols) | Non user-facing class to estimate a linear regression via OLS. |
| [Fepois](../reference/estimation.models.fepois_.Fepois.llms.md#pyfixest.estimation.models.fepois_.Fepois) | Estimate a Poisson regression model. |
| [Feiv](../reference/estimation.models.feiv_.Feiv.llms.md#pyfixest.estimation.models.feiv_.Feiv) | Non user-facing class to estimate an IV model using a 2SLS estimator. |
| [Feglm](../reference/estimation.models.feglm_.Feglm.llms.md#pyfixest.estimation.models.feglm_.Feglm) | Base class for the estimation of a fixed-effects GLM model. |
| [FixestMulti](../reference/estimation.FixestMulti_.FixestMulti.llms.md#pyfixest.estimation.FixestMulti_.FixestMulti) | Results container holding every model fitted by one public-API call. |
| [Quantreg](../reference/estimation.quantreg.quantreg_.Quantreg.llms.md#pyfixest.estimation.quantreg.quantreg_.Quantreg) | Quantile regression model. |

## Post-Estimation Methods

Methods available on a fitted model object, e.g. the result of a call to `feols()`. Defined on `Feols` and shared by `Fepois`, `Feglm`, `Feiv`, and `Quantreg` through inheritance.

|  |  |
|----|----|
| [Feols.tidy](../reference/estimation.models.feols_.Feols.tidy.llms.md#pyfixest.estimation.models.feols_.Feols.tidy) | Tidy model outputs. |
| [Feols.coef](../reference/estimation.models.feols_.Feols.coef.llms.md#pyfixest.estimation.models.feols_.Feols.coef) | Estimated coefficients as a pandas Series. |
| [Feols.se](../reference/estimation.models.feols_.Feols.se.llms.md#pyfixest.estimation.models.feols_.Feols.se) | Coefficient standard errors as a pandas Series. |
| [Feols.tstat](../reference/estimation.models.feols_.Feols.tstat.llms.md#pyfixest.estimation.models.feols_.Feols.tstat) | Coefficient t-statistics as a pandas Series. |
| [Feols.pvalue](../reference/estimation.models.feols_.Feols.pvalue.llms.md#pyfixest.estimation.models.feols_.Feols.pvalue) | Coefficient p-values as a pandas Series. |
| [Feols.confint](../reference/estimation.models.feols_.Feols.confint.llms.md#pyfixest.estimation.models.feols_.Feols.confint) | Fitted model confidence intervals. |
| [Feols.resid](../reference/estimation.models.feols_.Feols.resid.llms.md#pyfixest.estimation.models.feols_.Feols.resid) | Fitted model residuals. |
| [Feols.vcov](../reference/estimation.models.feols_.Feols.vcov.llms.md#pyfixest.estimation.models.feols_.Feols.vcov) | Compute covariance matrices for an estimated regression model. |
| [Feols.predict](../reference/estimation.models.feols_.Feols.predict.llms.md#pyfixest.estimation.models.feols_.Feols.predict) | Predict values of the model on new data. |
| [Feols.fixef](../reference/estimation.models.feols_.Feols.fixef.llms.md#pyfixest.estimation.models.feols_.Feols.fixef) | Compute the coefficients of (swept out) fixed effects for a regression model. |
| [Feols.get_performance](../reference/estimation.models.feols_.Feols.get_performance.llms.md#pyfixest.estimation.models.feols_.Feols.get_performance) | Get Goodness-of-Fit measures. |
| [Feols.wald_test](../reference/estimation.models.feols_.Feols.wald_test.llms.md#pyfixest.estimation.models.feols_.Feols.wald_test) | Conduct Wald test. |
| [Feols.wildboottest](../reference/estimation.models.feols_.Feols.wildboottest.llms.md#pyfixest.estimation.models.feols_.Feols.wildboottest) | Run a wild cluster bootstrap based on an object of type “Feols”. |
| [Feols.ritest](../reference/estimation.models.feols_.Feols.ritest.llms.md#pyfixest.estimation.models.feols_.Feols.ritest) | Conduct Randomization Inference (RI) test against a null hypothesis of |
| [Feols.ccv](../reference/estimation.models.feols_.Feols.ccv.llms.md#pyfixest.estimation.models.feols_.Feols.ccv) | Compute the Causal Cluster Variance following Abadie et al (QJE 2023). |
| [Feols.decompose](../reference/estimation.models.feols_.Feols.decompose.llms.md#pyfixest.estimation.models.feols_.Feols.decompose) | Implement the Gelbach (2016) decomposition method for mediation analysis. |
| [Feols.update](../reference/estimation.models.feols_.Feols.update.llms.md#pyfixest.estimation.models.feols_.Feols.update) | Update coefficients for new observations using Sherman-Morrison formula. |
| [Feols.evalue](../reference/estimation.models.feols_.Feols.evalue.llms.md#pyfixest.estimation.models.feols_.Feols.evalue) | Compute coefficient-wise SAVI e-values. |
| [Feols.pvalue_savi](../reference/estimation.models.feols_.Feols.pvalue_savi.llms.md#pyfixest.estimation.models.feols_.Feols.pvalue_savi) | Compute coefficient-wise SAVI sequential p-values. |

## Summarize and Visualize

Summary tables and coefficient plots for fitted models.

|  |  |
|----|----|
| [summary](../reference/report.summary.llms.md#pyfixest.report.summary) | Print a summary of estimation results for each estimated model. |
| [etable](../reference/report.etable.llms.md#pyfixest.report.etable) | Generate a table summarizing the results of multiple regression models. |
| [coefplot](../reference/report.coefplot.llms.md#pyfixest.report.coefplot) | Plot model coefficients with confidence intervals. |
| [iplot](../reference/report.iplot.llms.md#pyfixest.report.iplot) | Plot model coefficients for variables interacted via “i()” syntax, with |
| [qplot](../reference/report.qplot.llms.md#pyfixest.report.qplot) | Plot regression quantiles. |

## Data Sets

Synthetic data generators used throughout the documentation and tests.

|  |  |
|----|----|
| [get_data](../reference/utils.utils.get_data.llms.md#pyfixest.utils.utils.get_data) | Create a random example data set. |
| [get_ivf_data](../reference/utils.dgps.get_ivf_data.llms.md#pyfixest.utils.dgps.get_ivf_data) | Synthetic data for the motherhood penalty IV application (IVF instrument). |
| [get_bartik_data](../reference/utils.dgps.get_bartik_data.llms.md#pyfixest.utils.dgps.get_bartik_data) | Synthetic data for a Bartik (shift-share) IV application on immigration and wages. |
| [get_encouragement_data](../reference/utils.dgps.get_encouragement_data.llms.md#pyfixest.utils.dgps.get_encouragement_data) | Synthetic data for an A/B encouragement design IV application. |
| [get_twin_data](../reference/utils.dgps.get_twin_data.llms.md#pyfixest.utils.dgps.get_twin_data) | Generate twin study data for returns to education. |
| [get_worker_panel](../reference/utils.dgps.get_worker_panel.llms.md#pyfixest.utils.dgps.get_worker_panel) | Generate a worker-firm panel dataset with two-way fixed effects. |
| [get_motherhood_event_study_data](../reference/utils.dgps.get_motherhood_event_study_data.llms.md#pyfixest.utils.dgps.get_motherhood_event_study_data) | Generate a fertility-timing panel for motherhood-penalty event studies. |

## Formula Parsing & Model Matrix

Internal APIs for formula parsing and model matrix construction.

|  |  |
|----|----|
| [Formula](../reference/estimation.formula.parse.Formula.llms.md#pyfixest.estimation.formula.parse.Formula) | A formulaic-compliant formula. |
| [ModelMatrix](../reference/estimation.formula.model_matrix.ModelMatrix.llms.md#pyfixest.estimation.formula.model_matrix.ModelMatrix) | A wrapper around formulaic.ModelMatrix for the specification of PyFixest models. |
| [factor_interaction](../reference/estimation.formula.factor_interaction.factor_interaction.llms.md#pyfixest.estimation.formula.factor_interaction.factor_interaction) | Fixest-style i() operator for categorical encoding with interactions. |

## Demeaning

Fixed-effects demeaning: the `demean()` workhorse and the configurable backends. See the [Choosing a Demeaner Backend](../how-to/demeaner-backends.llms.md) guide for how to pick one.

|  |  |
|----|----|
| [demean](../reference/estimation.demean.llms.md#pyfixest.estimation.demean) | Demean an array. |
| [BaseDemeaner](../reference/demeaners.BaseDemeaner.llms.md#pyfixest.demeaners.BaseDemeaner) | Base configuration shared by all fixed-effects demeaners. |
| [MapDemeaner](../reference/demeaners.MapDemeaner.llms.md#pyfixest.demeaners.MapDemeaner) | Method of Alternating Projections (MAP) demeaner. |
| [LsmrDemeaner](../reference/demeaners.LsmrDemeaner.llms.md#pyfixest.demeaners.LsmrDemeaner) | Sparse LSMR demeaner. |
| [Preconditioner](../reference/core.demean.Preconditioner.llms.md#pyfixest.core.demean.Preconditioner) | Opaque handle to a pre-built within preconditioner (Additive Schwarz or |

## Misc / Utilities

Other PyFixest internals and utilities.

|  |  |
|----|----|
| [detect_singletons](../reference/core.detect_singletons.detect_singletons.llms.md#pyfixest.core.detect_singletons.detect_singletons) | Detect singleton fixed effects in a dataset. |
| [ssc](../reference/utils.utils.ssc.llms.md#pyfixest.utils.utils.ssc) | Set the small sample correction factor applied in `get_ssc()`. |
| [get_ssc](../reference/utils.utils.get_ssc.llms.md#pyfixest.utils.utils.get_ssc) | Compute small sample adjustment factors. |
| [optimal_mixture_precision](../reference/estimation.post_estimation.savi.optimal_mixture_precision.llms.md#pyfixest.estimation.post_estimation.savi.optimal_mixture_precision) | Compute the mixture precision that minimizes SAVI sequence width |
| [model_matrix_fixest](../reference/estimation.deprecated.model_matrix_fixest_.model_matrix_fixest.llms.md#pyfixest.estimation.deprecated.model_matrix_fixest_.model_matrix_fixest) | Create model matrices for fixed effects estimation. |
