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
| [EventStudyDesign](../reference/did.saturated_twfe.EventStudyDesign.llms.md#pyfixest.did.saturated_twfe.EventStudyDesign) | Event-study design a fitted event study was estimated on. |
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

## Fitted State

Data classes and model-matrix components exposed by fitted estimators.

|  |  |
|----|----|
| [EstimationOptions](../reference/estimation.state.EstimationOptions.llms.md#pyfixest.estimation.state.EstimationOptions) | Estimation options a fitted model was built with. |
| [GlmEstimationOptions](../reference/estimation.state.GlmEstimationOptions.llms.md#pyfixest.estimation.state.GlmEstimationOptions) | Estimation options of a GLM fit, including the IRLS options. |
| [QuantregEstimationOptions](../reference/estimation.state.QuantregEstimationOptions.llms.md#pyfixest.estimation.state.QuantregEstimationOptions) | Estimation options of a quantile-regression fit. |
| [ModelDescription](../reference/estimation.state.ModelDescription.llms.md#pyfixest.estimation.state.ModelDescription) | What a fitted model estimates: its formula, estimator, and design names. |
| [ModelMatrix](../reference/estimation.state.ModelMatrix.llms.md#pyfixest.estimation.state.ModelMatrix) | A wrapper around formulaic.ModelMatrix for the specification of PyFixest models. |
| [ObservationWeights](../reference/estimation.state.ObservationWeights.llms.md#pyfixest.estimation.state.ObservationWeights) | Canonical observation weights retained by a fitted model. |
| [EstimationSample](../reference/estimation.state.EstimationSample.llms.md#pyfixest.estimation.state.EstimationSample) | Summary of the sample used to fit a model. |
| [DroppedRowCounts](../reference/estimation.state.DroppedRowCounts.llms.md#pyfixest.estimation.state.DroppedRowCounts) | Counts of rows removed at each sample-filtering stage. |
| [FixedEffectCounts](../reference/estimation.state.FixedEffectCounts.llms.md#pyfixest.estimation.state.FixedEffectCounts) | Level counts of the absorbed fixed effects in the estimation sample. |
| [WithinLinearData](../reference/estimation.state.WithinLinearData.llms.md#pyfixest.estimation.state.WithinLinearData) | Response and regressors after demeaning by the fixed effects. |
| [WithinIvData](../reference/estimation.state.WithinIvData.llms.md#pyfixest.estimation.state.WithinIvData) | Response, regressors, and instruments after demeaning by the fixed effects. |
| [CollinearityCheck](../reference/estimation.state.CollinearityCheck.llms.md#pyfixest.estimation.state.CollinearityCheck) | Outcome of one rank check that removes collinear design columns. |
| [GlmWorkingState](../reference/estimation.state.GlmWorkingState.llms.md#pyfixest.estimation.state.GlmWorkingState) | Final GLM IRLS state in within scale. |
| [FitStatistics](../reference/estimation.state.FitStatistics.llms.md#pyfixest.estimation.state.FitStatistics) | Goodness-of-fit measures of a fitted model. |
| [FittedValues](../reference/estimation.state.FittedValues.llms.md#pyfixest.estimation.state.FittedValues) | In-sample predictions / fitted values on link and the response scale. |
| [SandwichComponents](../reference/estimation.state.SandwichComponents.llms.md#pyfixest.estimation.state.SandwichComponents) | Scores, Hessian, and bread of a fitted model’s sandwich covariance. |
| [VarianceCovariance](../reference/estimation.state.VarianceCovariance.llms.md#pyfixest.estimation.state.VarianceCovariance) | Variance-covariance estimate of the coefficients and its building blocks. |
| [WaldTest](../reference/estimation.state.WaldTest.llms.md#pyfixest.estimation.state.WaldTest) | Wald test of a linear hypothesis R @ beta = q. |
| [FirstStage](../reference/estimation.state.FirstStage.llms.md#pyfixest.estimation.state.FirstStage) | First-stage regression retained by a fitted 2SLS model. |
| [FirstStageDiagnostics](../reference/estimation.state.FirstStageDiagnostics.llms.md#pyfixest.estimation.state.FirstStageDiagnostics) | Instrument-strength diagnostics of a 2SLS first stage. |
| [QuantregSolution](../reference/estimation.state.QuantregSolution.llms.md#pyfixest.estimation.state.QuantregSolution) | Solution of the Frisch-Newton interior point solver. |
| [VcovSpec](../reference/estimation.state.VcovSpec.llms.md#pyfixest.estimation.state.VcovSpec) | The requested covariance estimator, parsed from the `vcov` argument. |
| [FixedEffectEstimates](../reference/estimation.state.FixedEffectEstimates.llms.md#pyfixest.estimation.state.FixedEffectEstimates) | Fixed-effect estimates recovered by `fixef()`. |
| [FixedEffect](../reference/estimation.state.FixedEffect.llms.md#pyfixest.estimation.state.FixedEffect) | Columnar coefficient data belonging to one fixed effect. |
| [RitestStatistics](../reference/estimation.state.RitestStatistics.llms.md#pyfixest.estimation.state.RitestStatistics) | Randomization-inference draws, the sample statistic, and the p-value. |
| [CoefficientTable](../reference/estimation.state.CoefficientTable.llms.md#pyfixest.estimation.state.CoefficientTable) | Coefficient estimates with their standard errors, statistics, and bounds. |
| [Capabilities](../reference/estimation.state.Capabilities.llms.md#pyfixest.estimation.state.Capabilities) | Inference and post-estimation features a fitted model supports. |

## Formula Parsing & Model Matrix

Internal APIs for formula parsing and model matrix construction.

|  |  |
|----|----|
| [Formula](../reference/estimation.formula.parse.Formula.llms.md#pyfixest.estimation.formula.parse.Formula) | A formulaic-compliant formula. |
| [create_model_matrix](../reference/estimation.formula.model_matrix.create_model_matrix.llms.md#pyfixest.estimation.formula.model_matrix.create_model_matrix) | Create a ModelMatrix from a formula and data. |
| [factor_interaction](../reference/estimation.formula.transforms.factor_interaction.factor_interaction.llms.md#pyfixest.estimation.formula.transforms.factor_interaction.factor_interaction) | Fixest-style i() operator for categorical encoding with interactions. |

## Demeaning

Fixed-effects demeaning: the `demean()` workhorse and the configurable backends. See the [Choosing a Demeaner Backend](../how-to/demeaner-backends.llms.md) guide for how to pick one.

|  |  |
|----|----|
| [demean](../reference/estimation.demean.llms.md#pyfixest.estimation.demean) | Demean an array. |
| [BaseDemeaner](../reference/demeaners.BaseDemeaner.llms.md#pyfixest.demeaners.BaseDemeaner) | Base configuration shared by all fixed-effects demeaners. |
| [MapDemeaner](../reference/demeaners.MapDemeaner.llms.md#pyfixest.demeaners.MapDemeaner) | Method of Alternating Projections (MAP) demeaner. |
| [LsmrDemeaner](../reference/demeaners.LsmrDemeaner.llms.md#pyfixest.demeaners.LsmrDemeaner) | Sparse LSMR demeaner. |
| [Preconditioner](../reference/core.demean.Preconditioner.llms.md#pyfixest.core.demean.Preconditioner) | A pre-built preconditioner; pass it back to skip the factorisation. |

## Misc / Utilities

Other PyFixest internals and utilities.

|  |  |
|----|----|
| [detect_singletons](../reference/core.detect_singletons.detect_singletons.llms.md#pyfixest.core.detect_singletons.detect_singletons) | Detect singleton fixed effects in a dataset. |
| [ssc](../reference/utils.utils.ssc.llms.md#pyfixest.utils.utils.ssc) | Set the small sample correction factor applied in `get_ssc()`. |
| [Ssc](../reference/utils.utils.Ssc.llms.md#pyfixest.utils.utils.Ssc) | Small-sample correction options. |
| [get_ssc](../reference/utils.utils.get_ssc.llms.md#pyfixest.utils.utils.get_ssc) | Compute the small sample adjustment factor and the degrees of freedom. |
| [DegreesOfFreedomCounts](../reference/utils.utils.DegreesOfFreedomCounts.llms.md#pyfixest.utils.utils.DegreesOfFreedomCounts) | Model counts that enter the small-sample correction. |
| [SmallSampleCorrection](../reference/utils.utils.SmallSampleCorrection.llms.md#pyfixest.utils.utils.SmallSampleCorrection) | Result of `get_ssc()`. |
| [optimal_mixture_precision](../reference/estimation.post_estimation.savi.optimal_mixture_precision.llms.md#pyfixest.estimation.post_estimation.savi.optimal_mixture_precision) | Compute the mixture precision that minimizes SAVI sequence width |
