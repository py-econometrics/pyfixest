# PyFixest Function Reference

## Estimation Functions

User facing estimation functions

|  |  |
|----|----|
| [estimation.api.feols.feols](../reference/estimation.api.feols.feols.llms.md#pyfixest.estimation.api.feols.feols) | Estimate a linear regression model with fixed effects using fixest formula syntax. |
| [estimation.api.fepois.fepois](../reference/estimation.api.fepois.fepois.llms.md#pyfixest.estimation.api.fepois.fepois) | Estimate Poisson regression model with fixed effects using the `ppmlhdfe` algorithm. |
| [estimation.api.feglm.feglm](../reference/estimation.api.feglm.feglm.llms.md#pyfixest.estimation.api.feglm.feglm) | Estimate GLM regression models with fixed effects. |
| [estimation.api.quantreg.quantreg](../reference/estimation.api.quantreg.quantreg.llms.md#pyfixest.estimation.api.quantreg.quantreg) | Fit a quantile regression model using the interior point algorithm from Portnoy and Koenker (1997). |
| [estimation.post_estimation.savi.optimal_mixture_precision](../reference/estimation.post_estimation.savi.optimal_mixture_precision.llms.md#pyfixest.estimation.post_estimation.savi.optimal_mixture_precision) | Compute the mixture precision that minimizes SAVI sequence width |
| [did.estimation.did2s](../reference/did.estimation.did2s.llms.md#pyfixest.did.estimation.did2s) | Estimate a Difference-in-Differences model using Gardner’s two-step DID2S estimator. |
| [did.estimation.lpdid](../reference/did.estimation.lpdid.llms.md#pyfixest.did.estimation.lpdid) | Local projections approach to estimation. |
| [did.estimation.event_study](../reference/did.estimation.event_study.llms.md#pyfixest.did.estimation.event_study) | Estimate Event Study Model. |
| [estimation.post_estimation.multcomp.bonferroni](../reference/estimation.post_estimation.multcomp.bonferroni.llms.md#pyfixest.estimation.post_estimation.multcomp.bonferroni) | Compute Bonferroni adjusted p-values for multiple hypothesis testing. |
| [estimation.post_estimation.multcomp.rwolf](../reference/estimation.post_estimation.multcomp.rwolf.llms.md#pyfixest.estimation.post_estimation.multcomp.rwolf) | Compute Romano-Wolf adjusted p-values for multiple hypothesis testing. |
| [estimation.post_estimation.multcomp.wyoung](../reference/estimation.post_estimation.multcomp.wyoung.llms.md#pyfixest.estimation.post_estimation.multcomp.wyoung) | Compute the Westfall-Young adjusted p-values for multiple hypothesis testing. |

## Estimation Classes

Details on Methods and Attributes

|  |  |
|----|----|
| [demeaners.BaseDemeaner](../reference/demeaners.BaseDemeaner.llms.md#pyfixest.demeaners.BaseDemeaner) | Base configuration shared by all fixed-effects demeaners. |
| [demeaners.MapDemeaner](../reference/demeaners.MapDemeaner.llms.md#pyfixest.demeaners.MapDemeaner) | Method of Alternating Projections (MAP) demeaner. |
| [demeaners.LsmrDemeaner](../reference/demeaners.LsmrDemeaner.llms.md#pyfixest.demeaners.LsmrDemeaner) | Sparse LSMR demeaner. |
| [estimation.models.feols\_.Feols](../reference/estimation.models.feols_.Feols.llms.md#pyfixest.estimation.models.feols_.Feols) | Non user-facing class to estimate a linear regression via OLS. |
| [estimation.models.fepois\_.Fepois](../reference/estimation.models.fepois_.Fepois.llms.md#pyfixest.estimation.models.fepois_.Fepois) | Estimate a Poisson regression model. |
| [estimation.models.feiv\_.Feiv](../reference/estimation.models.feiv_.Feiv.llms.md#pyfixest.estimation.models.feiv_.Feiv) | Non user-facing class to estimate an IV model using a 2SLS estimator. |
| [estimation.models.feglm\_.Feglm](../reference/estimation.models.feglm_.Feglm.llms.md#pyfixest.estimation.models.feglm_.Feglm) | Base class for the estimation of a fixed-effects GLM model. |
| [estimation.models.felogit\_.Felogit](../reference/estimation.models.felogit_.Felogit.llms.md#pyfixest.estimation.models.felogit_.Felogit) | Class for the estimation of a fixed-effects logit model. |
| [estimation.models.feprobit\_.Feprobit](../reference/estimation.models.feprobit_.Feprobit.llms.md#pyfixest.estimation.models.feprobit_.Feprobit) | Class for the estimation of a fixed-effects probit model. |
| [estimation.models.fegaussian\_.Fegaussian](../reference/estimation.models.fegaussian_.Fegaussian.llms.md#pyfixest.estimation.models.fegaussian_.Fegaussian) | Class for the estimation of a fixed-effects GLM with normal errors. |
| [estimation.FixestMulti\_.FixestMulti](../reference/estimation.FixestMulti_.FixestMulti.llms.md#pyfixest.estimation.FixestMulti_.FixestMulti) | Results container holding every model fitted by one public-API call. |
| [estimation.quantreg.quantreg\_.Quantreg](../reference/estimation.quantreg.quantreg_.Quantreg.llms.md#pyfixest.estimation.quantreg.quantreg_.Quantreg) | Quantile regression model. |

## Summarize and Visualize

Post-Processing of Estimation Results

|  |  |
|----|----|
| [did.visualize.panelview](../reference/did.visualize.panelview.llms.md#pyfixest.did.visualize.panelview) | Generate a panel view of the treatment variable over time for each unit. |
| [report.summary](../reference/report.summary.llms.md#pyfixest.report.summary) | Print a summary of estimation results for each estimated model. |
| [report.etable](../reference/report.etable.llms.md#pyfixest.report.etable) | Generate a table summarizing the results of multiple regression models. |
| [report.coefplot](../reference/report.coefplot.llms.md#pyfixest.report.coefplot) | Plot model coefficients with confidence intervals. |
| [report.iplot](../reference/report.iplot.llms.md#pyfixest.report.iplot) | Plot model coefficients for variables interacted via “i()” syntax, with |

## Formula Parsing & Model Matrix

Internal APIs for formula parsing and model matrix construction

|  |  |
|----|----|
| [estimation.formula.parse.Formula](../reference/estimation.formula.parse.Formula.llms.md#pyfixest.estimation.formula.parse.Formula) | A formulaic-compliant formula. |
| [estimation.formula.model_matrix.ModelMatrix](../reference/estimation.formula.model_matrix.ModelMatrix.llms.md#pyfixest.estimation.formula.model_matrix.ModelMatrix) | A wrapper around formulaic.ModelMatrix for the specification of PyFixest models. |
| [estimation.formula.factor_interaction.factor_interaction](../reference/estimation.formula.factor_interaction.factor_interaction.llms.md#pyfixest.estimation.formula.factor_interaction.factor_interaction) | Fixest-style i() operator for categorical encoding with interactions. |

## Misc / Utilities

PyFixest internals and utilities

|  |  |
|----|----|
| [estimation.demean](../reference/estimation.demean.llms.md#pyfixest.estimation.demean) | Demean an array. |
| [core.detect_singletons.detect_singletons](../reference/core.detect_singletons.detect_singletons.llms.md#pyfixest.core.detect_singletons.detect_singletons) | Detect singleton fixed effects in a dataset. |
| [utils.utils.ssc](../reference/utils.utils.ssc.llms.md#pyfixest.utils.utils.ssc) | Set the small sample correction factor applied in `get_ssc()`. |
| [utils.utils.get_ssc](../reference/utils.utils.get_ssc.llms.md#pyfixest.utils.utils.get_ssc) | Compute small sample adjustment factors. |
| [estimation.deprecated.model_matrix_fixest\_.model_matrix_fixest](../reference/estimation.deprecated.model_matrix_fixest_.model_matrix_fixest.llms.md#pyfixest.estimation.deprecated.model_matrix_fixest_.model_matrix_fixest) | Create model matrices for fixed effects estimation. |
