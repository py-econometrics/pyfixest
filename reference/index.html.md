# PyFixest Function Reference {.doc .doc-index}

## Estimation Functions

User facing estimation functions


| | |
| --- | --- |
| [estimation.api.feols.feols](estimation.api.feols.feols.qmd#pyfixest.estimation.api.feols.feols) | Estimate a linear regression models with fixed effects using fixest formula syntax. |
| [estimation.api.fepois.fepois](estimation.api.fepois.fepois.qmd#pyfixest.estimation.api.fepois.fepois) | Estimate Poisson regression model with fixed effects using the `ppmlhdfe` algorithm. |
| [estimation.api.feglm.feglm](estimation.api.feglm.feglm.qmd#pyfixest.estimation.api.feglm.feglm) | Estimate GLM regression models with fixed effects. |
| [estimation.api.quantreg.quantreg](estimation.api.quantreg.quantreg.qmd#pyfixest.estimation.api.quantreg.quantreg) | Fit a quantile regression model using the interior point algorithm from Portnoy and Koenker (1997). |
| [did.estimation.did2s](did.estimation.did2s.qmd#pyfixest.did.estimation.did2s) | Estimate a Difference-in-Differences model using Gardner's two-step DID2S estimator. |
| [did.estimation.lpdid](did.estimation.lpdid.qmd#pyfixest.did.estimation.lpdid) | Local projections approach to estimation. |
| [did.estimation.event_study](did.estimation.event_study.qmd#pyfixest.did.estimation.event_study) | Estimate Event Study Model. |
| [estimation.post_estimation.multcomp.bonferroni](estimation.post_estimation.multcomp.bonferroni.qmd#pyfixest.estimation.post_estimation.multcomp.bonferroni) | Compute Bonferroni adjusted p-values for multiple hypothesis testing. |
| [estimation.post_estimation.multcomp.rwolf](estimation.post_estimation.multcomp.rwolf.qmd#pyfixest.estimation.post_estimation.multcomp.rwolf) | Compute Romano-Wolf adjusted p-values for multiple hypothesis testing. |

## Estimation Classes

Details on Methods and Attributes


| | |
| --- | --- |
| [estimation.models.feols_.Feols](estimation.models.feols_.Feols.qmd#pyfixest.estimation.models.feols_.Feols) | Non user-facing class to estimate a linear regression via OLS. |
| [estimation.models.fepois_.Fepois](estimation.models.fepois_.Fepois.qmd#pyfixest.estimation.models.fepois_.Fepois) | Estimate a Poisson regression model. |
| [estimation.models.feiv_.Feiv](estimation.models.feiv_.Feiv.qmd#pyfixest.estimation.models.feiv_.Feiv) | Non user-facing class to estimate an IV model using a 2SLS estimator. |
| [estimation.models.feglm_.Feglm](estimation.models.feglm_.Feglm.qmd#pyfixest.estimation.models.feglm_.Feglm) | Abstract base class for the estimation of a fixed-effects GLM model. |
| [estimation.models.felogit_.Felogit](estimation.models.felogit_.Felogit.qmd#pyfixest.estimation.models.felogit_.Felogit) | Class for the estimation of a fixed-effects logit model. |
| [estimation.models.feprobit_.Feprobit](estimation.models.feprobit_.Feprobit.qmd#pyfixest.estimation.models.feprobit_.Feprobit) | Class for the estimation of a fixed-effects probit model. |
| [estimation.models.fegaussian_.Fegaussian](estimation.models.fegaussian_.Fegaussian.qmd#pyfixest.estimation.models.fegaussian_.Fegaussian) | Class for the estimation of a fixed-effects GLM with normal errors. |
| [estimation.models.feols_compressed_.FeolsCompressed](estimation.models.feols_compressed_.FeolsCompressed.qmd#pyfixest.estimation.models.feols_compressed_.FeolsCompressed) | Non-user-facing class for compressed regression with fixed effects. |
| [estimation.FixestMulti_.FixestMulti](estimation.FixestMulti_.FixestMulti.qmd#pyfixest.estimation.FixestMulti_.FixestMulti) | A class to estimate multiple regression models with fixed effects. |
| [estimation.quantreg.quantreg_.Quantreg](estimation.quantreg.quantreg_.Quantreg.qmd#pyfixest.estimation.quantreg.quantreg_.Quantreg) | Quantile regression model. |

## Summarize and Visualize

Post-Processing of Estimation Results


| | |
| --- | --- |
| [did.visualize.panelview](did.visualize.panelview.qmd#pyfixest.did.visualize.panelview) | Generate a panel view of the treatment variable over time for each unit. |
| [report.summary](report.summary.qmd#pyfixest.report.summary) | Print a summary of estimation results for each estimated model. |
| [report.etable](report.etable.qmd#pyfixest.report.etable) | Generate a table summarizing the results of multiple regression models. |
| [report.dtable](report.dtable.qmd#pyfixest.report.dtable) | Generate descriptive statistics tables and create a booktab style table in |
| [report.coefplot](report.coefplot.qmd#pyfixest.report.coefplot) | Plot model coefficients with confidence intervals. |
| [report.iplot](report.iplot.qmd#pyfixest.report.iplot) | Plot model coefficients for variables interacted via "i()" syntax, with |
| [did.visualize.panelview](did.visualize.panelview.qmd#pyfixest.did.visualize.panelview) | Generate a panel view of the treatment variable over time for each unit. |

## Formula Parsing & Model Matrix

Internal APIs for formula parsing and model matrix construction


| | |
| --- | --- |
| [estimation.formula.parse.Formula](estimation.formula.parse.Formula.qmd#pyfixest.estimation.formula.parse.Formula) | A formulaic-compliant formula. |
| [estimation.formula.model_matrix.ModelMatrix](estimation.formula.model_matrix.ModelMatrix.qmd#pyfixest.estimation.formula.model_matrix.ModelMatrix) | A wrapper around formulaic.ModelMatrix for the specification of PyFixest models. |
| [estimation.formula.factor_interaction.factor_interaction](estimation.formula.factor_interaction.factor_interaction.qmd#pyfixest.estimation.formula.factor_interaction.factor_interaction) | Fixest-style i() operator for categorical encoding with interactions. |

## Misc / Utilities

PyFixest internals and utilities


| | |
| --- | --- |
| [estimation.internals.demean_.demean](estimation.internals.demean_.demean.qmd#pyfixest.estimation.internals.demean_.demean) | Demean an array. |
| [estimation.internals.detect_singletons_.detect_singletons](estimation.internals.detect_singletons_.detect_singletons.qmd#pyfixest.estimation.internals.detect_singletons_.detect_singletons) | Detect singleton fixed effects in a dataset. |
| [estimation.deprecated.model_matrix_fixest_.model_matrix_fixest](estimation.deprecated.model_matrix_fixest_.model_matrix_fixest.qmd#pyfixest.estimation.deprecated.model_matrix_fixest_.model_matrix_fixest) | Create model matrices for fixed effects estimation. |