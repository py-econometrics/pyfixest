<!-- Generated from docs/reference/index.qmd; do not edit. -->

# PyFixest Function Reference

## Estimation Functions

User facing estimation functions

| | |
| --- | --- |
| [estimation.api.feols.feols](estimation.api.feols.feols.md#pyfixest.estimation.api.feols.feols) | Estimate a linear regression model with fixed effects using fixest formula syntax. |
| [estimation.api.fepois.fepois](estimation.api.fepois.fepois.md#pyfixest.estimation.api.fepois.fepois) | Estimate Poisson regression model with fixed effects using the `ppmlhdfe` algorithm. |
| [estimation.api.feglm.feglm](estimation.api.feglm.feglm.md#pyfixest.estimation.api.feglm.feglm) | Estimate GLM regression models with fixed effects. |
| [estimation.api.quantreg.quantreg](estimation.api.quantreg.quantreg.md#pyfixest.estimation.api.quantreg.quantreg) | Fit a quantile regression model using the interior point algorithm from Portnoy and Koenker (1997). |
| [did.estimation.did2s](did.estimation.did2s.md#pyfixest.did.estimation.did2s) | Estimate a Difference-in-Differences model using Gardner's two-step DID2S estimator. |
| [did.estimation.lpdid](did.estimation.lpdid.md#pyfixest.did.estimation.lpdid) | Local projections approach to estimation. |
| [did.estimation.event_study](did.estimation.event_study.md#pyfixest.did.estimation.event_study) | Estimate Event Study Model. |
| [estimation.post_estimation.multcomp.bonferroni](estimation.post_estimation.multcomp.bonferroni.md#pyfixest.estimation.post_estimation.multcomp.bonferroni) | Compute Bonferroni adjusted p-values for multiple hypothesis testing. |
| [estimation.post_estimation.multcomp.rwolf](estimation.post_estimation.multcomp.rwolf.md#pyfixest.estimation.post_estimation.multcomp.rwolf) | Compute Romano-Wolf adjusted p-values for multiple hypothesis testing. |
| [estimation.post_estimation.multcomp.wyoung](estimation.post_estimation.multcomp.wyoung.md#pyfixest.estimation.post_estimation.multcomp.wyoung) | Compute the Westfall-Young adjusted p-values for multiple hypothesis testing. |

## Estimation Classes

Details on Methods and Attributes

| | |
| --- | --- |
| [demeaners.BaseDemeaner](demeaners.BaseDemeaner.md#pyfixest.demeaners.BaseDemeaner) | Base configuration shared by all fixed-effects demeaners. |
| [demeaners.MapDemeaner](demeaners.MapDemeaner.md#pyfixest.demeaners.MapDemeaner) | Method of Alternating Projections (MAP) demeaner. |
| [demeaners.LsmrDemeaner](demeaners.LsmrDemeaner.md#pyfixest.demeaners.LsmrDemeaner) | Sparse LSMR demeaner. |
| [core.demean.Preconditioner](core.demean.Preconditioner.md#pyfixest.core.demean.Preconditioner) | Opaque handle to a pre-built within preconditioner (Additive Schwarz or |
| [did.saturated_twfe.SaturatedEventStudy](did.saturated_twfe.SaturatedEventStudy.md#pyfixest.did.saturated_twfe.SaturatedEventStudy) | Saturated event study with cohort-specific effect curves. |
| [estimation.models.feols_.Feols](estimation.models.feols_.Feols.md#pyfixest.estimation.models.feols_.Feols) | Fitted OLS or weighted least-squares model. |
| [estimation.models.fepois_.Fepois](estimation.models.fepois_.Fepois.md#pyfixest.estimation.models.fepois_.Fepois) | Estimate a Poisson regression model. |
| [estimation.models.feiv_.Feiv](estimation.models.feiv_.Feiv.md#pyfixest.estimation.models.feiv_.Feiv) | Fitted instrumental-variable model estimated by two-stage least squares. |
| [estimation.models.feglm_.Feglm](estimation.models.feglm_.Feglm.md#pyfixest.estimation.models.feglm_.Feglm) | Base result class for fitted fixed-effects generalized linear models. |
| [estimation.models.felogit_.Felogit](estimation.models.felogit_.Felogit.md#pyfixest.estimation.models.felogit_.Felogit) | Class for the estimation of a fixed-effects logit model. |
| [estimation.models.feprobit_.Feprobit](estimation.models.feprobit_.Feprobit.md#pyfixest.estimation.models.feprobit_.Feprobit) | Class for the estimation of a fixed-effects probit model. |
| [estimation.models.fegaussian_.Fegaussian](estimation.models.fegaussian_.Fegaussian.md#pyfixest.estimation.models.fegaussian_.Fegaussian) | Class for the estimation of a fixed-effects GLM with normal errors. |
| [estimation.FixestMulti_.FixestMulti](estimation.FixestMulti_.FixestMulti.md#pyfixest.estimation.FixestMulti_.FixestMulti) | Container for models produced by one multiple-estimation call. |
| [estimation.quantreg.quantreg_.Quantreg](estimation.quantreg.quantreg_.Quantreg.md#pyfixest.estimation.quantreg.quantreg_.Quantreg) | Quantile regression model. |

## Summarize and Visualize

Post-Processing of Estimation Results

| | |
| --- | --- |
| [did.visualize.panelview](did.visualize.panelview.md#pyfixest.did.visualize.panelview) | Generate a panel view of the treatment variable over time for each unit. |
| [report.summary](report.summary.md#pyfixest.report.summary) | Print a summary of estimation results for each estimated model. |
| [report.etable](report.etable.md#pyfixest.report.etable) | Generate a table summarizing the results of multiple regression models. |
| [report.dtable](report.dtable.md#pyfixest.report.dtable) | Generate descriptive statistics tables and create a booktab style table in |
| [report.coefplot](report.coefplot.md#pyfixest.report.coefplot) | Plot model coefficients with confidence intervals. |
| [report.iplot](report.iplot.md#pyfixest.report.iplot) | Plot model coefficients for variables interacted via "i()" syntax, with |
| [report.qplot](report.qplot.md#pyfixest.report.qplot) | Plot regression quantiles. |

## Formula Parsing & Model Matrix

Internal APIs for formula parsing and model matrix construction

| | |
| --- | --- |
| [estimation.formula.parse.Formula](estimation.formula.parse.Formula.md#pyfixest.estimation.formula.parse.Formula) | A formulaic-compliant formula. |
| [estimation.formula.model_matrix.ModelMatrix](estimation.formula.model_matrix.ModelMatrix.md#pyfixest.estimation.formula.model_matrix.ModelMatrix) | A wrapper around formulaic.ModelMatrix for the specification of PyFixest models. |
| [estimation.formula.factor_interaction.factor_interaction](estimation.formula.factor_interaction.factor_interaction.md#pyfixest.estimation.formula.factor_interaction.factor_interaction) | Fixest-style i() operator for categorical encoding with interactions. |

## Misc / Utilities

PyFixest internals and utilities

| | |
| --- | --- |
| [estimation.demean](estimation.demean.md#pyfixest.estimation.demean) | Demean an array. |
| [core.detect_singletons.detect_singletons](core.detect_singletons.detect_singletons.md#pyfixest.core.detect_singletons.detect_singletons) | Detect singleton fixed effects in a dataset. |
| [utils.utils.ssc](utils.utils.ssc.md#pyfixest.utils.utils.ssc) | Set the small sample correction factor applied in `get_ssc()`. |
| [utils.utils.get_ssc](utils.utils.get_ssc.md#pyfixest.utils.utils.get_ssc) | Compute small sample adjustment factors. |
| [utils.utils.get_data](utils.utils.get_data.md#pyfixest.utils.utils.get_data) | Create a random example data set. |
| [utils.dgps.get_bartik_data](utils.dgps.get_bartik_data.md#pyfixest.utils.dgps.get_bartik_data) | Synthetic data for a Bartik (shift-share) IV application on immigration and wages. |
| [utils.dgps.get_encouragement_data](utils.dgps.get_encouragement_data.md#pyfixest.utils.dgps.get_encouragement_data) | Synthetic data for an A/B encouragement design IV application. |
| [utils.dgps.get_ivf_data](utils.dgps.get_ivf_data.md#pyfixest.utils.dgps.get_ivf_data) | Synthetic data for the motherhood penalty IV application (IVF instrument). |
| [utils.dgps.get_motherhood_event_study_data](utils.dgps.get_motherhood_event_study_data.md#pyfixest.utils.dgps.get_motherhood_event_study_data) | Generate a fertility-timing panel for motherhood-penalty event studies. |
| [utils.dgps.get_twin_data](utils.dgps.get_twin_data.md#pyfixest.utils.dgps.get_twin_data) | Generate twin study data for returns to education. |
| [utils.dgps.get_worker_panel](utils.dgps.get_worker_panel.md#pyfixest.utils.dgps.get_worker_panel) | Generate a worker-firm panel dataset with two-way fixed effects. |
| [estimation.deprecated.model_matrix_fixest_.model_matrix_fixest](estimation.deprecated.model_matrix_fixest_.model_matrix_fixest.md#pyfixest.estimation.deprecated.model_matrix_fixest_.model_matrix_fixest) | Create model matrices for fixed effects estimation. |
