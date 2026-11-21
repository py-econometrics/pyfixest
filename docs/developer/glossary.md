# pyfixest glossary

The words pyfixest uses for its econometric and estimation concepts in code,
docstrings, errors, and documentation. Terms are written out in prose; fixest
abbreviations such as `vcov` or `fixef` are API spellings and appear under
_Avoid_ as "(in prose)".

## Language

### Estimation

**Estimation function**:
A public function that fits one or more models from a formula and data, such
as `feols()`, `fepois()`, `feglm()`, `quantreg()`, or `event_study()`. One
estimation function can fit several estimators.
_Avoid_: method, estimation method, estimator (for the function)

**Estimator**:
The statistical procedure that defines a model's estimates, such as OLS, 2SLS,
Poisson, probit, quantile regression, TWFE, or DID2S.
_Avoid_: method, model type

**Solver**:
A numerical procedure that computes an estimator's solution, such as a linear
solve, the Frisch-Newton interior-point method, or a demeaner. Changing the
solver moves the estimates only within its tolerance.
_Avoid_: method, algorithm (as a separate concept)

**IWLS**:
Iteratively reweighted least squares, the solver that fits a GLM through
repeated weighted least-squares fits.
_Avoid_: IRLS, iterated weighted least squares

**Model**:
A formula and an estimator applied to one estimation sample: the thing being
estimated, whether or not it has been fitted. Its inference is not part of it.
_Avoid_: specification, regression (as synonyms)

**Fitted model**:
The object an estimation function returns for one model; "fit" is its short
form.
_Avoid_: model object, fitted result, estimation result, Fixest object, model
(for the object)

**Multiple estimation**:
One call to an estimation function that fits several models, through
stepwise formula operators, a sample split, or several quantiles.
_Avoid_: stepwise regression, multi-model fit

**Estimation options**:
The settings a fitted model was estimated with, other than its formula and
data.
_Avoid_: settings, config, arguments

### Formula

**Dependent variable**:
The left-hand-side variable of a model.
_Avoid_: depvar (in prose), response (for the variable), Y, target; outcome,
except in causal and difference-in-differences prose

**Regressor**:
A right-hand-side variable with an estimated coefficient. In an IV model it is
an exogenous regressor or an endogenous regressor.
_Avoid_: covariate, independent variable, explanatory variable, control, X

**Fixed effect**:
A term after `|` in a formula whose levels are absorbed instead of being
reported as coefficients. `C()` and `i()` terms are categorical regressors,
not fixed effects, even when they estimate the same intercepts.
_Avoid_: FE, fixef (in prose); fixed effect (for `C()` or `i()` terms)

**Level**:
One category of a fixed effect or a categorical regressor.
_Avoid_: group, cell

**Varying slope**:
A slope that differs by fixed-effect level, written `f1[x]`. A slope-only
effect, `f1[[x]]`, has the slopes without the level intercepts.
_Avoid_: random slope, interacted slope

**Categorical regressor**:
A categorical variable entered as regressors, one per level except a reference
level, written `C(f)` or `i(f)`. `i(f, x)` interacts those regressors with a
second variable.
_Avoid_: factor, dummies, fixed effects, interacted fixed effects

**Main equation**:
The equation of the dependent variable on the regressors that every model
has.
_Avoid_: second stage (outside IV)

**Design matrix**:
The regressors of the main equation, one column per estimated coefficient. In
an IV model it holds the endogenous regressors, not their first-stage fitted
values.
_Avoid_: X (in prose), model matrix, regressor matrix

### Instrumental variables

**Instrument block**:
The `[endogenous ~ instruments]` part of a formula.
_Avoid_: IV block, multistage block

**Instrument**:
A variable in the instrument block that is not a regressor of the main
equation; it enters the model only through the first stage. An exogenous
regressor repeated in the block is redundant, not an instrument.
_Avoid_: excluded instrument

**Instrument matrix**:
The instruments together with the exogenous regressors.
_Avoid_: instruments (for the matrix), Z

**First stage**:
In an IV model, the regression of an endogenous regressor on the exogenous
regressors and the instruments.
_Avoid_: first stage (for DID2S, unqualified)

**Second stage**:
In an IV model, the main equation with the endogenous regressors replaced by
their first-stage fitted values.

### Demeaning

**Demeaning**:
Projecting a variable on the fixed effects, including their varying slopes,
and keeping the residual.
_Avoid_: residualizing, partialling out, projecting out, within transformation

**Absorb**:
What a fixed effect does to its levels, and to the intercept when it has level
intercepts: they are accounted for by demeaning and not reported as
coefficients.

**Within data**:
The dependent variable and regressors after demeaning.
_Avoid_: demeaned data, transformed data

**Demeaner**:
The solver that performs demeaning, such as the method of alternating
projections (MAP) or LSMR.
_Avoid_: backend (alone), demeaning algorithm

**Demeaner backend**:
The library or kernel that implements a demeaner, such as Rust or numba for
MAP, and within or PyTorch for LSMR.
_Avoid_: backend (alone), engine

**Preconditioner**:
A transformation an iterative demeaner such as LSMR applies to the
fixed-effect system so that it converges in fewer iterations, such as the
diagonal or the additive Schwarz preconditioner. Like any solver choice, it
moves the estimates only within the tolerance.
_Avoid_: preconditioning method

### Estimation sample

**Estimation sample**:
The observations a model is fitted on, after sample splitting and every
dropping stage.
_Avoid_: sample (unqualified), estimation data

**Dropped row**:
An input row excluded from the estimation sample for a missing or infinite
value, a zero weight, a singleton, or separation.
_Avoid_: NA row, removed observation

**Singleton**:
A level of a fixed effect with level intercepts that occurs in exactly one
row. Its row is dropped by default.
_Avoid_: singleton observation, singleton group, singleton fixed effect (for
the row)

**Separation**:
The condition of a GLM observation whose dependent variable sits at a bound
that the mean reaches only as the linear predictor diverges, such as zero in a
Poisson model. Separated rows are dropped.
_Avoid_: perfect fit, perfect prediction

**Separation check**:
A procedure that detects separated observations: the fixed-effect check or the
iterative rectifier.
_Avoid_: separation method

**Sample split**:
Fitting the same formula separately on the subsets defined by the values of
one variable; each subset gives its own model. It is a form of multiple
estimation.
_Avoid_: subsample, by-group estimation

**Observation**:
What the number of observations counts. A row counts as one observation,
except under frequency weights, where it counts as many as its weight.
_Avoid_: N (unqualified), effective sample size

### Weights

**Observation weights**:
User-supplied per-row weights with one interpretation, analytic or frequency.
_Avoid_: weights (unqualified), regression weights, sample weights, WLS
weights

**Analytic weights**:
Observation weights that scale a row's contribution without changing the
number of observations. Written `aweights`.

**Frequency weights**:
Observation weights under which a row stands for as many identical
observations as its weight. Written `fweights`.

**Working weights**:
The weights of the final IWLS iteration. They already contain the observation
weights.
_Avoid_: weights (unqualified)

**Bootstrap weights**:
Random multipliers that the wild bootstrap draws per observation or per
cluster, such as Rademacher or Webb weights.
_Avoid_: weights type (in prose), weights (unqualified)

### Inference

**Inference**:
Standard errors, tests, p-values, and confidence intervals for a fitted
model's coefficients.
_Avoid_: standard errors (as the umbrella)

**Variance-covariance matrix**:
The estimated sampling covariance of a fitted model's coefficients under one
variance estimator.
_Avoid_: vcov (in prose), covariance (alone)

**Variance estimator**:
The procedure that estimates the variance-covariance matrix, such as iid,
heteroskedasticity-robust (HC1 to HC3), cluster-robust (CRV1, CRV3), HAC, or
nid. Changing it changes the inference, not the estimates.
_Avoid_: vcov type, covariance estimator, standard-error type, inference type,
estimator (alone)

**Inference type**:
Whether intervals and tests hold for each coefficient on its own, for several
coefficients simultaneously, or at any time under continuous monitoring
(anytime-valid).
_Avoid_: interval type

**Significance level**:
The error probability α of a test. Confidence intervals at significance level
α cover 1 − α.
_Avoid_: alpha (in prose), level, confidence level (for α)

**Small-sample correction**:
The finite-sample scaling of the variance-covariance matrix and the degrees of
freedom of its tests.
_Avoid_: ssc (in prose), dof adjustment, small sample adjustment

**Adjustment factor**:
The number by which the small-sample correction multiplies the
variance-covariance matrix.
_Avoid_: ssc (for the number)

**Cluster variable**:
A data column whose values define the clusters of cluster-robust inference;
several cluster variables give multiway clustering.
_Avoid_: clustervar (in prose), cluster (for the column), group

**Number of clusters**:
The count of distinct values of a cluster variable. For non-clustered variance
estimators, the small-sample correction counts each observation
(heteroskedasticity-robust) or each period (HAC) as a cluster.
_Avoid_: G (in prose)

**Coefficient table**:
A fitted model's estimates with their standard errors, test statistics,
p-values, and confidence intervals under its current variance-covariance
matrix.
_Avoid_: coeftable (in prose), results table, summary table

**Wald test**:
A test of linear restrictions Rβ = q on the coefficients, built from the
variance-covariance matrix, whether its statistic is reported on the χ² or the
F scale.
_Avoid_: F test (as a synonym)

**Wild bootstrap**:
Resampling inference that perturbs residuals with bootstrap weights drawn per
observation or per cluster.
_Avoid_: bootstrap (unqualified)

**Randomization inference**:
Inference that compares a coefficient's test statistic with its distribution
over random reassignments of a treatment variable.
_Avoid_: permutation test, ritest (in prose)

**Multiple testing**:
Testing several hypotheses at once, with p-values adjusted to control the
family-wise error rate, such as by Bonferroni, Romano-Wolf, or Westfall-Young.
_Avoid_: multiple hypothesis testing, multiple hypothesis corrections,
multiple comparisons

**Simultaneous confidence intervals**:
Confidence intervals for several coefficients that cover all of them jointly
with probability 1 − α, unlike pointwise intervals, which each cover one
coefficient.
_Avoid_: joint confidence intervals, confidence bands, joint confidence bands

**Causal cluster variance**:
A design-based variance estimator for treatment effects that accounts for the
share of clusters sampled and how treatment varies within clusters.
_Avoid_: CCV (in prose)

### Post-estimation

**Link scale**:
The scale of the linear predictor.

**Response scale**:
The scale of the dependent variable's mean. It coincides with the link scale
for linear models.

**Fitted values**:
A fitted model's in-sample predictions, on the link or the response scale.
_Avoid_: yhat, predictions (for in-sample values)

**Residual**:
The dependent variable minus the fitted mean, in the dependent variable's
units. Unqualified, it never means the working residual.
_Avoid_: response residual, raw residual, u_hat (in prose)

**Working residual**:
The residual of the weighted regression in the final IWLS iteration.

**Fit statistics**:
Goodness-of-fit measures of a fitted model, such as R², deviance, or
log-likelihood.
_Avoid_: fitstat (in prose), GOF, model stats

**Fixed-effect estimates**:
The estimated level intercepts and varying slopes of a model's fixed effects,
recovered after estimation.
_Avoid_: fixed effects (for the estimates), alpha, fixef (in prose)

**Refit**:
Rerunning a fitted model's estimator with its estimation options on other
data, or on its own estimation sample with changed columns.
_Avoid_: update, re-estimate

**Sherman-Morrison update**:
Updating an OLS fit's coefficients for appended observations without
refitting.
_Avoid_: update (in prose), online update

### Difference-in-differences

**Unit**:
The panel entity observed across periods.
_Avoid_: id, individual, idname (in prose)

**Period**:
A calendar time point of the panel.
_Avoid_: time (alone), tname (in prose)

**Cohort**:
The units whose treatment starts in the same period.
_Avoid_: group, treatment group, gname (in prose)

**Event time**:
The number of periods between a period and a unit's first treated period.
_Avoid_: relative time, relative period

**ATT**:
The average treatment effect on the treated.
