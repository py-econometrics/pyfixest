---
title: 'PyFixest: Fast High-Dimensional Fixed-Effects Regression in Python'
tags:
  - Python
  - econometrics
  - fixed effects
  - panel data
  - causal inference
  - regression
authors:
  - name: Alexander Fischer
    corresponding: true
    affiliation: '1'
  - name: Leonard Stimpfle
    orcid: 0009-0000-7033-7010
    affiliation: '2'
  - name: Kristof Schröder
    affiliation: '3'
  - name: Dirk Sliwka
    orcid: 0000-0002-8026-0165
    affiliation: '4'
  - name: Juan Camilo Orduz
    affiliation: '5'
  - name: Wenzhi Ding
    orcid: 0000-0002-4784-8848
    affiliation: '6'
affiliations:
  - index: 1
    name: trivago N.V., Germany
  - index: 2
    name: Ghent University, Belgium
  - index: 3
    name: appliedAI Institute for Europe gGmbH, Germany
  - index: 4
    name: University of Cologne, Germany
  - index: 5
    name: PyMC Labs, Germany
  - index: 6
    name: The Hong Kong Polytechnic University, Hong Kong SAR
date: 31 August 2026
bibliography: paper.bib
---

# Summary

`PyFixest` estimates relationships in large or grouped datasets while controlling
for unobserved differences across many workers, firms, places, time periods, or
other categories. It avoids constructing the enormous matrices of indicator
variables that would make conventional regression routines slow or infeasible.
The package provides linear, instrumental-variable, generalized-linear, and
quantile regression together with inference, reporting, and causal-analysis
tools in Python.

The user-facing foundation of `PyFixest` is the R package `fixest`
[@berge2026fixest]. `PyFixest` intentionally adopts `fixest`'s estimator names,
fixed-effects formula layout, multiple-estimation operators, variance-covariance
and small-sample-correction interfaces, singleton handling, collinearity
conventions, and many defaults wherever the corresponding feature is supported.
These are contributions of the `fixest` authors, not inventions of `PyFixest`.
Where the interfaces deliberately differ---including Python formula strings,
dictionary variance specifications, and default variance selection---the
documentation makes those differences explicit. We ask users to consider citing
`fixest` alongside this paper.

`PyFixest` is an independent Python and Rust implementation. Reference tests use
`rpy2` [@gautier2008rpy2] to compare its coefficients, inference, and fit
statistics with `fixest`; tolerances are chosen per estimator, with core OLS
comparisons typically using relative and absolute tolerances of $10^{-8}$.

# Statement of need

High-dimensional fixed effects are ubiquitous in applied economics and related
fields. Researchers may need to control simultaneously for hundreds of thousands
of workers, firms, products, locations, or time periods. Directly encoding every
category creates a design matrix that can exhaust memory and makes generic
solvers unnecessarily expensive. Applied researchers need estimators that absorb
these effects efficiently while preserving familiar formulas, robust inference,
and reproducible defaults.

The need extends beyond linear regression. Generalized linear models are commonly
fit by iteratively reweighted least squares, where changing working weights require
the fixed effects to be absorbed again at every iteration
[@correia2020fast; @stammann2017fast]. A one-time preprocessing step therefore
does not provide a general solution.

# State of the field

R and Stata have long offered specialized packages: `lfe` [@gaure2013lfe],
`fixest` [@berge2026fixest], `reghdfe` [@correia2023reghdfe], and `alpaca`
[@stammann2020package]. Julia provides `FixedEffectModels.jl`
[@fixedeffectmodelsjl_2025]. In Python, `statsmodels`
[@seabold2010statsmodels] does not provide a native multi-way high-dimensional
fixed-effects estimator. `pyhdfe` [@gortmaker_pyhdfe_2023] provides absorption
algorithms, and `linearmodels` [@linearmodels] exposes them through `AbsorbingLS`,
but these routes focus on linear models rather than an integrated estimation and
post-estimation workflow.

A separate package is warranted because `PyFixest` provides a Python-native
counterpart to an established R workflow: cross-language behavioral compatibility,
several estimator families, multiple estimation, and common reporting and
inference tools behind one interface. It builds on focused Python infrastructure
rather than reimplementing it. In particular, Formulaic
[@wardrop_formulaic_2026] supplies the extensible Wilkinson parser and model-matrix
machinery. `PyFixest` layers its `fixest`-inspired fixed-effect and
multiple-estimation grammar, formula expansion, and estimation planning on top;
Formulaic does not itself implement those `fixest` operators.

# Software design

## Absorbing fixed effects

Consider $y = X\beta + D\alpha + \varepsilon$, where $X$ contains the regressors
of interest and $D$ encodes the fixed effects. The Frisch-Waugh-Lovell theorem
[@frisch1933; @lovell1963] permits residualizing $y$ and every column of $X$
against $D$, then regressing the residualized outcome on the residualized
regressors. The resulting $\hat\beta$ equals the coefficient from the full
regression without requiring the full indicator matrix in the final solve.

The default Rust backend uses the method of alternating projections (MAP), or
iterative demeaning [@guimaraes2010; @gaure2013ols]. It repeatedly subtracts
weighted group means for each fixed effect. MAP has low setup costs and performs
well for dense, well-connected designs. On sparse worker-firm, patient-doctor, or
trade networks, information may propagate slowly between groups. For these
designs, `LsmrDemeaner` uses LSMR [@fong2011] and the `within` Rust library
[@within] to construct a preconditioner from pairwise blocks of the fixed-effect
Gram matrix. PyTorch provides an experimental LSMR backend on CPU, CUDA, and MPS.
The typed `demeaner=` configuration exposes this trade-off instead of selecting
one algorithm invisibly.

\autoref{fig:bench} reports median full-estimation runtimes over three runs for
the reproducible "simple" and "difficult" designs used in the `fixest`
benchmarks. Absolute values, and especially CPU-GPU comparisons, are
hardware-dependent. The difficult worker-firm-year design illustrates why
`PyFixest` offers both low-overhead MAP and a preconditioned solver rather than a
single backend.

![Median runtime for fixed-effects OLS ($k=10$) across PyFixest demeaners, R `fixest`, and Julia `FixedEffectModels.jl`. Scripts and result data are included in the repository; absolute timings are hardware-dependent.\label{fig:bench}](bench_readme.png)

## Architecture and scope

Formula parsing and model-matrix construction are separated from estimation
planning, fitting, and post-estimation. Public functions create a typed
configuration; a plan expands multiple-estimation formulas; runners construct and
fit each model; result objects delegate numerical post-estimation work to
standalone modules. Performance-critical demeaning, singleton detection, clustered
variance, and HAC loops live in Rust, while orchestration and non-hot-path
numerics remain readable NumPy. Optional backends are loaded only when requested.

The resulting API covers OLS, WLS, IV, Poisson, logit, probit, Gaussian GLMs, and
quantile regression [@koenker2001quantile]. It includes difference-in-differences
estimators [@gardner2022two; @dube2023local; @sun2021estimating], heteroskedastic,
cluster-robust and HAC inference, wild cluster bootstrap
[@roodman2019fast; @fischer2022fwildclusterboot], randomization inference
[@hess2017randomization], causal cluster variance [@abadie2023should],
multiple-testing corrections [@romano2005exact; @clarke2020romano], and safe
anytime-valid inference [@lindon2026anytime]. Reporting includes regression tables
for Great Tables, LaTeX, and Typst, coefficient plots, Gelbach decomposition
[@gelbach2016covariates], and weak-IV diagnostics [@lal2023much].

# Example

After `pip install pyfixest`, a fixed-effects regression with clustered inference
is:

```python
import pyfixest as pf

data = pf.get_data()
fit = pf.feols("Y ~ X1 | f1 + f2", data=data, vcov={"CRV1": "group_id"})
fit.etable()
```

The corresponding `fixest` call retains the same estimator name and formula
structure:

```r
library(fixest)
fit <- feols(Y ~ X1 | f1 + f2, data = data, vcov = ~group_id)
etable(fit)
```

Python necessarily uses a formula string and a dictionary for clustered
inference, but the model specification and defaults remain recognizable.

# Research impact statement

`PyFixest` has been developed publicly since 2022 and credits 60 contributors.
As of 30 August 2026, Pepy records more than 928,000 PyPI downloads, including
about 111,000 in the preceding 30 days [@pepy]; these installation counts include
automated and continuous-integration traffic and should not be read as unique
users.

Documented external uses include Instacart's analysis of high-cardinality
marketplace experiments [@knight2026instacart], the research workflow accompanying
*Large Scale Longitudinal Experiments* [@lal2024large], and use as the regression
backend for extended two-way fixed effects in `ModernDiD` [@moderndid2026]. The
repository provides extensive tutorials, cross-language reference tests, and
reproducible benchmark scripts. An archived release is available under the
Zenodo concept DOI
[10.5281/zenodo.15814089](https://doi.org/10.5281/zenodo.15814089).

# AI usage disclosure

Generative-AI tools have assisted parts of the project's code, tests,
documentation, and manuscript. Recorded tools include ChatGPT and GitHub Copilot
(historical hosted model versions were not retained), Anthropic Claude 3.5, and
OpenAI Codex with GPT-5. Assistance included code suggestions, refactoring, test
scaffolding, documentation, and drafting or copy-editing text. Human contributors
reviewed and edited every retained output, ran the numerical and cross-language
test suites, and made all econometric, architectural, authorship, and submission
decisions. No AI system is an author.

# Acknowledgements

We thank Laurent Bergé, Kyle Butts, Grant McDermott, and the `fixest` community.
`fixest` is the conceptual and API upstream for `PyFixest`; without its estimator
design, formula conventions, many defaults, and reporting ideas, `PyFixest` would
not exist in its present form [@berge2026fixest]. We also thank Matthew Wardrop and
the Formulaic contributors for the parsing and model-matrix infrastructure on
which the formula interface is built [@wardrop_formulaic_2026].

`PyFixest` shares no source code with the GPL-licensed `fixest` except for a
collinearity-detection routine that reimplements Bergé's C++ routine and is
distributed under the MIT license with his permission. We thank all `PyFixest`
contributors and participants in community development sprints. The appliedAI
Institute for Europe supported Kristof Schröder's work on Rust demeaning and the
`within` solver; responsibility for econometric validation and this manuscript
rests with the authors. Relevant institutional relationships are reflected in
the author affiliations above.

# References
