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
    email: alexander-fischer1801@t-online.de
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
  - name: Apoorva Lal
    affiliation: '7'
  - name: Daman Dhaliwal
    affiliation: '8'
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
  - index: 7
    name: OpenAI, United States
  - index: 8
    name: Independent researcher, Canada
date: 31 August 2026
bibliography: paper.bib
---

# Summary

`PyFixest` brings fast fixed-effects regression and the workflow of R's `fixest`
package to Python [@berge2026fixest]. It is built for empirical work in which a
model may include thousands or millions of workers, firms, products, places, or
time effects. Instead of expanding these categories into a large matrix of dummy
variables, `PyFixest` absorbs them before fitting the coefficients of interest.

The public API deliberately follows `fixest`: estimator names, fixed-effects
formula layout, multiple-estimation operators, variance interfaces,
small-sample corrections, singleton handling, collinearity conventions, and many
defaults all have their origin there. We regard these as contributions of the
`fixest` authors, not as inventions of `PyFixest`, and ask users to consider
citing `fixest` alongside this paper. `PyFixest` is nevertheless an independent
implementation, written in Python and Rust rather than translated from the R
package.

The package fits OLS, WLS, instrumental-variable, Poisson, generalized-linear,
and quantile regressions [@koenker2001quantile]. It also provides inference,
reporting, and causal-analysis tools. An economist working in Python can
therefore estimate a model, compute robust inference, and prepare output without
stitching together a different set of packages for each task.

# Statement of Need

A worker-firm panel can easily contain millions of observations and hundreds of
thousands of fixed-effect levels. Materializing one indicator column per level is
wasteful and can make an otherwise ordinary regression run out of memory. The
problem appears throughout applied economics and neighboring fields: matched
employer-employee data, marketplace experiments, hospital and physician panels,
trade flows, and education records all have this structure.

The intended users are applied researchers and data-science teams who want a
formula interface, econometric standard errors, and publication-ready output
without leaving Python. The problem is not limited to OLS. Poisson and other
generalized linear models are fitted with changing working weights, so their
fixed effects must be absorbed again during every iteration
[@correia2020fast; @stammann2017fast]. Demeaning the data once and handing the
result to a generic regression routine is not enough.

# State of the Field

R and Stata users can choose among mature tools such as `lfe` [@gaure2013lfe],
`fixest` [@berge2026fixest], `reghdfe` [@correia2023reghdfe], and `alpaca`
[@stammann2020package]. Julia offers `FixedEffectModels.jl`
[@fixedeffectmodelsjl_2025]. Python's `statsmodels` [@seabold2010statsmodels]
does not have a native multi-way high-dimensional fixed-effects estimator.
`pyhdfe` [@gortmaker_pyhdfe_2023] implements absorption algorithms, while
`linearmodels` [@linearmodels] exposes an absorbing least-squares estimator.
Those are useful building blocks for linear models, but they do not provide the
range of estimators and post-estimation methods familiar to `fixest` users.

We wanted the syntax and estimation flow familiar from `fixest`, a consistent
API across estimators, and high performance on fixed-effects problems. Extending
one of these packages to provide that combination would have changed its scope
substantially. We therefore built a separate package, but reuse focused
infrastructure where it already exists.
Formulaic [@wardrop_formulaic_2026] parses Wilkinson expressions and constructs
model matrices; a PyFixest layer adds fixed effects, multiple-estimation
operators, formula expansion, and estimation planning. This keeps formula parsing
separate from estimation code as the syntax grows.

# Software Design

The implementation separates model specification from numerical work. Public
functions such as `feols`, `fepois`, `feglm`, and `quantreg` first turn the user's
formula and options into a typed configuration. An estimation plan expands
multiple-model formulas and prepares the model matrices. Model classes then
coordinate fitting and inference, while numerical routines operate on arrays and
return small result objects. This keeps the econometric calculations testable
without tying them to a large model class.

For absorbed fixed-effects models, PyFixest does not construct the full
dummy-variable matrix. By the Frisch-Waugh-Lovell theorem
[@frisch1933; @lovell1963], it can residualize the outcome and regressors with
respect to the fixed effects and fit the smaller residualized problem. The
default Rust implementation uses alternating projections, or iterative demeaning
[@guimaraes2010; @gaure2013ols]. It has little setup cost and is fast for many
familiar panels. Sparse worker-firm or trade networks can be much harder:
information moves slowly between poorly connected groups, and alternating
projections may require many passes.

We chose to expose that trade-off. `LsmrDemeaner` combines LSMR [@fong2011] with
the `within` Rust library [@within] and a preconditioner built from pairwise
fixed-effect blocks. It costs more to set up but can be much faster on difficult
graphs. An experimental PyTorch backend runs LSMR on CPU, CUDA, or MPS. The
`demeaner=` argument makes the choice explicit rather than hiding a heuristic in
the estimator.

\autoref{fig:bench} shows this difference on the reproducible simple and
difficult designs used by the `fixest` benchmarks. Timings are medians of three
full estimation runs with ten covariates. Absolute values depend on hardware,
especially when a GPU is involved; the useful comparison is how the algorithms
respond when the fixed-effect graph becomes difficult.

![Median runtime for fixed-effects OLS ($k=10$) across PyFixest demeaners, R `fixest`, and Julia `FixedEffectModels.jl`. Scripts and result data are included in the repository; absolute timings are hardware-dependent.\label{fig:bench}](bench_readme.png)

Performance-sensitive demeaning, singleton detection, clustered variance, and
HAC loops live in Rust. Higher-level work remains in NumPy, where it is easier to
read and test. Beyond the core estimators, the package includes
difference-in-differences methods
[@gardner2022two; @dube2023local; @sun2021estimating], wild cluster bootstrap
[@roodman2019fast; @fischer2022fwildclusterboot], randomization inference
[@hess2017randomization], causal cluster variance [@abadie2023should],
multiple-testing corrections [@romano2005exact; @clarke2020romano], and safe
anytime-valid inference [@lindon2026anytime]. Reporting includes Great Tables,
LaTeX, and Typst tables, coefficient plots, Gelbach decomposition
[@gelbach2016covariates], and weak-IV diagnostics [@lal2023much].

Development has been significantly aided by the existence of `fixest` as a
reference implementation that we test against. Reference tests call R through
`rpy2` [@gautier2008rpy2] and compare coefficients, inference, and fit statistics
at estimator-specific tolerances; core OLS comparisons typically use relative
and absolute tolerances of $10^{-8}$. For example, Python requires formula
strings and dictionary variance specifications. The documentation calls out such
differences.

# Research Impact Statement

PyFixest is used in both research and production. Instacart has documented its
use for high-cardinality marketplace experiments [@knight2026instacart]. The
workflow for *Large Scale Longitudinal Experiments* uses the package
[@lal2024large], and `ModernDiD` relies on `feols`, `fepois`, and `feglm` for its
extended two-way fixed-effects estimator [@moderndid2026]. Together, these
examples show PyFixest being used for applied analysis, methods research, and as
infrastructure for other econometrics packages.

Development has taken place in public since 2022, with 60 credited contributors.
As of 30 August 2026, Pepy reports more than 928,000 PyPI downloads and about
111,000 during the preceding 30 days [@pepy]. These figures include automated
and continuous-integration traffic, so they are evidence of distribution rather
than a count of individual users. Tutorials, reference tests, and benchmark
scripts are maintained in the repository, and an archived release is available
under the Zenodo concept DOI
[10.5281/zenodo.15814089](https://doi.org/10.5281/zenodo.15814089).

# AI Usage Disclosure

OpenAI Codex with GPT-5 assisted with repository research, source checking,
drafting, and copy-editing. The authors reviewed and edited the text and remain
responsible for its claims, citations, authorship, and submission. No AI system is
an author.

# Acknowledgements

We thank Laurent Bergé, Kyle Butts, Grant McDermott, and the wider `fixest`
community. `fixest` is the conceptual and API upstream of `PyFixest`; without its
estimator design, formula conventions, defaults, and reporting ideas, this
package would look very different [@berge2026fixest]. We also thank Matthew
Wardrop and the Formulaic contributors for the parser and model-matrix
infrastructure [@wardrop_formulaic_2026].

PyFixest shares no source code with the GPL-licensed `fixest` apart from a
reimplementation of Bergé's collinearity-detection routine, distributed under the
MIT license with his permission. We thank the PyFixest contributors and the
participants in its community development sprints. The appliedAI Institute for
Europe supported Kristof Schröder's work on Rust demeaning and the `within`
solver; responsibility for econometric validation and this manuscript rests with
the authors. Relevant institutional relationships are listed in the affiliations
above.

# References
