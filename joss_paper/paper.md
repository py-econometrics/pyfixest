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
    affiliation: '6'
  - name: Daman Dhaliwal
    affiliation: '7'
  - name: Wenzhi Ding
    orcid: 0000-0002-4784-8848
    affiliation: '8'
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
    name: OpenAI, United States
  - index: 7
    name: Independent researcher, Canada
  - index: 8
    name: The Hong Kong Polytechnic University, Hong Kong SAR
date: 31 August 2026
bibliography: paper.bib
---

# Summary

`PyFixest` brings the high-dimensional fixed-effects estimation workflow of R's
`fixest` package [@berge2026fixest] to Python. It estimates ordinary and
weighted least squares (`feols`), instrumental-variable models, Poisson
(`fepois`) and generalized linear models (`feglm`), quantile regressions
(`quantreg`) [@koenker2001quantile], and difference-in-differences models, and
pairs them with cluster-robust and bootstrap inference, regression tables, and
coefficient plots.

The public API mirrors `fixest` in its estimator names, formula syntax, and
multiple-estimation operators, as well as its numerical defaults for variance
estimation, small-sample corrections, singleton removal, and collinearity
detection. Because these conventions originate with `fixest`, we ask users to
cite @berge2026fixest together with this paper. `PyFixest` itself is an
independent Python and Rust implementation built around `Formulaic`
[@wardrop_formulaic_2026], NumPy [@harris2020array], `pandas`
[@mckinney2010data], and `polars` [@vink_polars_2026].

# Statement of Need

Empirical panels in economics and neighboring social and data sciences routinely
span millions of observations and hundreds of thousands of fixed-effect levels
across workers, firms, products, or time periods, where dummy-variable encoding
is infeasible in memory. In linear regression, the fixed effects are projected
out before fitting the coefficients of interest; in Poisson and other
generalized linear models fitted by iteratively reweighted least squares (IRLS),
that weighted projection is repeated at every iteration as the working weights
change [@correia2020fast; @stammann2017fast].

Applied work in Python also requires multi-way clustered and
heteroskedasticity-robust standard errors, joint hypothesis tests, event-study
and difference-in-differences estimators, and publication-ready regression
tables. `PyFixest` combines fixed-effect absorption, inference, and reporting
in one interface so users do not have to stitch together separate libraries.

# State of the Field

R and Stata offer mature fixed-effects packages including `lfe` [@gaure2013lfe],
`fixest` [@berge2026fixest], `reghdfe` [@correia2023reghdfe], and `alpaca`
[@stammann2020package], while Julia provides `FixedEffectModels.jl`
[@fixedeffectmodelsjl_2025]. In Python, `statsmodels` [@seabold2010statsmodels]
lacks a native multi-way high-dimensional fixed-effects estimator. `pyhdfe`
[@gortmaker_pyhdfe_2023] provides array-level demeaning routines, and
`linearmodels` [@linearmodels] builds on it for absorbing least squares, but
neither supports generalized linear models with absorbed fixed effects,
`fixest`-style multiple estimation, or integrated post-estimation tables and
plots.

We chose to mirror `fixest`'s API in a standalone package rather than design a
new interface or extend `statsmodels` or `linearmodels`: `fixest` already
provides a concise, well-tested grammar for econometric workflows that we
admired as R users and wanted to bring to Python without reinventing the wheel.
For Wilkinson formula parsing and dense or sparse design-matrix construction,
`PyFixest` builds on `Formulaic` [@wardrop_formulaic_2026], adding a
preprocessing layer for fixed-effects slots, instrumental-variable subformulas,
and stepwise multiple-estimation operators (`sw`, `sw0`, `csw`, `csw0`).

# Software Design

When a formula specifies multiple outcomes or stepwise covariate blocks,
`PyFixest` expands the call into an estimation plan grouped by fixed-effect
structure and observation weights. Models within the same group share factor
encodings and cached demeaned arrays, so estimating a sequence of
specifications avoids redundant projections.

For models with absorbed fixed effects, `PyFixest` applies the
Frisch-Waugh-Lovell theorem [@frisch1933; @lovell1963] to residualize the
outcome and regressors with respect to the fixed effects. The classic method of
alternating projections (MAP) iteratively sweeps group means
[@guimaraes2010; @gaure2013ols]. MAP has low per-iteration cost and converges
rapidly on dense or well-connected panels, but can stall on sparse bipartite or
tripartite graphs (such as matched worker-firm or trade panels) where
information propagates slowly across weakly connected components.

By default, `PyFixest` instead solves the weighted residualization problem with
preconditioned LSMR [@fong2011] through the `within` Rust library [@within],
which also supports absorbed varying slopes. Its adaptive preconditioner starts
with diagonal scaling and switches to an additive Schwarz preconditioner built
from pairwise fixed-effect blocks when convergence slows. Users can also select
Rust alternating projections (`MapDemeaner`) or an experimental PyTorch LSMR
backend on CPU, CUDA, or Apple MPS with the `demeaner=` argument.

\autoref{fig:bench} compares these approaches on the simple and difficult panel
designs from the `fixest` benchmark suite ($k=10$ covariates, medians of three
end-to-end runs). On the well-connected designs, all solvers scale similarly
with sample size. On the difficult three-way worker-firm-year graph,
preconditioned LSMR (`PyFixest within`, the default `LsmrDemeaner`) finishes in
about 10 seconds at $10^7$ observations, whereas alternating projections
(`PyFixest MAP`) approaches $10^3$ seconds.

![Median runtime for fixed-effects OLS ($k=10$) across `PyFixest` demeaners, R `fixest`, and Julia `FixedEffectModels.jl`. Scripts and result data are included in the repository; absolute timings are hardware-dependent.\label{fig:bench}](bench_readme.png)

Singleton detection, multi-way clustered covariance kernels, and
heteroskedasticity- and autocorrelation-consistent (HAC) standard errors are
likewise implemented in Rust, while higher-level model orchestration stays in
Python and NumPy. Beyond core OLS, IV, and GLM routines, `PyFixest` implements
two-stage [@gardner2022two], local-projections [@dube2023local], and
Sun-Abraham interaction-weighted [@sun2021estimating]
difference-in-differences estimators; Gelbach decompositions
[@gelbach2016covariates]; and weak-instrument diagnostics [@lal2023much].
Inference methods include the wild cluster bootstrap
[@roodman2019fast; @fischer2022fwildclusterboot], randomization inference
[@hess2017randomization], causal cluster variance [@abadie2023should],
Romano-Wolf multiple-testing corrections [@romano2005exact; @clarke2020romano],
and anytime-valid inference [@lindon2026anytime]. For reporting, `etable`
delegates table rendering to `maketables` [@maketables] (producing Great Tables,
LaTeX, Typst, and Word output), and `coefplot` and `iplot` generate coefficient
and event-study figures.

Continuous integration validates numerical parity against R's `fixest` through
`rpy2` [@gautier2008rpy2], checking point estimates, standard errors, and fit
statistics (at $10^{-8}$ relative and absolute tolerance for core OLS).
Intentional API departures follow Python conventions, such as string formulas
and dictionary variance specifications like `vcov={"CRV1": "id"}`, and are
documented in the user guide.

# Research Impact Statement

Since its first release in 2022, `PyFixest` has been cited or used in about 100
working papers, journal articles, and dissertations indexed by Google Scholar.
Outside academic research, Instacart uses `PyFixest` to analyze high-cardinality
marketplace experiments [@knight2026instacart], and the Python package
`ModernDiD` calls `feols`, `fepois`, and `feglm` inside its extended two-way
fixed-effects estimator [@moderndid2026].

`PyFixest` is developed on GitHub with 60 contributors and has recorded over
1.4 million downloads on PyPI [@pepy]. The repository provides user tutorials,
the `rpy2` test suite, and reproducible benchmark scripts, and releases are
archived on Zenodo under concept DOI
[10.5281/zenodo.15814089](https://doi.org/10.5281/zenodo.15814089).

# AI Usage Disclosure

We used large language models from different providers to assist with drafting
and editing this paper. The authors reviewed the manuscript and take full
responsibility for its content.

# Acknowledgements

We thank Laurent Bergé, Kyle Butts, Grant McDermott, and the wider `fixest`
community for their work on `fixest` and its extensions [@berge2026fixest].
`PyFixest` shares no source code with the GPL-licensed `fixest` apart from a
Python and Rust reimplementation of Bergé's collinearity-detection algorithm,
distributed under the MIT license with his permission. We also thank Matthew
Wardrop and the `Formulaic` contributors [@wardrop_formulaic_2026], as well as
all `PyFixest` contributors and community sprint participants. The appliedAI
Institute for Europe supported Kristof Schröder's work on Rust demeaning and the
`within` solver; responsibility for econometric validation and this manuscript
rests with the authors. Relevant institutional relationships are listed in the
affiliations above.

# References
