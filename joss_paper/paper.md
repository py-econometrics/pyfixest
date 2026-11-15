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

`PyFixest` brings the high-dimensional fixed-effects workflow of R's `fixest`
package [@berge2026fixest] to Python. It implements ordinary and weighted least
squares and instrumental-variable regression (`feols`), Poisson (`fepois`) and
generalized linear models (`feglm`), quantile regression (`quantreg`)
[@koenker2001quantile], and several difference-in-differences estimators,
alongside cluster-robust and bootstrap inference, regression tables (`etable`),
and coefficient plots (`coefplot`, `iplot`).

The package follows `fixest`'s estimator names, formula syntax, and
multiple-estimation operators, and adopts its default conventions for variance
estimation, small-sample degrees-of-freedom adjustments, singleton removal, and
collinearity handling. Because these design choices originate with `fixest`, we
ask users to cite @berge2026fixest alongside this paper. Under the hood,
`PyFixest` is an independent Python and Rust implementation built on
`Formulaic` [@wardrop_formulaic_2026] and NumPy [@harris2020array], accepting
both `pandas` [@mckinney2010data] and `polars` [@vink_polars_2026] DataFrames.

# Statement of Need

Panel datasets in economics and the social sciences often contain millions of
observations and hundreds of thousands of worker, firm, or product effects,
frequently interacted with time. Including explicit indicator variables at this
scale is both memory-prohibitive and computationally infeasible. Instead,
linear estimators project out the fixed effects via the Frisch-Waugh-Lovell
(FWL) theorem [@frisch1933; @lovell1963] before estimating the slope
parameters. In Poisson and other generalized linear models fitted by
iteratively reweighted least squares (IRLS), this weighted projection must be
repeated at every iteration as the working weights update
[@correia2020fast; @stammann2017fast].

Furthermore, demeaning data manually in a separate preprocessing step and
passing the residuals to a generic regression routine produces wrong
degrees-of-freedom adjustments and invalid cluster-robust standard errors when
fixed effects are nested within clusters. Applied empirical work requires
fixed-effect absorption to be coupled directly with econometric inference,
hypothesis testing, and reporting.

# State of the Field

Mature high-dimensional fixed-effects packages exist in R (`lfe`
[@gaure2013lfe], `fixest` [@berge2026fixest], `alpaca` [@stammann2020package]),
Stata (`reghdfe` [@correia2023reghdfe], `ppmlhdfe` [@correia2020fast]), and
Julia (`FixedEffectModels.jl` [@fixedeffectmodelsjl_2025]). In Python,
`statsmodels` [@seabold2010statsmodels] does not natively absorb multiple
high-dimensional fixed effects. `pyhdfe` [@gortmaker_pyhdfe_2023] provides
array-level demeaning routines, and `linearmodels` [@linearmodels] wraps
`pyhdfe` in its `AbsorbingLS` estimator, but neither supports absorbed fixed
effects in instrumental-variable or generalized linear models, `fixest`-style
stepwise estimation, or difference-in-differences estimators.

We built `PyFixest` as a standalone package mirroring `fixest` rather than
extending `statsmodels` or `linearmodels`. As longtime R users, we found
`fixest`'s formula syntax and estimation workflow concise and already familiar
to many empirical researchers, and supporting its multi-model semantics would
have required architectural changes outside the scope of existing Python
libraries.

# Software Design

`PyFixest` uses `Formulaic` [@wardrop_formulaic_2026] to parse Wilkinson
formulas and build dense or sparse design matrices, after a preprocessing pass
that extracts fixed-effects slots, instrumental-variable subformulas, and
stepwise operators (`sw`, `sw0`, `csw`, `csw0`). When a call specifies multiple
outcomes or stepwise covariate blocks, `PyFixest` groups the resulting
specifications by fixed-effect structure and observation weights. Models in the
same group share factor encodings and cached demeaned arrays, so each unique
variable is projected only once.

To residualize the outcome and regressors with respect to the fixed effects
(and optional varying slopes), the classic method of alternating projections
(MAP) iteratively sweeps out group means [@guimaraes2010; @gaure2013ols]. MAP
has low per-iteration overhead and converges quickly on dense or well-connected
panels, but it can stall on sparse bipartite or tripartite graphs (such as
matched worker-firm or trade panels) where information propagates slowly across
weakly connected components.

By default, `PyFixest` instead solves the weighted residualization problem with
preconditioned LSMR [@fong2011] via the `within` Rust crate [@within]. Its
adaptive preconditioner starts with diagonal scaling and switches to an
additive Schwarz preconditioner built from pairwise fixed-effect blocks when
convergence slows. Users can also switch to Rust alternating projections
(`MapDemeaner`) or an experimental PyTorch LSMR backend (on CPU, CUDA, or Apple
MPS) via the `demeaner=` argument.

\autoref{fig:bench} compares these solvers on the simple and difficult panel
designs from the `fixest` benchmark suite ($k=10$ covariates, medians of three
end-to-end runs). On the well-connected designs, all solvers scale similarly
with sample size. On the difficult three-way worker-firm-year graph,
preconditioned LSMR (`PyFixest within`, the default `LsmrDemeaner`) finishes in
about 10 seconds at $10^7$ observations, whereas alternating projections
(`PyFixest MAP`) approaches $10^3$ seconds.

![Median runtime for fixed-effects OLS ($k=10$) across `PyFixest` demeaners, R `fixest`, and Julia `FixedEffectModels.jl`. Scripts and result data are included in the repository; absolute timings are hardware-dependent.\label{fig:bench}](bench_readme.png)

Singleton removal, multi-way cluster-robust variance calculations, and
heteroskedasticity- and autocorrelation-consistent (HAC) standard errors are
also written in Rust, while model fitting and post-estimation live in Python
and NumPy. On top of the core estimators, `PyFixest` provides
difference-in-differences routines, including two-stage [@gardner2022two],
local-projections [@dube2023local], and Sun-Abraham interaction-weighted
[@sun2021estimating] estimators, as well as Gelbach decompositions
[@gelbach2016covariates] and weak-instrument diagnostics [@lal2023much].
Supported inference methods include the wild cluster bootstrap
[@roodman2019fast; @fischer2022fwildclusterboot], randomization inference
[@hess2017randomization], causal cluster variance [@abadie2023should],
Romano-Wolf multiple-testing corrections [@romano2005exact; @clarke2020romano],
and anytime-valid inference [@lindon2026anytime]. For output, `etable` renders
Great Tables, LaTeX, Typst, and Word tables through `maketables` [@maketables],
and `coefplot` and `iplot` plot coefficients and event-study paths.

Our test suite checks numerical agreement with R's `fixest` via `rpy2`
[@gautier2008rpy2] across point estimates, standard errors, and fit statistics
(within $10^{-8}$ relative and absolute tolerance for core OLS). Where the
interface departs from `fixest`, it does so to follow Python syntax, such as
passing formulas as strings and clustering options as dictionaries
(`vcov={"CRV1": "id"}`).

# Research Impact Statement

Since its initial release in 2022, `PyFixest` has been cited or used in roughly
100 working papers, journal articles, and dissertations indexed on Google
Scholar. In industry, Instacart uses `PyFixest` to analyze high-cardinality
marketplace experiments [@knight2026instacart]. It also serves as a backend for
other econometric libraries, such as `ModernDiD`, which calls `feols`,
`fepois`, and `feglm` in its extended two-way fixed-effects estimator
[@moderndid2026].

Sixty contributors have participated in developing `PyFixest` on GitHub, and
the package has been downloaded over 1.4 million times from PyPI [@pepy].
Documentation, tutorials, the `rpy2` validation suite, and benchmark scripts
are available in the repository, with releases archived on Zenodo under concept
DOI [10.5281/zenodo.15814089](https://doi.org/10.5281/zenodo.15814089).

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
