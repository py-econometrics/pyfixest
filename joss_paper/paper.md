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
date: 23 November 2026
bibliography: paper.bib
---

# Summary

`PyFixest` brings the formula syntax and high-dimensional fixed-effects workflow
of R's `fixest` package [@berge2026fixest] to Python. It estimates ordinary and
weighted least squares and instrumental-variable models (`feols`), Poisson
(`fepois`) and generalized linear models (`feglm`), quantile regressions
(`quantreg`) [@koenker2001quantile], and difference-in-differences estimators.
Cluster-robust and bootstrap inference, regression tables (`etable`), and
coefficient and event-study plots (`coefplot`, `iplot`) ship in the same
package.

The package follows `fixest`'s estimator names, formula syntax, and
multiple-estimation operators, and adopts its default conventions for variance
estimation, small-sample degrees-of-freedom adjustments, singleton removal, and
collinearity handling. These design choices originate with `fixest`, so we ask
users to cite @berge2026fixest alongside this paper. `PyFixest` is an
independent implementation in Python and Rust built on `Formulaic`
[@wardrop_formulaic_2026] and `NumPy` [@harris2020array], and accepts both
`pandas` [@mckinney2010data] and `polars` [@vink_polars_2026] DataFrames.

# Statement of Need

Panel datasets in economics and the social sciences often contain millions of
observations and hundreds of thousands of worker, firm, or product effects,
frequently interacted with time. Constructing explicit dummy matrices at this
scale requires too much memory and time. Instead, linear estimators sweep out
the fixed effects using the Frisch-Waugh-Lovell theorem [@frisch1933;
@lovell1963] before estimating the slope parameters. In Poisson and other
generalized linear models fit by iteratively reweighted least squares, each
iteration solves a weighted least-squares step, so the fixed effects must be
absorbed again whenever the working weights change [@correia2020fast;
@stammann2017fast].

Demeaning variables in a separate preprocessing step and passing the residuals
to a standard regression routine also breaks inference: the regression does not
account for the absorbed parameters in its degrees of freedom, nor does it
apply the nested-cluster adjustments needed when fixed effects are nested
within clusters.

# State of the Field

Mature high-dimensional fixed-effects packages exist in R (`lfe`
[@gaure2013lfe], `fixest` [@berge2026fixest], `alpaca` [@stammann2020package]),
Stata (`reghdfe` [@correia2023reghdfe], `ppmlhdfe` [@correia2020fast]), and
Julia (`FixedEffectModels.jl` [@fixedeffectmodelsjl_2025]). In Python,
`statsmodels` [@seabold2010statsmodels] does not natively absorb multiple
high-dimensional fixed effects. `pyhdfe` [@gortmaker_pyhdfe_2023] provides
array-level demeaning routines, and `linearmodels` [@linearmodels] wraps
`pyhdfe` in `AbsorbingLS` for single linear regressions, without absorbed fixed
effects in instrumental-variable or generalized linear models, stepwise
formulas, or difference-in-differences estimators.

We built `PyFixest` as a standalone package instead of extending `statsmodels`
or `linearmodels`. A single `fixest` formula can expand into dozens of
specifications that share demeaned arrays, a workflow that does not fit the
single-model design of existing Python regression libraries.

# Software Design

A preprocessing pass first extracts fixed-effect terms, instrumental-variable
subformulas, and stepwise operators (`sw`, `sw0`, `csw`, `csw0`); `Formulaic`
[@wardrop_formulaic_2026] then parses the remaining Wilkinson formulas into
dense or sparse design matrices. For calls with multiple outcomes or stepwise
covariate blocks, `PyFixest` groups specifications that share the same fixed
effects and observation weights. Models in the same group reuse factor
encodings and cached demeaned arrays, so each variable is demeaned only once.

Sweeping out fixed effects (and optional varying slopes) is most commonly done
by alternating projections (MAP), which iteratively subtracts group means
[@guimaraes2010; @gaure2013ols], or by sparse least-squares solvers such as
diagonally preconditioned LSMR [@fong2011]. Accelerated MAP (as in `fixest` and
`reghdfe`) and diagonal LSMR (as in `FixedEffectModels.jl`) have low setup cost
and serve as strong defaults on well-connected panels. Both can slow down on
sparsely connected graphs (such as matched worker-firm or trade panels with low
mobility), where few co-occurrences link groups and updates travel slowly
across the graph.

By default, `PyFixest` solves the weighted residualization step with
preconditioned LSMR [@fong2011] via the `within` Rust crate [@within;
@fischer2026graph]. Its adaptive preconditioner starts with diagonal scaling and
switches to an additive Schwarz preconditioner, which solves smaller
subproblems on pairs of fixed effects, if convergence slows. Users can select
Rust alternating projections (`MapDemeaner`) or an experimental `PyTorch`
diagonal-LSMR backend (CPU, CUDA, or Apple MPS) through the `demeaner=`
argument.

\autoref{fig:bench} compares these backends against `fixest` and
`FixedEffectModels.jl` on the simple and difficult designs from the `fixest`
benchmark suite ($k=10$ covariates, medians of three end-to-end runs). On the
three well-connected or two-way panels, all solvers scale similarly with sample
size: `fixest` is fastest at $10^7$ observations (about 2–4 seconds),
`FixedEffectModels.jl` has the lowest overhead on small samples, and
`PyFixest within` pays a modest overhead on the simple three-way panel (about
16 seconds). The separation appears on the difficult three-way worker-firm-year
panel, where sparse links between workers and firms slow down standard solvers.
At $10^7$ observations, `PyFixest within` finishes on CPU in about 10 seconds
and diagonal LSMR on GPU (`PyFixest torch GPU`) in about 14 seconds, compared
with roughly 100–170 seconds for `fixest` and `FixedEffectModels.jl` and nearly
900 seconds for unaccelerated MAP (`PyFixest MAP`).

![Median runtime for fixed-effects OLS ($k=10$) across `PyFixest` demeaners, R `fixest`, and Julia `FixedEffectModels.jl`. Scripts and result data are included in the repository; absolute timings are hardware-dependent.\label{fig:bench}](bench_readme.png)

Rust also handles singleton detection, multi-way cluster-robust variance
matrices, and heteroskedasticity- and autocorrelation-consistent (HAC) standard
errors; estimation logic and post-estimation stay in Python and `NumPy`. The
core estimators (`feols`, `fepois`, `feglm`) return a common result interface
that higher-level routines build on directly. These include two-stage
[@gardner2022two], local-projections [@dube2023local], and interaction-weighted
[@sun2021estimating] difference-in-differences estimators, alongside Gelbach
decompositions [@gelbach2016covariates] and weak-instrument diagnostics
[@lal2023much]. Inference routines add the wild cluster bootstrap
[@roodman2019fast; @fischer2022fwildclusterboot], randomization inference
[@hess2017randomization], causal cluster variance [@abadie2023should],
Romano-Wolf stepdown corrections [@romano2005exact; @clarke2020romano], and
anytime-valid inference [@lindon2026anytime]. On the same result objects,
`etable` (via `maketables` [@maketables]) exports Great Tables, LaTeX, Typst,
or Word tables, and `coefplot` and `iplot` generate coefficient and event-study
plots.

Our test suite checks numerical agreement with R's `fixest` via `rpy2`
[@gautier2008rpy2] across point estimates, standard errors, and fit statistics
(within $10^{-8}$ relative and absolute tolerance for core OLS). Interface
differences from `fixest` mostly follow Python conventions, such as passing
formulas as strings and variance specifications as dictionaries
(`vcov={"CRV1": "id"}`).

# Research Impact Statement

Since its initial release in 2022, `PyFixest` has been cited or used in roughly
100 working papers, journal articles, and dissertations indexed on Google
Scholar (as of November 2026). In industry, Instacart uses `PyFixest` to
analyze high-cardinality marketplace experiments [@knight2026instacart]. Other
econometric libraries also build on it: `ModernDiD`, for example, uses `feols`,
`fepois`, and `feglm` in its extended two-way fixed-effects estimator
[@moderndid2026].

Around 60 people have contributed to `PyFixest` on GitHub, and the package has
over 1.4 million downloads on PyPI, including automated and
continuous-integration traffic [@pepy]. Documentation, tutorials, the `rpy2`
validation suite, and benchmark scripts are in the GitHub repository, and
releases are archived on Zenodo
([10.5281/zenodo.15814089](https://doi.org/10.5281/zenodo.15814089)).

# AI Usage Disclosure

We used large language models from different providers to assist with drafting
and editing this paper. The authors reviewed the manuscript and take full
responsibility for its content.

# Acknowledgements

We thank Laurent Bergé, Kyle Butts, Grant McDermott, and the wider `fixest`
community for their work on `fixest` and its extensions [@berge2026fixest].
`PyFixest` is written independently of the GPL-licensed `fixest`, aside from a
Python and Rust port of Bergé's collinearity-detection algorithm, distributed
under the MIT license with his permission. We also thank Matthew Wardrop and
the `Formulaic` contributors [@wardrop_formulaic_2026], as well as all
`PyFixest` contributors and community sprint participants. The appliedAI
Institute for Europe supported Kristof Schröder's work on Rust demeaning and the
`within` solver. Relevant institutional relationships are listed in the
affiliations above.

# References
