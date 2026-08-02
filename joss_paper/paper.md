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
    affiliation: 1
  - name: Leonard Stimpfle
    affiliation: 1
  - name: Kristof Schroeder
    affiliation: 1
  - name: Dirk Sliwka
    affiliation: 1
  - name: Juan Orduz
    affiliation: 1
  - name: Wenzhi Ding
    affiliation: 1
affiliations:
  - name: Affiliation TBD
    index: 1
date: 2 August 2026
bibliography: paper.bib
---

# Summary

`PyFixest` fits regressions with multiple, potentially high-dimensional fixed
effects in Python. It reimplements the API of the R package `fixest`
[@berge2026fixest]: formula syntax, variance-covariance choices, small-sample
corrections, singleton handling and collinearity rules all follow `fixest`, so
the same analysis runs in either language with almost no edits.

`PyFixest` is written from scratch in Python and Rust rather than translated
from `fixest`'s source. Its test suite compares
coefficients, standard errors, test statistics, and confidence intervals with
`fixest` through `rpy2` [@gautier2008rpy2], using a tolerance of $10^{-8}$.

The interface and the defaults that `PyFixest` reimplements are the work of the
`fixest` authors. We therefore ask users of `PyFixest` to consider citing `fixest`
[@berge2026fixest] alongside this paper, even if they haven't used it in their analyses..

# Statement of need

Regressions that absorb several sets of categorical effects are common in
applied economics. @goldsmith2026tracking finds that roughly half of papers in
leading economics and finance journals mention fixed effects. Labor economists
separate worker heterogeneity from firm wage premia, health economists compare
physicians within regions, and education researchers control for students,
teachers, and schools simultaneously. These designs often include hundreds of
thousands of categorical levels.

R and Stata users have had dedicated tools for this for more than a decade:
`lfe` [@gaure2013lfe], `fixest` [@berge2026fixest] and `reghdfe`
[@correia2023reghdfe] for linear models, `alpaca` [@stammann2020package] for
GLMs. Julia has `FixedEffectModels.jl` [@fixedeffectmodelsjl_2025]. Python has
been thinner. `statsmodels`
[@seabold2010statsmodels] ships no multi-way fixed-effects estimator, so the
common workaround is to demean with `pyhdfe` [@gortmaker_pyhdfe_2023] and pass
the residualized data to a generic OLS routine; `linearmodels` [@linearmodels]
offers `AbsorbingLS`, which also delegates demeaning to `pyhdfe`. Both work for
linear models. Neither carries over to GLMs fit by iteratively reweighted least
squares, where the working weights change at every step and the design has to be
re-demeaned each time [@correia2020fast; @stammann2017fast].

`PyFixest` covers OLS, WLS, IV, GLMs, and quantile regression behind one
`fixest`-style API (`feols`, `fepois`, `feglm`, `quantreg`), together with
several demeaning backends and post-estimation methods.
Formulas are parsed with `formulaic` [@wardrop_formulaic_2024], whose
Wilkinson-formula implementation carries the multiple-estimation syntax
`PyFixest` inherits from `fixest`.

# Absorbing fixed effects

Consider $y = X\beta + D\alpha + \varepsilon$, where $X$ contains the regressors
of interest and $D$ is the fixed-effect design. In applied work, $D$ may have
millions of columns. Forming the cross-product of $[X \;\; D]$, let alone
inverting it, is infeasible. The Frisch-Waugh-Lovell theorem
[@frisch1933; @lovell1963] avoids this calculation: residualize $y$ and each
column of $X$ against $D$, then regress the residualized outcome on the
residualized regressors. The resulting $\hat\beta$ equals the coefficient from
the full regression. In most applications, $\alpha$ is a nuisance parameter.

This reduces estimation to a sequence of least-squares projections onto the
column space of $D$. The projections share the Gramian $G = D'WD$ and differ
only in their right-hand side.

The standard approach is the method of alternating projections (MAP), also
called iterative demeaning [@guimaraes2010; @gaure2013ols]. Each diagonal block
of $G$ contains the total weight for each level of one factor. Holding the other
factors fixed, the update for one factor is therefore just a pass through the
data that subtracts weighted group means. MAP cycles through the factors, for
example workers, firms, and years, until the residual converges. The individual
sweeps are cheap, and implementations accelerate the sequence
[@gaure2013ols; @berge2026fixest].

The off-diagonal blocks of $G$ do not appear in the update rule. They contain
cross-tabulations, such as the workers observed at each firm, and affect later
updates only through the current residual. Convergence therefore depends on the
fixed-effects co-occurrence graph. When many workers move between firms, the
graph is well connected and MAP may converge in a few sweeps. When mobility is
limited, sorting is strong, or one factor is nearly nested in another, many more
sweeps may be required [@gaure2013ols]. The graph structures that weaken the
identification of worker and firm effects also slow their computation.

A Krylov solver can instead use a preconditioner: a cheap approximation to
$G^{-1}$ that preserves the solution while reducing the number of iterations.
`PyFixest` supports LSMR [@fong2011] with a preconditioner from the `within`
library [@within]. The preconditioner uses factor-pair blocks of the Gramian.
After a sign change, each block is a graph Laplacian that admits a sparse
approximate factorization. Constructing it adds overhead that is not repaid for
dense, well-connected designs, so MAP remains the default. It can substantially
reduce runtimes for sparse or strongly sorted graphs. @fischer2026graph describe
the construction and its benchmarks.

# Demeaner backends and performance

The choice of solver is exposed through a typed `demeaner=` argument. All
backends solve the same problem and agree up to solver tolerance; they differ in
speed:

- `MapDemeaner(backend="rust")`, the default: MAP in Rust, no optional
  dependencies.
- `MapDemeaner(backend="numba")`: the same algorithm through Numba.
- `LsmrDemeaner()`: preconditioned LSMR through `within`, using the factor-pair
  (additive Schwarz) preconditioner.
- `LsmrDemeaner(backend="torch", device=...)`: experimental LSMR on CPU or GPU
  (CUDA, MPS) via PyTorch.

\autoref{fig:bench} reports median runtimes on the "simple" and "difficult"
designs from the [fixest
benchmarks](https://github.com/kylebutts/fixest_benchmarks) with $k=10$
covariates, for `PyFixest` MAP, `PyFixest` `within` LSMR, `PyFixest` torch on
CUDA, R `fixest`, and Julia `FixedEffectModels.jl`.

![Median runtime for fixed-effects OLS ($k=10$) across PyFixest demeaners, R `fixest`, and Julia `FixedEffectModels.jl`. Absolute timings are hardware-dependent.\label{fig:bench}](bench_readme.png)

On the "simple" designs, the implementations in the figure have similar
runtimes and MAP is competitive. The difficult worker-firm-year design separates
them: at ten million observations, the `within` backend is nearly two orders of
magnitude faster than plain `PyFixest` MAP and about one order of magnitude
faster than the accelerated MAP in `fixest`. The package documentation includes
additional AKM-style examples with limited mobility and nested effects.

# Distinctive features

Beyond fixed-effects OLS, `PyFixest` includes:

- **IV and GLM**: instrumental variables and GLMs (Poisson, logit, probit,
  Gaussian) with high-dimensional fixed effects, including separation handling
  for Poisson [@correia2020fast].
- **Quantile regression** with an interior-point solver [@koenker2001quantile].
- **Difference-in-differences**: TWFE, two-stage imputation
  [@gardner2022two; @butts2021did2s], local projections
  [@dube2023local; @busch2023lpdid], and Sun-Abraham event studies
  [@sun2021estimating].
- **Inference**: heteroskedasticity- and cluster-robust variance estimators
  including CRV3 [@mackinnon2023fast], HAC, the wild cluster bootstrap
  [@roodman2019fast; @fischer2022fwildclusterboot], randomization inference
  [@hess2017randomization], the causal cluster variance estimator
  [@abadie2023should], Romano-Wolf corrections
  [@romano2005exact; @clarke2020romano], and simultaneous confidence bands
  [@montiel2019simultaneous].
- **Reporting**: publication-ready tables (`etable`) through Great Tables or
  LaTeX booktabs, coefficient plots, and multiple-estimation syntax (`csw`,
  `sw`) that reuses the demeaning cache across related formulas.
- **Post-estimation**: Gelbach decomposition [@gelbach2016covariates], weak-IV
  diagnostics [@lal2023much], and compressed-regression workflows
  [@wong2021you; @lal2024large].

# Example

After `pip install pyfixest`:

```python
import pyfixest as pf

data = pf.get_data()
pf.feols("Y ~ X1 | f1 + f2", data=data).summary()
```

```
Estimation:  OLS
Dep. var.: Y, Fixed effects: f1+f2
Inference:  CRV1
Observations:  997

| Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
|:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
| X1            |     -0.919 |        0.065 |   -14.057 |      0.000 | -1.053 |  -0.786 |
---
RMSE: 1.441   R2: 0.609   R2 Within: 0.2
```

Multiple estimation and clustered inference use the same formula grammar as
`fixest`:

```python
fit = pf.feols("Y + Y2 ~ X1 | csw0(f1, f2)", data=data, vcov={"CRV1": "group_id"})
fit.etable()
```

To use a different solver, pass a `demeaner`:

```python
# sparse or strongly sorted fixed effects: preconditioned LSMR
pf.feols("Y ~ X1 | f1 + f2", data=data, demeaner=pf.LsmrDemeaner())

# dense fixed effects: Rust MAP (the default)
pf.feols("Y ~ X1 | f1 + f2", data=data, demeaner=pf.MapDemeaner(backend="rust"))
```

The corresponding `fixest` call in R is near-identical in syntax and, where the
defaults are shared, returns the same point estimates, standard errors and fit
statistics.

# Research impact

The project is developed openly on
[GitHub](https://github.com/py-econometrics/pyfixest), documented at
[pyfixest.org](https://pyfixest.org/), and distributed on PyPI under the MIT
license. More than 50 people have contributed code.

PyPI records more than 820,000 downloads and roughly 78,000 downloads in the
last 30 days [@pepy]. Instacart uses `PyFixest` for fixed-effects regressions in
its experimentation platform, estimating marketplace treatment effects across
high-cardinality geographies [@knight2026instacart]. Google Scholar identifies
about thirty working papers and preprints that name `PyFixest` as part of their
estimation stack. These papers currently cite the GitHub repository because
`PyFixest` has had no citable version of record. This submission and the Zenodo
concept DOI
[10.5281/zenodo.15814089](https://doi.org/10.5281/zenodo.15814089) provide one.

# AI usage disclosure

Much of `PyFixest` predates the routine use of large language models. We have
since used them to assist with coding, refactoring, tests, documentation, and
parts of this manuscript's drafting and copy-editing. A human author reviews
and validates every change; the authors made the econometric decisions. No AI
system is an author of this paper.

# Acknowledgements

We thank Laurent Bergé, Kyle Butts, Grant McDermott, and the `fixest` community.
`PyFixest` follows their API innovations, and we therefore ask users to consider citing `fixest`
[@berge2026fixest] alongside this work. `PyFixest` is MIT-licensed and shares no
source code with `fixest`, which is published under GPL, except for a collinearity-detection routine that
reimplements Bergé's C++ code and is relicensed under MIT with his permission.
The documentation records this and the other packages whose conventions we use
or test against. We also thank all `PyFixest` contributors, the appliedAI
Institute, and the supporters of community sprints and development time.

# References
