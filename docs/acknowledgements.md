# Acknowledgements

PyFixest is a Python package for fast high-dimensional fixed effects regression. Like many open-source software projects, PyFixest builds on work and ideas of many other packages. In this section, we want to acknowledge and express our appreciation for the authors of these packages and their hard work and creativity that PyFixest builds on.

Unless explicitly noted otherwise, all PyFixest code is written independently from scratch. The packages listed below have influenced PyFixest's API design, algorithmic choices, and test validation, but no source code has been copied except where explicitly stated (with license and permission details provided inline).


## fixest (R)

If science is made by "standing on the shoulders of giants", in case of `PyFixest`, there is mostly one giant - Laurent Bergé's formidable fixest R package. `fixest` is so good we decided to stick to its API and conventions as closely as Python allows when starting to work on a fixed effects regression package in Python. Without `fixest`, PyFixest likely wouldn't exist - or at the very least, it would look very different.

We have borrowed the following API conventions and ideas directly from fixest:

| Feature | What PyFixest borrows |
|---|---|
| **Formula syntax** | `feols()`, `fepois()`, `feglm()` function and argument names; the `i()` interaction operator; stepwise `sw()`, `sw0()`, `csw()`, `csw0()` expansion; fixed effects interactions via `fe1^fe2` |
| **Multiple Estimation Optimizations** | The core idea that most of the work of fixed effects regression can be pooled / cached when estimating multiple models with the same fixed effects structure|
| **Demeaning / FWL** | The alternating-projections algorithm in PyFixest is a standalone implementation in numba/rust, but uses the same convergence criteria and default parameters |
| **Small-sample corrections** | All defaults for the `ssc()` function - `adj`, `fixef_K`, `cluster_adj`, `cluster_df` - mirror fixest exactly (see fixest's [standard errors vignette](https://cran.r-project.org/web/packages/fixest/vignettes/standard_errors.html)) |
| **Collinearity detection** | The algorithm is a Python/Numba re-implementation of Laurent Berge's [C++ routine in fixest](https://github.com/lrberge/fixest/blob/a4d1a9bea20aa7ab7ab0e0f1d2047d8097971ad7/src/lm_related.cpp#L130), re-licensed under MIT with Laurent's permission |
| **Post-estimation** | `etable()`, `coefplot()`, `iplot()`, `coef()` etc mirror fixest's output and plotting functionality |
| **On the fly variance covariance adjustments** | As in `fixest`, you can adjust the vcov post estimation by calling a `vcov()` method on the results object |
| **Predict method for fixed effects** | The `predict()`  and `fixef()` methods in PyFixest mirrors fixest's functionality for obtaining fitted values, fixed effects, and linear predictions |

PyFixest is tested against fixest via **rpy2** to ensure numerical equivalence
within machine precision (usually `rtol = 1e-08`, `atol = 1e-08`) for coefficients,
standard errors, t-statistics, p-values, confidence intervals, etc for OLS, IV, Poisson, and GLM models.

---

## By functionality

### Poisson regression

| Package | Language | Role |
|---|---|---|
| [**ppmlhdfe**](https://github.com/sergiocorreia/ppmlhdfe) | Stata | Correia, Guimaraes & Zylkin (2020). PyFixest's Poisson estimator (`fepois`) implements the ppmlhdfe algorithm, including its separation detection and acceleration strategies. Test datasets for separation examples are taken from the ppmlhdfe repository (MIT license) |

### Instrumental variables

| Package | Language | Role |
|---|---|---|
| [**ivDiag**](https://yiqingxu.org/packages/ivDiag/) | R | Tests for weak instrument diagnostics (first-stage F-statistics) are compared against ivDiag |

### Quantile regression

| Package | Language | Role |
|---|---|---|
| [**quantreg**](https://cran.r-project.org/package=quantreg) | R | PyFixest's `quantreg()` implementation is tested against R's quantreg package (by Roger Koenker) for coefficient and NID standard error equivalence |
| [**qreg2**](https://ideas.repec.org/c/boc/bocode/s457369.html) | Stata | Parente & Santos Silva (2016). PyFixest's cluster-robust quantile regression standard errors are tested against Stata's qreg2 output |

### Difference-in-Differences

| Package | Language | Role |
|---|---|---|
| [**did2s**](https://github.com/kylebutts/did2s) | R | PyFixest's DID2S estimator's API is strongly inspired by Kyle Butts' R package (MIT license) and we have relied on Kyle's writeup of the method for our own implementation. Tests compare coefficients and standard errors against the R implementation |
| [**lpdid**](https://github.com/alexCardazzi/lpdid) | R / Stata | PyFixest's local-projections DID estimator is "highly influenced by Alex Cardazzi's R code (published under MIT) for the lpdid package" (source comment in `lpdid.py`). Test data (`.dta` files) and reference values are obtained from both R and Stata runs |

### Panel data visualization

| Package | Language | Role |
|---|---|---|
| [**panelView**](https://yiqingxu.org/packages/panelView/) | R | Yiqing Xu & Licheng Liu. PyFixest's `panelview()` function for visualizing treatment patterns and outcomes in panel data is inspired by the panelView R package |

### Wild cluster bootstrap

| Package | Language | Role |
|---|---|---|
| [**boottest**](https://github.com/droodman/boottest) | Stata | Roodman et al. (2019). The fast wild cluster bootstrap methodology traces back to Roodman's Stata boottest package |
| [**fwildclusterboot**](https://github.com/s3alfisc/fwildclusterboot) | R | An R implementation of the "fast and wild" algorithm by Roodman et al.  |
| [**wildboottest**](https://github.com/s3alfisc/wildboottest) | Python | A python port of `fwildclusterboot`, relying on algorithms developed in MacKinnon, Nielsen and Webb. |

### Randomization inference

| Package | Language | Role |
|---|---|---|
| [**ritest**](https://grantmcdermott.r-universe.dev/ritest) | R | PyFixest's `ritest()` method's API heavily borrows from Grant McDermott's R port and is tested against it. |
| [**ritest**](https://github.com/simonheb/ritest) | Stata | Grant's `ritest` is itself inspired by Simon Heß `ritest` Stata package.|

### Multiple hypothesis testing (Romano-Wolf)

| Package | Language | Role |
|---|---|---|
| [**wildrwolf**](https://s3alfisc.r-universe.dev/wildrwolf) | R | PyFixest's `rwolf()` Romano-Wolf correction is tested against the wildrwolf R package for both HC and CRV inference |
| [**wildwyoung**](https://s3alfisc.r-universe.dev/wildwyoung) | R | An R implementation of the Westfall-Young correction using the wild bootstrap |
| [**rwolf**](https://ideas.repec.org/c/boc/bocode/s458862.html) | Stata | Clarke (2024). Stata implementation of the Romano-Wolf stepdown procedure |
| [**wyoung**](https://ideas.repec.org/c/boc/bocode/s458609.html) | Stata | Jones, Molitor & Reif (2019). Stata implementation of the Westfall-Young stepdown procedure |

### Causal cluster variance

| Package | Language | Role |
|---|---|---|
| [**TSCB-CCV**](https://github.com/Daniel-Pailanir/TSCB-CCV) | Stata | Pailanir & Clarke. PyFixest's CCV implementation (Abadie et al., QJE 2023) is tested against Daniel Pailanir and Damian Clarke's Stata implementation (GPL-3 license). Test data is loaded from Stata `.dta` files |

### Gelbach decomposition

| Package / Author | Language | Role |
|---|---|---|
| **b1x2** | Stata | Gelbach (2016). PyFixest's `decompose()` method is tested against hardcoded results from Gelbach's `b1x2` Stata package.|
| [Apoorva Lal](https://gist.github.com/apoorvalal/e7dc9f3e52dcd9d51854b28b3e8a7ba4) | Python | The initial implementation of Gelbach's decomposition was based on Apoorva's gist |

### Demeaning and fixed effects recovery

| Package | Language | Role |
|---|---|---|
| [**lfe**](https://cran.r-project.org/package=lfe) | R | We based our first implementation of the MAP algorithm on the description in the "how lfe works" vignette and learned how to obtain estimates of swept-out fixed effects post-estimation by using the LSMR solver |
| [**pyhdfe**](https://github.com/jeffgortmaker/pyhdfe) | Python | PyFixest's demeaning results are tested against pyhdfe to ensure equivalence. `pyfixest`'s first MVP ran its demeaning algorithm based on `pyhdfe`. |

---

## Test infrastructure

The following packages are used in PyFixest's test suite to bridge between
Python and R:

| Package | Language | Role |
|---|---|---|
| [**rpy2**](https://rpy2.github.io/) | Python | The bridge between Python and R that powers all cross-language test comparisons |
