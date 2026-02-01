# Acknowledgements

Like many open-source software projects, PyFixest builds on work and ideas first developed in other packages. In this section, we want to acknowledge and express our appreciation for the authors of these packages and their creativity and hard work.

# Software

> Unless explicitly stated otherwise, all PyFixest code is written independently from scratch. The packages listed below have influenced PyFixest's API design or algorithmic choices, or are used for testing PyFixest as reference implementations, but no source code has been copied except where explicitly stated (with license and permission details provided inline).
## fixest (R)

If open source software is made by "standing on the shoulders of giants", in case of `PyFixest`, there is mostly one very big giant - [Laurent Bergé's](https://sites.google.com/site/laurentrberge/) formidable [fixest](https://github.com/lrberge/fixest/) R package. `fixest` is so good we decided to stick to its API and conventions as closely as Python allows when starting to work on a fixed effects regression package in Python. Without `fixest`, PyFixest likely wouldn't exist - or at the very least, it would look very different. Most importantly, `fixest` has shaped our understanding of what a user-friendly regression package should look like and what functionality it should offer.

More concretely, we have borrowed the following API conventions and ideas directly from fixest:

| Feature | What PyFixest borrows |
|---|---|
| **Formula syntax** | `feols()`, `fepois()`, `feglm()` function and argument names; the `i()` interaction operator; multiple estimation syntax via `sw()`, `sw0()`, `csw()`, `csw0()`; fixed effects interactions via `fe1^fe2` |
| **Multiple Estimation Optimizations** | The core idea that most of the work of fixed effects regression can be pooled / cached when estimating multiple models with the same fixed effects structure|
| **Demeaning / FWL** | The alternating-projections algorithm in PyFixest is a standalone implementation in numba/rust, but uses the same convergence criteria and default parameters |
| **Small-sample corrections** | The `ssc()` function to control small sample adjustments, and all of its defaults - `adj`, `fixef_K`, `cluster_adj`, `cluster_df` - mirror fixest exactly (see fixest's [standard errors vignette](https://cran.r-project.org/web/packages/fixest/vignettes/standard_errors.html)) |
| **Collinearity detection** | The algorithm is a Rust/Numba re-implementation of Laurent Berge's [C++ routine in fixest](https://github.com/lrberge/fixest/blob/a4d1a9bea20aa7ab7ab0e0f1d2047d8097971ad7/src/lm_related.cpp#L130), re-licensed under MIT with Laurent's permission |
| **Post-estimation** | `etable()`, `coefplot()`, `iplot()`, `coef()` etc mirror fixest's output and plotting functionality |
| **On the fly variance covariance adjustments** | As in `fixest`, you can adjust the vcov post estimation by calling a `vcov()` method on the results object (`Feols` in pyfixest and `fixest` in `fixest`) |
| **Predict method for fixed effects** | The `predict()`  and `fixef()` methods in PyFixest mirrors fixest's functionality for obtaining fitted values, fixed effects, and linear predictions |

You can learn more about fixest [on github](https://github.com/lrberge/fixest), via its [documentation](https://lrberge.github.io/fixest/), or by reading the [associated paper](https://arxiv.org/abs/2601.21749). 

PyFixest is tested against fixest via **rpy2** to ensure numerical equivalence
(usually `rtol = 1e-08`, `atol = 1e-08`) for coefficients,
standard errors, t-statistics, p-values, confidence intervals, etc for OLS, IV, Poisson, and GLM models.

---

## By functionality

### Poisson regression

| Package | Language | Role |
|---|---|---|
| [**ppmlhdfe**](https://github.com/sergiocorreia/ppmlhdfe) | Stata |  PyFixest's Poisson estimator (`fepois`) implements the ppmlhdfe algorithm as described in Correia, Guimaraes & Zylkin (2020), including its separation detection and acceleration strategies. Test datasets for separation examples are taken from the ppmlhdfe repository (MIT license) |

### Instrumental variables

| Package | Language | Role |
|---|---|---|
| [**ivDiag**](https://yiqingxu.org/packages/ivDiag/) | R | The IV diagnostics implementations are validated against ivDiag. |

### Quantile regression

| Package | Language | Role |
|---|---|---|
| [**quantreg**](https://cran.r-project.org/package=quantreg) | R | PyFixest's `quantreg()` implementation is tested against R's quantreg package (by Roger Koenker) for coefficient and NID standard error equivalence |
| [**qreg2**](https://ideas.repec.org/c/boc/bocode/s457369.html) | Stata | PyFixest's cluster-robust standard errors for quantile regression are tested against Stata's qreg2 output, which is based on work by Parente & Santos Silva (2016). |

### Difference-in-Differences

| Package | Language | Role |
|---|---|---|
| [**did2s**](https://github.com/kylebutts/did2s) | R | PyFixest's DID2S estimator's API is strongly inspired by Kyle Butts' R package (MIT license) and we have relied on Kyle's writeup of the method for our own implementation. Tests compare coefficients and standard errors against the R implementation |
| [**lpdid**](https://github.com/alexCardazzi/lpdid) | R | PyFixest's local-projections DID estimator is highly influenced by Alex Cardazzi's R code (published under MIT) for the lpdid package. We also test against the R implementation |
| [**lpdid**](https://github.com/danielegirardi/lpdid) | Stata | We also test our implementation against Daniel Busch's and Daniele Girardi's Stata implementation of local-projections DID. |

### Panel data visualization

| Package | Language | Role |
|---|---|---|
| [**panelView**](https://yiqingxu.org/packages/panelView/) | R | PyFixest's `panelview()` function for visualizing treatment patterns and outcomes in panel data is inspired by the panelView R package by Mou, Liu and Xu.|

### Randomization inference

| Package | Language | Role |
|---|---|---|
| [**ritest**](https://github.com/grantmcdermott/ritest) | R | PyFixest's `ritest()` method's API heavily borrows from Grant McDermott's R port and is tested against it. |
| [**ritest**](https://github.com/simonheb/ritest) | Stata | Grant's `ritest` is itself inspired by Simon Heß `ritest` Stata package.|

### Wild cluster bootstrap

| Package | Language | Role |
|---|---|---|
| [**wildboottest**](https://github.com/py-econometrics/wildboottest) | Python | PyFixest loads classes from `wildboottest` to run wild bootstrap inference. `wildboottest` is a Python port of `fwildclusterboot`. |
| [**fwildclusterboot**](https://github.com/s3alfisc/fwildclusterboot) | R | An R implementation of the "fast and wild" algorithm by Roodman et al.  |
| [**boottest**](https://github.com/droodman/boottest) | Stata | Roodman et al. (2019). The fast wild cluster bootstrap methodology traces back to Roodman's Stata boottest package |

### Multiple hypothesis testing (Romano-Wolf)

| Package | Language | Role |
|---|---|---|
| [**wildrwolf**](https://github.com/s3alfisc/wildrwolf) | R | PyFixest's `rwolf()` Romano-Wolf correction is tested against the wildrwolf R package for both HC and CRV inference. |
| [**wildwyoung**](https://github.com/s3alfisc/wildwyoung) | R | An R implementation of the Westfall-Young correction using the wild bootstrap |
| [**rwolf**](https://github.com/damiancclarke/rwolf) | Stata | A Stata implementation of the Romano-Wolf stepdown procedure that inspired development of `rwolf`. |
| [**wyoung**](https://github.com/reifjulian/wyoung) | Stata | A Stata implementation of the Westfall-Young stepdown procedure by Jones, Molitor & Reif.|

### Causal cluster variance

| Package | Language | Role |
|---|---|---|
| [**TSCB-CCV**](https://github.com/Daniel-Pailanir/TSCB-CCV) | Stata | Pailanir & Clarke. PyFixest's CCV implementation (Abadie et al., QJE 2023) is tested against Daniel Pailanir and Damian Clarke's Stata implementation. Test data is loaded from Stata `.dta` files |

### Gelbach decomposition

| Package / Author | Language | Role |
|---|---|---|
| [**b1x2**](https://ideas.repec.org/c/boc/bocode/s457814.html)| Stata |PyFixest's `decompose()` method is tested against hardcoded results from Gelbach's `b1x2` Stata package.|
| [Apoorva's Linear Mediation Gist](https://gist.github.com/apoorvalal/e7dc9f3e52dcd9d51854b28b3e8a7ba4) | Python | The initial implementation of Gelbach's decomposition was based on Apoorva's gist |

### Demeaning and fixed effects recovery

| Package | Language | Role |
|---|---|---|
| [**lfe**](https://cran.r-project.org/web/packages/lfe/vignettes/lfehow.pdf) | R | We based our first implementation of the MAP algorithm on the description in the "how lfe works" vignette. |
| [**pyhdfe**](https://github.com/jeffgortmaker/pyhdfe) | Python | PyFixest's demeaning results are tested against pyhdfe to ensure equivalence. `pyfixest`'s first MVP was built using `pyhdfe` it ran its demeaning algorithm via `pyhdfe` MAP algo. |

---

## Test infrastructure

The following packages are used in PyFixest's test suite to bridge between
Python and R:

| Package | Language | Role |
|---|---|---|
| [**rpy2**](https://rpy2.github.io/) | Python | The bridge between Python and R that powers all cross-language test comparisons |

## Other Software

Here we list other foundational software without which a project like `PyFixest` would not be possible:

- `formulaic`
- `numpy`
- `numba`
- `pandas`
- `scipy`
- `matplotlib`
- `great-tables` / `maketables`
- `pyo3`

# Papers and Algorithms

- Bergé, L. R., Butts, K., & McDermott, G. (2026). "Fast and user-friendly econometrics estimations: The R package fixest." [arXiv:2601.21749](https://arxiv.org/abs/2601.21749).
- Correia, S., Guimarães, P., & Zylkin, T. (2020). "ppmlhdfe: Fast Poisson estimation with high-dimensional fixed effects." *The Stata Journal*, 20(1). [arXiv:1903.01690](https://arxiv.org/abs/1903.01690).
- Gaure, S. (2013). "OLS with multiple high dimensional category variables." *Computational Statistics & Data Analysis*, 66, 8-18. [Vignette: How lfe works](https://cran.r-project.org/web/packages/lfe/vignettes/lfehow.pdf).
- Guimarães, P. & Portugal, P. (2010). "A simple feasible procedure to fit models with high-dimensional fixed effects." *The Stata Journal*, 10(4), 628-649. [DOI:10.1177/1536867X1101000406](https://doi.org/10.1177/1536867X1101000406).
- Koenker, R. & Ng, P. (2005). "A Frisch-Newton Algorithm for Sparse Quantile Regression." *Acta Mathematicae Applicatae Sinica*, 21(2), 225-236. [DOI:10.1007/s10255-005-0231-1](https://doi.org/10.1007/s10255-005-0231-1).
