![](figures/pyfixest-logo.png)

# PyFixest: Fast High-Dimensional Fixed Effects Regression in Python

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/license/mit)
![Python Versions](https://img.shields.io/badge/Python-3.9–3.14-blue)
[![PyPI -Version](https://img.shields.io/pypi/v/pyfixest.svg)](https://pypi.org/project/pyfixest/)
[![Project Chat][chat-badge]][chat-url]
[![image](https://codecov.io/gh/py-econometrics/pyfixest/branch/master/graph/badge.svg)](https://codecov.io/gh/py-econometrics/pyfixest)
[![Known Bugs](https://img.shields.io/github/issues/py-econometrics/pyfixest/bug?color=red&label=Bugs)](https://github.com/py-econometrics/pyfixest/issues?q=is%3Aissue+is%3Aopen+label%3Abug)
[![File an Issue](https://img.shields.io/github/issues/py-econometrics/pyfixest)](https://github.com/py-econometrics/pyfixest/issues)
[![Downloads](https://static.pepy.tech/badge/pyfixest)](https://pepy.tech/project/pyfixest)
[![Downloads](https://static.pepy.tech/badge/pyfixest/month)](https://pepy.tech/project/pyfixest)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pixi Badge][pixi-badge]][pixi-url]
[![Open in Codespaces](https://img.shields.io/badge/Open%20in-Codespaces-blue?logo=github)](https://github.com/py-econometrics/pyfixest/codespaces)
[![Donate | GiveDirectly](https://img.shields.io/static/v1?label=GiveDirectly&message=Donate&color=blue&style=flat-square)](https://github.com/py-econometrics/pyfixest?tab=readme-ov-file#support-pyfixest)
[![PyPI](https://img.shields.io/pypi/v/pyfixest)](https://pypi.org/project/pyfixest)
[![Citation](https://img.shields.io/badge/Cite%20as-PyFixest-blue)](https://github.com/py-econometrics/pyfixest?tab=readme-ov-file#how-to-cite)

[pixi-badge]:https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square
[pixi-url]: https://pixi.sh

[chat-badge]: https://img.shields.io/discord/1259933360726216754.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2&style=flat-square
[chat-url]: https://discord.gg/gBAydeDMVK

[Docs](https://py-econometrics.github.io/pyfixest/pyfixest.html) · [Function & API Reference](https://py-econometrics.github.io/pyfixest/reference/) · [DeepWiki](https://deepwiki.com/py-econometrics/pyfixest) · [Report Bugs & Request Features](https://github.com/py-econometrics/pyfixest/issues) · [Changelog](https://py-econometrics.github.io/pyfixest/changelog.html)

`PyFixest` is a Python package for fast high-dimensional fixed effects regression.

The package aims to mimic the syntax and functionality of the formidable [fixest](https://github.com/lrberge/fixest) package as closely as Python allows: if you know `fixest` well, the goal is that you won't have to read the docs to get started! In particular, this means that all of `fixest's` defaults are mirrored by `PyFixest`.

For a quick introduction, you can take a look at the [quickstart](https://py-econometrics.github.io/pyfixest/quickstart.html) or the regression chapter of [Arthur Turrell's](https://github.com/aeturrell) book on [Coding for Economists](https://aeturrell.github.io/coding-for-economists/econmt-regression.html#imports). You can find documentation of all user facing functions in the [Function Reference](https://py-econometrics.github.io/pyfixest/reference/) section of the [documentation](https://py-econometrics.github.io/pyfixest/pyfixest.html).

For questions on `PyFixest`, head on over to our [github discussions](https://github.com/py-econometrics/pyfixest/discussions), or (more informally) join our [Discord server](https://discord.gg/gBAydeDMVK).

## Support PyFixest

If you enjoy using `PyFixest`, please consider donating to [GiveDirectly](https://donate.givedirectly.org/dedicate) and dedicating your donation to `pyfixest.dev@gmail.com`.
You can also leave a message through the donation form - your support and encouragement mean a lot to the developers!

## Features

-   **OLS**, **WLS** and **IV** Regression with Fixed-Effects Demeaning via [Frisch-Waugh-Lovell](https://bookdown.org/ts_robinson1994/10EconometricTheorems/frisch.html)
-   **Poisson Regression** following the [pplmhdfe algorithm](https://journals.sagepub.com/doi/full/10.1177/1536867X20909691)
-   Probit, Logit and Gaussian Family **GLMs** (currently without fixed effects demeaning, this is WIP)
-   **Quantile Regression** using an Interior Point Solver
-   Multiple Estimation Syntax
-   Several **Robust**, **Cluster Robust** and **HAC Variance-Covariance** Estimators
-   **Wild Cluster Bootstrap** Inference (via
    [wildboottest](https://github.com/py-econometrics/wildboottest))
-   **Difference-in-Differences** Estimators:
    -   The canonical Two-Way Fixed Effects Estimator
    -   [Gardner's two-stage
        ("`Did2s`")](https://jrgcmu.github.io/2sdd_current.pdf)
        estimator
    -   Basic Versions of the Local Projections estimator following
        [Dube et al (2023)](https://www.nber.org/papers/w31184)
    - The fully saturated Event-Study estimator following [Sun & Abraham (2021)](https://www.sciencedirect.com/science/article/abs/pii/S030440762030378X)
- **Multiple Hypothesis Corrections** following the Procedure by [Romano and Wolf](https://journals.sagepub.com/doi/pdf/10.1177/1536867X20976314) and **Simultaneous Confidence Intervals** using a **Multiplier Bootstrap**
- Fast **Randomization Inference** as in the [ritest Stata package](https://hesss.org/ritest.pdf)
- The **Causal Cluster Variance Estimator (CCV)** following [Abadie et al.](https://economics.mit.edu/sites/default/files/2022-09/When%20Should%20You%20Adjust%20Standard%20Errors%20for%20Clustering.pdf)
- Regression **Decomposition** following [Gelbach (2016)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1425737)
- **Publication-ready tables** with [Great Tables](https://posit-dev.github.io/great-tables/articles/intro.html) or LaTex booktabs


## Installation

You can install the release version from `PyPI` by running

```py
# inside an active virtual environment
python -m pip install pyfixest
```

or the development version from github by running

```py
python -m pip install git+https://github.com/py-econometrics/pyfixest
```

For visualization features using the lets-plot backend, install the optional dependency:

```py
python -m pip install pyfixest[plots]
```

Note that matplotlib is included by default, so you can always use the matplotlib backend for plotting even without installing the optional lets-plot dependency.

### GPU Acceleration (Optional)

PyFixest supports GPU-accelerated fixed effects demeaning via CuPy. To enable GPU acceleration, install CuPy matching your CUDA version:

```bash
# For CUDA 11.x, 12.x, 13.x
pip install cupy-cuda11x
pip install cupy-cuda12x
pip install cupy-cuda13x
```

Once installed, you can use GPU-accelerated demeaning by setting the `demean_backend` parameter:

```python
# Use GPU with float32 and float64 precision
pf.feols("Y ~ X1 | f1 + f2", data=data, demean_backend="cupy32")
pf.feols("Y ~ X1 | f1 + f2", data=data, demean_backend="cupy64")
```

## Benchmarks

All benchmarks follow the [fixest
benchmarks](https://github.com/lrberge/fixest/tree/master/_BENCHMARK).
All non-pyfixest timings are taken from the `fixest` benchmarks.

![](benchmarks/lets-plot-images/benchmarks_ols.svg)
![](benchmarks/lets-plot-images/benchmarks_poisson.svg)
![](benchmarks/quantreg_benchmarks.png)


## Quickstart


```python
import pyfixest as pf

data = pf.get_data()
pf.feols("Y ~ X1 | f1 + f2", data=data).summary()
```

    ###

    Estimation:  OLS
    Dep. var.: Y, Fixed effects: f1+f2
    Inference:  CRV1
    Observations:  997

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | X1            |     -0.919 |        0.065 |   -14.057 |      0.000 | -1.053 |  -0.786 |
    ---
    RMSE: 1.441   R2: 0.609   R2 Within: 0.2


### Multiple Estimation

You can estimate multiple models at once by using [multiple estimation
syntax](https://aeturrell.github.io/coding-for-economists/econmt-regression.html#multiple-regression-models):



```python
# OLS Estimation: estimate multiple models at once
fit = pf.feols("Y + Y2 ~X1 | csw0(f1, f2)", data = data, vcov = {'CRV1':'group_id'})
# Print the results
fit.etable()
```

                               est1               est2               est3               est4               est5               est6
    ------------  -----------------  -----------------  -----------------  -----------------  -----------------  -----------------
    depvar                        Y                 Y2                  Y                 Y2                  Y                 Y2
    ------------------------------------------------------------------------------------------------------------------------------
    Intercept      0.919*** (0.121)   1.064*** (0.232)
    X1            -1.000*** (0.117)  -1.322*** (0.211)  -0.949*** (0.087)  -1.266*** (0.212)  -0.919*** (0.069)  -1.228*** (0.194)
    ------------------------------------------------------------------------------------------------------------------------------
    f2                            -                  -                  -                  -                  x                  x
    f1                            -                  -                  x                  x                  x                  x
    ------------------------------------------------------------------------------------------------------------------------------
    R2                        0.123              0.037              0.437              0.115              0.609              0.168
    S.E. type          by: group_id       by: group_id       by: group_id       by: group_id       by: group_id       by: group_id
    Observations                998                999                997                998                997                998
    ------------------------------------------------------------------------------------------------------------------------------
    Significance levels: * p < 0.05, ** p < 0.01, *** p < 0.001
    Format of coefficient cell:
    Coefficient (Std. Error)




### Adjust Standard Errors "on-the-fly"

Standard Errors can be adjusted after estimation, "on-the-fly":


```python
fit1 = fit.fetch_model(0)
fit1.vcov("hetero").summary()
```

    Model:  Y~X1
    ###

    Estimation:  OLS
    Dep. var.: Y
    Inference:  hetero
    Observations:  998

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | Intercept     |      0.919 |        0.112 |     8.223 |      0.000 |  0.699 |   1.138 |
    | X1            |     -1.000 |        0.082 |   -12.134 |      0.000 | -1.162 |  -0.838 |
    ---
    RMSE: 2.158   R2: 0.123


### Poisson Regression via `fepois()`

You can estimate Poisson Regressions via the `fepois()` function:


```python
poisson_data = pf.get_data(model = "Fepois")
pf.fepois("Y ~ X1 + X2 | f1 + f2", data = poisson_data).summary()
```

    ###

    Estimation:  Poisson
    Dep. var.: Y, Fixed effects: f1+f2
    Inference:  CRV1
    Observations:  997

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | X1            |     -0.007 |        0.035 |    -0.190 |      0.850 | -0.075 |   0.062 |
    | X2            |     -0.015 |        0.010 |    -1.449 |      0.147 | -0.035 |   0.005 |
    ---
    Deviance: 1068.169


### IV Estimation via three-part formulas

Last, `PyFixest` also supports IV estimation via three part formula
syntax:


```python
fit_iv = pf.feols("Y ~ 1 | f1 | X1 ~ Z1", data = data)
fit_iv.summary()
```

    ###

    Estimation:  IV
    Dep. var.: Y, Fixed effects: f1
    Inference:  CRV1
    Observations:  997

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | X1            |     -1.025 |        0.115 |    -8.930 |      0.000 | -1.259 |  -0.790 |
    ---

## Quantile Regression via `pf.quantreg`

```python
fit_qr = pf.quantreg("Y ~ X1 + X2", data = data, quantile = 0.5)
```

## Call for Contributions

Thanks for showing interest in contributing to `pyfixest`! We appreciate all
contributions and constructive feedback, whether that be reporting bugs, requesting
new features, or suggesting improvements to documentation.

**Upcoming:** We're hosting a [PyFixest Sprint in Heilbronn](https://py-econometrics.github.io/pyfixest/pyfixest-sprint.html) with [AppliedAI](https://www.appliedai.de/) in late February/early March 2026. Interested in joining? [Learn more and get in touch](https://py-econometrics.github.io/pyfixest/pyfixest-sprint.html).

If you'd like to get involved, but are not yet sure how, please feel free to send us an [email](alexander-fischer1801@t-online.de). Some familiarity with
either Python or econometrics will help, but you really don't need to be a `numpy` core developer or have published in [Econometrica](https://onlinelibrary.wiley.com/journal/14680262) =) We'd be more than happy to invest time to help you get started!

## Contributors ✨

Thanks goes to these wonderful people:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/styfenschaer"><img src="https://avatars.githubusercontent.com/u/79762922?v=4?s=80" width="80px;" alt="styfenschaer"/><br /><sub><b>styfenschaer</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=styfenschaer" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://www.nkeleher.com/"><img src="https://avatars.githubusercontent.com/u/5607589?v=4?s=80" width="80px;" alt="Niall Keleher"/><br /><sub><b>Niall Keleher</b></sub></a><br /><a href="#infra-NKeleher" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=NKeleher" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://wenzhi-ding.com"><img src="https://avatars.githubusercontent.com/u/30380959?v=4?s=80" width="80px;" alt="Wenzhi Ding"/><br /><sub><b>Wenzhi Ding</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=Wenzhi-Ding" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://apoorvalal.github.io/"><img src="https://avatars.githubusercontent.com/u/12086926?v=4?s=80" width="80px;" alt="Apoorva Lal"/><br /><sub><b>Apoorva Lal</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=apoorvalal" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Aapoorvalal" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://juanitorduz.github.io"><img src="https://avatars.githubusercontent.com/u/22996444?v=4?s=80" width="80px;" alt="Juan Orduz"/><br /><sub><b>Juan Orduz</b></sub></a><br /><a href="#infra-juanitorduz" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=juanitorduz" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://s3alfisc.github.io/"><img src="https://avatars.githubusercontent.com/u/19531450?v=4?s=80" width="80px;" alt="Alexander Fischer"/><br /><sub><b>Alexander Fischer</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=s3alfisc" title="Code">💻</a> <a href="#infra-s3alfisc" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://www.aeturrell.com"><img src="https://avatars.githubusercontent.com/u/11294320?v=4?s=80" width="80px;" alt="aeturrell"/><br /><sub><b>aeturrell</b></sub></a><br /><a href="#tutorial-aeturrell" title="Tutorials">✅</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=aeturrell" title="Documentation">📖</a> <a href="#promotion-aeturrell" title="Promotion">📣</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/leostimpfle"><img src="https://avatars.githubusercontent.com/u/31652181?v=4?s=80" width="80px;" alt="leostimpfle"/><br /><sub><b>leostimpfle</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=leostimpfle" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Aleostimpfle" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/baggiponte"><img src="https://avatars.githubusercontent.com/u/57922983?v=4?s=80" width="80px;" alt="baggiponte"/><br /><sub><b>baggiponte</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=baggiponte" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/sanskriti2005"><img src="https://avatars.githubusercontent.com/u/150411024?v=4?s=80" width="80px;" alt="Sanskriti"/><br /><sub><b>Sanskriti</b></sub></a><br /><a href="#infra-sanskriti2005" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/Jayhyung"><img src="https://avatars.githubusercontent.com/u/40373774?v=4?s=80" width="80px;" alt="Jaehyung"/><br /><sub><b>Jaehyung</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=Jayhyung" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://alexstephenson.me"><img src="https://avatars.githubusercontent.com/u/24926205?v=4?s=80" width="80px;" alt="Alex"/><br /><sub><b>Alex</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=asteves" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/greenguy33"><img src="https://avatars.githubusercontent.com/u/8525718?v=4?s=80" width="80px;" alt="Hayden Freedman"/><br /><sub><b>Hayden Freedman</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=greenguy33" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=greenguy33" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/saidamir"><img src="https://avatars.githubusercontent.com/u/20246711?v=4?s=80" width="80px;" alt="Aziz Mamatov"/><br /><sub><b>Aziz Mamatov</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=saidamir" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/rafimikail"><img src="https://avatars.githubusercontent.com/u/61386867?v=4?s=80" width="80px;" alt="rafimikail"/><br /><sub><b>rafimikail</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=rafimikail" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://www.linkedin.com/in/benjamin-knight/"><img src="https://avatars.githubusercontent.com/u/12180931?v=4?s=80" width="80px;" alt="Benjamin Knight"/><br /><sub><b>Benjamin Knight</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=b-knight" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://dsliwka.github.io/"><img src="https://avatars.githubusercontent.com/u/49401450?v=4?s=80" width="80px;" alt="Dirk Sliwka"/><br /><sub><b>Dirk Sliwka</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=dsliwka" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=dsliwka" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/daltonm-bls"><img src="https://avatars.githubusercontent.com/u/78225214?v=4?s=80" width="80px;" alt="daltonm-bls"/><br /><sub><b>daltonm-bls</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Adaltonm-bls" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/marcandre259"><img src="https://avatars.githubusercontent.com/u/19809475?v=4?s=80" width="80px;" alt="Marc-André"/><br /><sub><b>Marc-André</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=marcandre259" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Amarcandre259" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/kylebutts"><img src="https://avatars.githubusercontent.com/u/19961439?v=4?s=80" width="80px;" alt="Kyle F Butts"/><br /><sub><b>Kyle F Butts</b></sub></a><br /><a href="#data-kylebutts" title="Data">🔣</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://fosstodon.org/@marcogorelli"><img src="https://avatars.githubusercontent.com/u/33491632?v=4?s=80" width="80px;" alt="Marco Edward Gorelli"/><br /><sub><b>Marco Edward Gorelli</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/pulls?q=is%3Apr+reviewed-by%3AMarcoGorelli" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://arelbundock.com"><img src="https://avatars.githubusercontent.com/u/987057?v=4?s=80" width="80px;" alt="Vincent Arel-Bundock"/><br /><sub><b>Vincent Arel-Bundock</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=vincentarelbundock" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/IshwaraHegde97"><img src="https://avatars.githubusercontent.com/u/187858441?v=4?s=80" width="80px;" alt="IshwaraHegde97"/><br /><sub><b>IshwaraHegde97</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=IshwaraHegde97" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/RoyalTS"><img src="https://avatars.githubusercontent.com/u/702580?v=4?s=80" width="80px;" alt="Tobias Schmidt"/><br /><sub><b>Tobias Schmidt</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=RoyalTS" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/escherpf"><img src="https://avatars.githubusercontent.com/u/3789736?v=4?s=80" width="80px;" alt="escherpf"/><br /><sub><b>escherpf</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Aescherpf" title="Bug reports">🐛</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=escherpf" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://www.ivanhigueram.com"><img src="https://avatars.githubusercontent.com/u/9004403?v=4?s=80" width="80px;" alt="Iván Higuera Mendieta"/><br /><sub><b>Iván Higuera Mendieta</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=ivanhigueram" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/adamvig96"><img src="https://avatars.githubusercontent.com/u/52835042?v=4?s=80" width="80px;" alt="Ádám Vig"/><br /><sub><b>Ádám Vig</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=adamvig96" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://szymon.info"><img src="https://avatars.githubusercontent.com/u/13162607?v=4?s=80" width="80px;" alt="Szymon Sacher"/><br /><sub><b>Szymon Sacher</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=elchorro" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/AronNemeth"><img src="https://avatars.githubusercontent.com/u/96979880?v=4?s=80" width="80px;" alt="AronNemeth"/><br /><sub><b>AronNemeth</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=AronNemeth" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/DTchebotarev"><img src="https://avatars.githubusercontent.com/u/6100882?v=4?s=80" width="80px;" alt="Dmitri Tchebotarev"/><br /><sub><b>Dmitri Tchebotarev</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=DTchebotarev" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/FuZhiyu"><img src="https://avatars.githubusercontent.com/u/11523699?v=4?s=80" width="80px;" alt="FuZhiyu"/><br /><sub><b>FuZhiyu</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3AFuZhiyu" title="Bug reports">🐛</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=FuZhiyu" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://www.marceloortizm.com"><img src="https://avatars.githubusercontent.com/u/70643379?v=4?s=80" width="80px;" alt="Marcelo Ortiz M."/><br /><sub><b>Marcelo Ortiz M.</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=mortizm1988" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/jestover"><img src="https://avatars.githubusercontent.com/u/11298484?v=4?s=80" width="80px;" alt="Joseph Stover"/><br /><sub><b>Joseph Stover</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=jestover" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/JaapCTJ"><img src="https://avatars.githubusercontent.com/u/157970664?v=4?s=80" width="80px;" alt="JaapCTJ"/><br /><sub><b>JaapCTJ</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=JaapCTJ" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://shapiromh.com"><img src="https://avatars.githubusercontent.com/u/2327833?v=4?s=80" width="80px;" alt="Matt Shapiro"/><br /><sub><b>Matt Shapiro</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=shapiromh" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/schroedk"><img src="https://avatars.githubusercontent.com/u/38397037?v=4?s=80" width="80px;" alt="Kristof Schröder"/><br /><sub><b>Kristof Schröder</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=schroedk" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/WiktorTheScriptor"><img src="https://avatars.githubusercontent.com/u/162815872?v=4?s=80" width="80px;" alt="Wiktor "/><br /><sub><b>Wiktor </b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=WiktorTheScriptor" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://damandhaliwal.me"><img src="https://avatars.githubusercontent.com/u/48694192?v=4?s=80" width="80px;" alt="Daman Dhaliwal"/><br /><sub><b>Daman Dhaliwal</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=damandhaliwal" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://markkaj.github.io/"><img src="https://avatars.githubusercontent.com/u/24573803?v=4?s=80" width="80px;" alt="Jaakko Markkanen"/><br /><sub><b>Jaakko Markkanen</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Amarkkaj" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://jsr-p.github.io/"><img src="https://avatars.githubusercontent.com/u/49307119?v=4?s=80" width="80px;" alt="Jonas Skjold Raaschou-Pedersen"/><br /><sub><b>Jonas Skjold Raaschou-Pedersen</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=jsr-p" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=jsr-p" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="http://bobbyho.me"><img src="https://avatars.githubusercontent.com/u/10739091?v=4?s=80" width="80px;" alt="Bobby Ho"/><br /><sub><b>Bobby Ho</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=bobby1030" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/Erica-Ryan"><img src="https://avatars.githubusercontent.com/u/27149581?v=4?s=80" width="80px;" alt="Erica Ryan"/><br /><sub><b>Erica Ryan</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=Erica-Ryan" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/souhil25"><img src="https://avatars.githubusercontent.com/u/185626340?v=4?s=80" width="80px;" alt="Souhil Abdelmalek Louddad"/><br /><sub><b>Souhil Abdelmalek Louddad</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=souhil25" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://dpananos.github.io"><img src="https://avatars.githubusercontent.com/u/20362941?v=4?s=80" width="80px;" alt="Demetri Pananos"/><br /><sub><b>Demetri Pananos</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=Dpananos" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/pdoupe"><img src="https://avatars.githubusercontent.com/u/215010049?v=4?s=80" width="80px;" alt="Patrick Doupe"/><br /><sub><b>Patrick Doupe</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=pdoupe" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://www.linkedin.com/in/ariadnaaz/"><img src="https://avatars.githubusercontent.com/u/35080247?v=4?s=80" width="80px;" alt="Ariadna Albors Zumel"/><br /><sub><b>Ariadna Albors Zumel</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=Ariadnaaz" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/blucap"><img src="https://avatars.githubusercontent.com/u/614800?v=4?s=80" width="80px;" alt="Martien Lubberink"/><br /><sub><b>Martien Lubberink</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=blucap" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/jonpcohen"><img src="https://avatars.githubusercontent.com/u/48955223?v=4?s=80" width="80px;" alt="jonpcohen"/><br /><sub><b>jonpcohen</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=jonpcohen" title="Documentation">📖</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Acknowledgements

We thank all institutions that have funded or supported work on PyFixest!

<img src="figures/aai-institute-logo.svg" width="185">

## How to Cite

If you want to cite PyFixest, you can use the following BibTeX entry:

```bibtex
@software{pyfixest,
  author  = {{The PyFixest Authors}},
  title   = {{pyfixest: Fast high-dimensional fixed effect estimation in Python}},
  year    = {2025},
  url     = {https://github.com/py-econometrics/pyfixest}
}
```
