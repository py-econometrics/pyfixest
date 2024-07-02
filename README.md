![](figures/pyfixest-logo.png)

# PyFixest: Fast High-Dimensional Fixed Effects Regression in Python

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit)
[![PyPI -Version](https://img.shields.io/pypi/v/pyfixest.svg)](https://pypi.org/project/pyfixest/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyfixest.svg)
[![Downloads](https://static.pepy.tech/badge/pyfixest)](https://pepy.tech/project/pyfixest)
[![Downloads](https://static.pepy.tech/badge/pyfixest/month)](https://pepy.tech/project/pyfixest)
[![image](https://codecov.io/gh/py-econometrics/pyfixest/branch/master/graph/badge.svg)](https://codecov.io/gh/py-econometrics/pyfixest)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)

`PyFixest` is a Python implementation of the formidable [fixest](https://github.com/lrberge/fixest) package for fast high-dimensional fixed effects regression.

The package aims to mimic `fixest` syntax and functionality as closely as Python allows: if you know `fixest` well, the goal is that you won't have to read the docs to get started! In particular, this means that all of `fixest's` defaults are mirrored by `PyFixest` - currently with only [one small exception](https://github.com/py-econometrics/pyfixest/issues/260).

Nevertheless, for a quick introduction, you can take a look at the [documentation](https://py-econometrics.github.io/pyfixest/pyfixest.html) or the regression chapter of [Arthur Turrell's](https://github.com/aeturrell) book on [Coding for Economists](https://aeturrell.github.io/coding-for-economists/econmt-regression.html#imports).

## Features

-   **OLS**, **WLS** and **IV** Regression
-   **Poisson Regression** following the [pplmhdfe algorithm](https://journals.sagepub.com/doi/full/10.1177/1536867X20909691)
-   Multiple Estimation Syntax
-   Several **Robust** and **Cluster Robust Variance-Covariance** Estimators
-   **Wild Cluster Bootstrap** Inference (via
    [wildboottest](https://github.com/py-econometrics/wildboottest))
-   **Difference-in-Differences** Estimators:
    -   The canonical Two-Way Fixed Effects Estimator
    -   [Gardner's two-stage
        ("`Did2s`")](https://jrgcmu.github.io/2sdd_current.pdf)
        estimator
    -   Basic Versions of the Local Projections estimator following
        [Dube et al (2023)](https://www.nber.org/papers/w31184)
- **Multiple Hypothesis Corrections** following the Procedure by [Romano and Wolf](https://journals.sagepub.com/doi/pdf/10.1177/1536867X20976314) and **Simultaneous Confidence Intervals** using a **Multiplier Bootstrap**
- Fast **Randomization Inference** as in the [ritest Stata package](https://hesss.org/ritest.pdf)
- The **Causal Cluster Variance Estimator (CCV)** following [Abadie et al.](https://economics.mit.edu/sites/default/files/2022-09/When%20Should%20You%20Adjust%20Standard%20Errors%20for%20Clustering.pdf)


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

## Benchmarks

All benchmarks follow the [fixest
benchmarks](https://github.com/lrberge/fixest/tree/master/_BENCHMARK).
All non-pyfixest timings are taken from the `fixest` benchmarks.

![](figures/benchmarks_ols.svg)
![](figures/benchmarks_poisson.svg)

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

## Call for Contributions

Thanks for showing interest in contributing to `pyfixest`! We appreciate all
contributions and constructive feedback, whether that be reporting bugs, requesting
new features, or suggesting improvements to documentation.

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
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/styfenschaer"><img src="https://avatars.githubusercontent.com/u/79762922?v=4?s=100" width="100px;" alt="styfenschaer"/><br /><sub><b>styfenschaer</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=styfenschaer" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.nkeleher.com/"><img src="https://avatars.githubusercontent.com/u/5607589?v=4?s=100" width="100px;" alt="Niall Keleher"/><br /><sub><b>Niall Keleher</b></sub></a><br /><a href="#infra-NKeleher" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=NKeleher" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://wenzhi-ding.com"><img src="https://avatars.githubusercontent.com/u/30380959?v=4?s=100" width="100px;" alt="Wenzhi Ding"/><br /><sub><b>Wenzhi Ding</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=Wenzhi-Ding" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://apoorvalal.github.io/"><img src="https://avatars.githubusercontent.com/u/12086926?v=4?s=100" width="100px;" alt="Apoorva Lal"/><br /><sub><b>Apoorva Lal</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=apoorvalal" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Aapoorvalal" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://juanitorduz.github.io"><img src="https://avatars.githubusercontent.com/u/22996444?v=4?s=100" width="100px;" alt="Juan Orduz"/><br /><sub><b>Juan Orduz</b></sub></a><br /><a href="#infra-juanitorduz" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=juanitorduz" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://s3alfisc.github.io/"><img src="https://avatars.githubusercontent.com/u/19531450?v=4?s=100" width="100px;" alt="Alexander Fischer"/><br /><sub><b>Alexander Fischer</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=s3alfisc" title="Code">💻</a> <a href="#infra-s3alfisc" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.aeturrell.com"><img src="https://avatars.githubusercontent.com/u/11294320?v=4?s=100" width="100px;" alt="aeturrell"/><br /><sub><b>aeturrell</b></sub></a><br /><a href="#tutorial-aeturrell" title="Tutorials">✅</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=aeturrell" title="Documentation">📖</a> <a href="#promotion-aeturrell" title="Promotion">📣</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/leostimpfle"><img src="https://avatars.githubusercontent.com/u/31652181?v=4?s=100" width="100px;" alt="leostimpfle"/><br /><sub><b>leostimpfle</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=leostimpfle" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Aleostimpfle" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/baggiponte"><img src="https://avatars.githubusercontent.com/u/57922983?v=4?s=100" width="100px;" alt="baggiponte"/><br /><sub><b>baggiponte</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=baggiponte" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sanskriti2005"><img src="https://avatars.githubusercontent.com/u/150411024?v=4?s=100" width="100px;" alt="Sanskriti"/><br /><sub><b>Sanskriti</b></sub></a><br /><a href="#infra-sanskriti2005" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Jayhyung"><img src="https://avatars.githubusercontent.com/u/40373774?v=4?s=100" width="100px;" alt="Jaehyung"/><br /><sub><b>Jaehyung</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=Jayhyung" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://alexstephenson.me"><img src="https://avatars.githubusercontent.com/u/24926205?v=4?s=100" width="100px;" alt="Alex"/><br /><sub><b>Alex</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=asteves" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/saidamir"><img src="https://avatars.githubusercontent.com/u/20246711?v=4?s=100" width="100px;" alt="Aziz Mamatov"/><br /><sub><b>Aziz Mamatov</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=saidamir" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
