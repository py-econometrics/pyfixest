---
pagetitle: readme
quartodoc:
  package: pyfixest
  sections:
  - contents:
    - estimation.feols
    - estimation.fepois
    - did.estimation.did2s
    - did.estimation.lpdid
    - did.estimation.event_study
    desc: |
      User facing estimation functions
    title: Estimation Functions
  - contents:
    - feols
    - fepois
    - feiv
    desc: |
      Details on Methods and Attributes
    title: Estimation Classes
  - contents:
    - summarize.summary
    - summarize.etable
    - visualize.coefplot
    - visualize.iplot
    desc: |
      Post-Processing of Estimation Results
    title: Summarize and Visualize
  - contents:
    - demean
    - detect_singletons
    - model_matrix_fixest
    desc: |
      PyFixest internals and utilities
    title: Misc / Utilities
  sidebar: \_sidebar.yml
toc-title: Table of contents
website:
  navbar:
    favicon: figures/pyfixest-logo.png
    left:
    - file: readme.qmd
      text: PyFixest
    - file: vignettes/tutorial.ipynb
      text: Quickstart
    - file: vignettes/difference-in-differences.ipynb
      text: Difference-in-Differences Estimation
    - file: vignettes/Replicating-the-Effect.ipynb
      text: Replicating 'The Effect' with PyFixest
    - file: reference/index.qmd
      text: Documentation
    - file: vignettes/news.qmd
      text: Changelog
    page-footer:
      center: |
        Developed by [Alexander Fischer](https://github.com/s3alfisc)
        and [Styfen Schär](https://github.com/styfenschaer)
    page_navigation: true
    right:
    - href: "https://github.com/s3alfisc/pyfixest/"
      icon: github
    search: true
  sidebar:
  - contents:
    - reference/index.qmd
    - contents:
      - reference/estimation.feols.qmd
      - reference/estimation.fepois.qmd
      - reference/did.estimation.did2s.qmd
      - reference/did.estimation.lpdid.qmd
      - reference/did.estimation.event_study.qmd
      section: Estimation Functions
    - contents:
      - reference/feols.qmd
      - reference/fepois.qmd
      - reference/feiv.qmd
      section: Estimation Classes
    - contents:
      - reference/summarize.summary.qmd
      - reference/summarize.etable.qmd
      - reference/visualize.coefplot.qmd
      - reference/visualize.iplot.qmd
      section: Summarize and Visualize
    - contents:
      - reference/demean.qmd
      - reference/detect_singletons.qmd
      - reference/model_matrix_fixest.qmd
      section: Misc / Utilities
    id: reference
    style: floating
  - id: dummy-sidebar
    style: floating
---

![](figures/pyfixest-logo.png)

------------------------------------------------------------------------

# PyFixest: Fast High-Dimensional Fixed Effects Regression in Python

[![PyPI -
Version](https://img.shields.io/pypi/v/pyfixest.svg)](https://pypi.org/project/pyfixest/)
![PyPI - Python
Version](https://img.shields.io/pypi/pyversions/pyfixest.svg) ![PyPI -
Downloads](https://img.shields.io/pypi/dm/pyfixest.png)
[![image](https://codecov.io/gh/s3alfisc/pyfixest/branch/master/graph/badge.svg)](https://codecov.io/gh/s3alfisc/pyfixest)

`PyFixest` is a Python implementation of the formidable
[fixest](https://github.com/lrberge/fixest) package for fast
high-dimensional fixed effects regression. The package aims to mimic
`fixest` syntax and functionality as closely as Python allows: if you
know `fixest` well, the goal is that you won't have to read the docs to
get started! In particular, this means that all of `fixest's` defaults
are mirrored by `PyFixest` - currently with only [one small
exception](https://github.com/s3alfisc/pyfixest/issues/260).
Nevertheless, for a quick introduction, you can take a look at the
[tutorial](https://s3alfisc.github.io/pyfixest/tutorial/) or the
regression chapter of [Arthur Turrell's](https://github.com/aeturrell)
book on [Coding for
Economists](https://aeturrell.github.io/coding-for-economists/econmt-regression.html#imports).

## Features

-   OLS and IV Regression
-   Poisson Regression
-   Multiple Estimation Syntax
-   Several Robust and Cluster Robust Variance-Covariance Types
-   Wild Cluster Bootstrap Inference (via
    [wildboottest](https://github.com/s3alfisc/wildboottest))
-   Difference-in-Difference Estimators:
    -   The canonical Two-Way Fixed Effects Estimator
    -   [Gardner's two-stage
        ("`Did2s`")](https://jrgcmu.github.io/2sdd_current.pdf)
        estimator
    -   Basic Versions of the Local Projections estimator following
        [Dube et al (2023)](https://www.nber.org/papers/w31184)

## Installation

You can install the release version from `PyPi` by running

``` py
pip install pyfixest
```

or the development version from github by running

``` py
pip install git+https://github.com/s3alfisc/pyfixest.git
```

## News

`PyFixest` `0.13` adds support for the [local projections
Difference-in-Differences
Estimator](https://s3alfisc.github.io/pyfixest/difference-in-differences-estimation/).

## Benchmarks

All benchmarks follow the [fixest
benchmarks](https://github.com/lrberge/fixest/tree/master/_BENCHMARK).
All non-pyfixest timings are taken from the `fixest` benchmarks.

![](./benchmarks/lets-plot-images/benchmarks_ols.svg)
![](./benchmarks/lets-plot-images/benchmarks_poisson.svg)

## Quickstart

### Fixed Effects Regression via `feols()`

You can estimate a linear regression models just as you would in
`fixest` - via `feols()`:

::: {.cell execution_count="1"}
``` {.python .cell-code}
from pyfixest.estimation import feols, fepois
from pyfixest.utils import get_data
from pyfixest.summarize import etable

data = get_data()
feols("Y ~ X1 | f1 + f2", data=data).summary()
```

::: {.cell-output .cell-output-display}
```{=html}

            <div id="QL9rAO"></div>
            <script type="text/javascript" data-lets-plot-script="library">
                if(!window.letsPlotCallQueue) {
                    window.letsPlotCallQueue = [];
                }; 
                window.letsPlotCall = function(f) {
                    window.letsPlotCallQueue.push(f);
                };
                (function() {
                    var script = document.createElement("script");
                    script.type = "text/javascript";
                    script.src = "https://cdn.jsdelivr.net/gh/JetBrains/lets-plot@v4.2.0/js-package/distr/lets-plot.min.js";
                    script.onload = function() {
                        window.letsPlotCall = function(f) {f();};
                        window.letsPlotCallQueue.forEach(function(f) {f();});
                        window.letsPlotCallQueue = [];
                        
                    };
                    script.onerror = function(event) {
                        window.letsPlotCall = function(f) {};    // noop
                        window.letsPlotCallQueue = [];
                        var div = document.createElement("div");
                        div.style.color = 'darkred';
                        div.textContent = 'Error loading Lets-Plot JS';
                        document.getElementById("QL9rAO").appendChild(div);
                    };
                    var e = document.getElementById("QL9rAO");
                    e.appendChild(script);
                })()
            </script>
            
```
:::

::: {.cell-output .cell-output-stdout}
    ###

    Estimation:  OLS
    Dep. var.: Y, Fixed effects: f1+f2
    Inference:  CRV1
    Observations:  997

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
    |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
    | X1            |     -0.919 |        0.065 |   -14.057 |      0.000 |  -1.053 |   -0.786 |
    ---
    RMSE: 1.441   R2: 0.609   R2 Within: 0.2
:::
:::

### Multiple Estimation

You can estimate multiple models at once by using [multiple estimation
syntax](https://aeturrell.github.io/coding-for-economists/econmt-regression.html#multiple-regression-models):

::: {.cell execution_count="2"}
``` {.python .cell-code}
# OLS Estimation: estimate multiple models at once
fit = feols("Y + Y2 ~X1 | csw0(f1, f2)", data = data, vcov = {'CRV1':'group_id'})
# Print the results
etable([fit.fetch_model(i) for i in range(6)])
```

::: {.cell-output .cell-output-stdout}
    Model:  Y~X1
    Model:  Y2~X1
    Model:  Y~X1|f1
    Model:  Y2~X1|f1
    Model:  Y~X1|f1+f2
    Model:  Y2~X1|f1+f2
                              est1               est2               est3               est4               est5               est6
    ------------  ----------------  -----------------  -----------------  -----------------  -----------------  -----------------
    depvar                       Y                 Y2                  Y                 Y2                  Y                 Y2
    -----------------------------------------------------------------------------------------------------------------------------
    Intercept     0.919*** (0.121)   1.064*** (0.232)
    X1             -1.0*** (0.117)  -1.322*** (0.211)  -0.949*** (0.087)  -1.266*** (0.212)  -0.919*** (0.069)  -1.228*** (0.194)
    -----------------------------------------------------------------------------------------------------------------------------
    f1                           -                  -                  x                  x                  x                  x
    f2                           -                  -                  -                  -                  x                  x
    -----------------------------------------------------------------------------------------------------------------------------
    R2                       0.123              0.037              0.437              0.115              0.609              0.168
    S.E. type         by: group_id       by: group_id       by: group_id       by: group_id       by: group_id       by: group_id
    Observations               998                999                997                998                997                998
    -----------------------------------------------------------------------------------------------------------------------------
    Significance levels: * p < 0.05, ** p < 0.01, *** p < 0.001
:::
:::

### Adjust Standard Errors "on-the-fly"

Standard Errors can be adjusted after estimation, "on-the-fly":

::: {.cell execution_count="3"}
``` {.python .cell-code}
fit1 = fit.fetch_model(0)
fit1.vcov("hetero").summary()
```

::: {.cell-output .cell-output-stdout}
    Model:  Y~X1
    ###

    Estimation:  OLS
    Dep. var.: Y
    Inference:  hetero
    Observations:  998

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
    |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
    | Intercept     |      0.919 |        0.112 |     8.223 |      0.000 |   0.699 |    1.138 |
    | X1            |     -1.000 |        0.082 |   -12.134 |      0.000 |  -1.162 |   -0.838 |
    ---
    RMSE: 2.158   R2: 0.123
:::
:::

### Poisson Regression via `fepois()`

You can estimate Poisson Regressions via the `fepois()` function:

::: {.cell execution_count="4"}
``` {.python .cell-code}
poisson_data = get_data(model = "Fepois")
fepois("Y ~ X1 + X2 | f1 + f2", data = poisson_data).summary()
```

::: {.cell-output .cell-output-stdout}
    ###

    Estimation:  Poisson
    Dep. var.: Y, Fixed effects: f1+f2
    Inference:  CRV1
    Observations:  997

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
    |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
    | X1            |     -0.008 |        0.035 |    -0.239 |      0.811 |  -0.076 |    0.060 |
    | X2            |     -0.015 |        0.010 |    -1.471 |      0.141 |  -0.035 |    0.005 |
    ---
    Deviance: 1068.836
:::
:::

### IV Estimation via three-part formulas

Last, `PyFixest` also supports IV estimation via three part formula
syntax:

::: {.cell execution_count="5"}
``` {.python .cell-code}
fit_iv = feols("Y ~ 1 | f1 | X1 ~ Z1", data = data)
fit_iv.summary()
```

::: {.cell-output .cell-output-stdout}
    ###

    Estimation:  IV
    Dep. var.: Y, Fixed effects: f1
    Inference:  CRV1
    Observations:  997

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
    |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
    | X1            |     -1.025 |        0.115 |    -8.930 |      0.000 |  -1.259 |   -0.790 |
    ---
:::
:::
