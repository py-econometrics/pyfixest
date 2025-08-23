---
title: 'PyFixest: A Python Port of fixest for High-Dimensional Fixed-Effects Regression'
date: 27 April 2025
author:
  - name: Alexander Fischer
    affiliation: Trivago
---


## Summary

PyFixest is an open-source Python library that implements efficient routines for regression analysis with multiple potentially high-dimensional fixed effects by applying the Frisch-Waugh-Lovell theorem. It is a faithful port of the R package fixest [@berge2018], aiming to replicate fixest’s core design principles and functionality within the Python ecosystem. Users familiar with fixest in R can seamlessly transition their analysis to Python using the same syntax and obtaining identical results. Likewise, users of PyFixest should find it easy to port their analysis to R and use the fixest package.

## Statement of Need

Fixed-effects models with high-dimensional categorical effects are ubiquitous in the social sciences, where researchers often need to control for multiple levels of unobserved heterogeneity. To efficiently estimate such models, researchers typically rely on specialized software that handle the computational challenges associated with high-dimensional fixed effects via the application of the Frisch-Waugh-Lowell (FWL) theorem (@correia2023reghdfe and correia2020fast in Stata, @gaure2013lfe, @berge2018 and @stammann2020package in R, @fixedeffectmodelsjl_2025 in Julia). In practice, the FWL theorem is efficiently implemented by an iterative "within-transformation" (demeaning) approach, which avoids the need to create large matrices of dummy variables, which can be computationally expensive and memory-intensive, especially with large datasets and/or many fixed effects [see @gaure2013ols for details].

The Python eco-system lacks an optimized library for this purpose, and *PyFixest* aims to fill this gap. *statsmodels*, the dominant statistical regression library in Python [seabold2010statsmodels] does not provide OLS and GLM functionality that natively handles multiple group fixed effects in an optimized and efficient way via the FWL theorem. Instead, users either have to inefficiently one-hot encode all categorical features in large dummy matrices, or have to rely on a workaround that involves demeaning the design matrix and outcome variable via the *pyhdfe* library (@gortmaker_pyhdfe_2023) before passing the transformed data to *statsmodels* OLS class for fitting the model. GLM estimation is generally not supported via this two-step procedure, as the iterated least squares (ILS) approach used to fit such models requires that design matrix and outcome variable are demeand in every iteration (@correia2020fast, stammann2017fast). Another package, *linearmodels*  provides a class for estimating linear models with multiple fixed effects, *AbsorbingLS*, which calls *pyhdfe* to perform the demeaning step.

By contrast, the R community has benefited from specialized tools like *lfe* [@gaure2013lfe] and *fixest* [@berge2018] for high-dimensional fixed effects regression. In particular, *fixest* has introduced extremely performant and user-friendly regression software to handle multiple (potentially high-dimensional) fixed effects. Relative to *pyhdfe*'s algorithms, which are all implemented via *numpy*, the demeaning algorithm of *fixest* is orders of magnitudes faster. Beyond computational efficiency, *fixest* has introduced a rich set of features for post-estimation analysis, including methods to easily summarize and plot regression results.

*PyFixest* aims to faithfully implement *fixest*'s core functionality - efficient routines for OLS, IV, and Poisson regression with fixed effects - syntax, and post-estimation functionality. Identical input arguments to the main estimation functions that both packages share - *feols*, *fepois* and *feglm* - should produce identical results in R and Python. All of *fixest*'s core defaults, including the choice of variance covariance matrices, small sample corrections, handling of singleton fixed effects, and the treatment of multicollinear variables are preserved by *PyFixest*. To ensure identical behavior, both libraries are thoroughly tested against each other using rpy2 (gautier2008rpy2).

In addition to supporting fixed-effects and instrumental variable estimation, *PyFixest* provides support for regression weights, fast poisson regression with fixed effect demeaning (@correia2020fast), quantile regression (@koenker2001quantile) and a range of modern event study estimators, including the linear projections approach (@dube2023local, @busch2023lpdid), the two-stage difference-in-differences imputation estimator (@gardner2022two, @butts2021did2s), and the fully-saturated event study estimator proposed by @sun2021estimating. The package provides comprehensive options for calculating non-standard inference, including cluster robust variance estimators (CRV1 and CRV3, see @mackinnon2023fast), wild cluster bootstrap (@roodman2019fast, @fischer2022fwildclusterboot), randomization inference (@hess2017randomization), and the causal cluster variance estimator (@abadie2023should). PyFixest also implements methods to control the family-wise error rate by implementing the Romano-Wolf correction (@clarke2020romano and @romano2005exact), and enables users to compute simultaneous confidence bands through a multiplier bootstrap approach (@montiel2019simultaneous). Additionally, *PyFixest* provides support for Gelbach's regression decomposition (@gelbach2016covariates), Lal's event study specification test (@lal2025can), and estimation strategies based on compression & sufficient statistics as described in
@wong2021you and @lal2024large.*PyFixest* also provides multiple diagnostics for weak instruments (@lal2023much). To the best of our knowledge, most of these methods are only implemented in *PyFixest* within the Python ecosystem.

Below, we provide a brief code example that showcases that PyFixest's and fixest's syntax is nearly identical, and that both packages produce identical results.

## A (brief) Code Example


```python
import pyfixest as pf
from pyfixest.utils.dgps import get_sharkfin

df = get_sharkfin()
```

We can fit a simple regression model via the *pf.feols()* function:

```python
fit = pf.feols(
  fml = "Y ~ treat | unit + year",
  data = df,
  vcov = "hetero"
)
pf.etable(fit, digits = 4)
```

In R and via *fixest*, we could have done the same with (almost) identical syntax:

```r
library(reticulate)
library(fixest)
df = py$df

fit = feols(
  fml = Y ~ treat | unit + year,
  data = df,
  vcov = "hetero"
)
etable(fit)
```
*PyFixest* and *fixest* produce an identical point estimate, standard error, and R2 value, etc.
