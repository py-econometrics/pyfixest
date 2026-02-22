PyFixest now experimentally supports quantile regression!

```{python}
%load_ext autoreload

import pyfixest as pf
data = pf.get_data()
```

## Basic Example

Just as in `statsmodels`, the function that runs a quantile regression is `quantreg()`.

Below, we loop over 10 different quantiles.

```{python}
%%capture
fits = pf.quantreg(
  fml = "Y~X1 + X2",
  data = data,
  quantile=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
)
```

We can inspect the quantile regression results using the dedicated `qplot()` function.

```{python}
pf.qplot(fits, nrow = 2)
```

We observe some heterogeneity in the intercept, but all other variants are homogeneous across users.

## Solvers

By default, `pf.quantreg` uses an interior-point solver as in [Koenker and Ng (2004)](http://www.econ.uiuc.edu/~roger/research/sparse/fn3.pdf) (`methd = "fn"`). This is different to e.g. `statsmodels`, which implements an iterated weighted least squares solver.

For big data sets with many observations, it is often sensible to use an interior-point solver with pre-processing (as in [Portnoy and Koenker (1997)](https://experts.illinois.edu/en/publications/the-gaussian-hare-and-the-laplacian-tortoise-computability-of-squ), see [Chernozhukov et al (2019)](https://arxiv.org/abs/1909.05782) for details), which can speed up the estimation time significantly. Because the pre-processing step requires taking a random sample, the method assumes that observations are independent. Additionally, for the purpose of reproducibility, it is advisable to set a seed.

You can access the "preprocessing frisch-newton" algorithm by setting the `method` argument to `"pfn"`:

```{python}
%%capture
fit_fn = pf.quantreg(
  fml = "Y ~ X1",
  method = "fn",     # standard frisch newton interior point solver
  data = data,
)
fit_pfn = pf.quantreg(
  fml = "Y ~ X1",
  method = "pfn",   # standard frisch newton interior point solver with pre-processing
  seed = 92,         # set a seed for reproducibility
  data = data,
)

pf.etable([fit_fn, fit_pfn])
```

## Quantile Regression Process

Instead of running multiple independent quantile regression via a for-loop, the literature on quantile regression has developed multiple algorithms to speed up the "quantile regression process". Two such algorithms are described in detail in Chernozhukov, Fernandez-Val and Melly (2019) and are implemented in PyFixest. They can be accessed via the `multi_method` argument, and both can significantly speed up estimation time of the full quantile regression process.

```{python}
fml = "Y~X1"
method = "pfn"
seed = 929
quantiles = [0.1, 0.5, 0.9]

fit_multi1 = pf.quantreg(
  fml = fml,
  data = data,
  method = method,
  multi_method = "cfm1",  # this is algorithm 2 in CFM, the 1rst algorithm for the full qr process
  seed = seed,
  quantile = quantiles,
)

fit_multi2 = pf.quantreg(
  fml = fml,
  data = data,
  method = method,
  multi_method = "cfm2",  # this is algorithm 3 in CFM, the 2nd algorithm for the full qr process
  seed = seed,
  quantile = quantiles
)

pf.etable(fit_multi1.to_list() +  fit_multi2.to_list())
```

Note that the first method `cfm1` is exactly identical to running separate regressions per quantile, while the second method `cfm2` is only **asymptotically** identical.

You can combine different estimation `method`'s with different `multi_methods`:

```{python}
fit_multi2a = pf.quantreg(
  fml = "Y~X1",
  data = data,
  method = "fn",
  multi_method = "cfm1",
  seed = 233,
  quantile = [0.25, 0.75]
)

fit_multi2b = pf.quantreg(
  fml = "Y~X1",
  data = data,
  method = "pfn",
  multi_method = "cfm1",
  seed = 233,
  quantile = [0.25, 0.75]
)

pf.etable(fit_multi2a.to_list() +  fit_multi2b.to_list())

```

## Inference

By default, the `"iid", "hetero"` and cluster robust variance estimators implement (sandwich) estimators as in [Powell (1991)](https://econpapers.repec.org/paper/attwimass/8818.htm), using a uniform kernel to estimate the "sparsity".

The cluster robust estimator follows Parente & Santos Silva. See this [slide set](https://www.stata.com/meeting/uk15/abstracts/materials/uk15_santossilva.pdf) or the [Journal of Econometrics paper](https://repository.essex.ac.uk/8976/1/dp728.pdf) for details.

Additionally, `pf.quantreg` supports the `"nid"` ("non-iid") estimator from [Hendricks and Koenker (1991)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1992.10475175), which uses a linear approximation of the conditional quantile function.

```{python}
fit_nid = pf.quantreg("Y ~ X1 + X2 + f1", data = data, quantile = 0.5, vcov = "nid")
fit_crv = pf.quantreg("Y ~ X1 + X2 + f1", data = data, quantile = 0.5, vcov = {"CRV1": "f1"})
```

## Performance

Here we benchmark the performance of the solvers accessible via the `method` and `multi_method` arguments.

### Different Solvers

Tba.

### Quantile Regression Process

We fit a quantile regression process on $q = 0.1, 0.2, ..., 0.9$ quantiles and vary sample size and number of covariates. We test pyfixest's implementation of the quantile regression process against a "naive" for loop implementation via `statsmodels`. We can see that both `multi_method = "cmf1"` and `multi_method = "cmf2"` outperform
the for-loop strategy for large problems. Note that the plot is in log-scale!

![](figures/quantreg_benchmarks.png)

# Literature

- Victor Chernozhukov, Iván Fernández-Val, Blaise Melly (2019): Fast Algorithms for the Quantile Regression Process - [link](https://arxiv.org/abs/1909.05782)
- Hendricks & Koenker (1991): Hierarchical spline models for conditional quantiles and the demand for electricity - [link](https://www.tandfonline.com/doi/abs/10.1080/01621459.1992.10475175)
- Koenker and Ng (2004): A Frisch-Newton Algorithm for Sparse Quantile Regression - [link](http://www.econ.uiuc.edu/~roger/research/sparse/fn3.pdf)
- Parente & Santos Silva (2015): Quantile Regression with Clustered Data - [link](https://econpapers.repec.org/article/bpjjecome/v_3a5_3ay_3a2016_3ai_3a1_3ap_3a1-15_3an_3a5.htm)
- Portnoy & Koenker (1997): The gaussian hare and the laplacian tortoise: Computability of squared-error versus absolute-error estimators - [link](https://experts.illinois.edu/en/publications/the-gaussian-hare-and-the-laplacian-tortoise-computability-of-squ)
- Powell (1991): Estimation of monotonic regression models under quantile restrictions - [link](https://econpapers.repec.org/paper/attwimass/8818.htm)
