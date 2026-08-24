# A first Regression with PyFixest

## Installation

`PyFixest` is a Python package for fast high-dimensional fixed effects regression. In this tutorial, we’ll show you how to fit your first regression with `PyFixest`.

You can install `pyfixest` from [PyPi](https://pypi.org/project/pyfixest/) via

``` bash
pip install -U pyfixest
```

## A First Regression: The Causal Returns to Education via Twin Studies

We want to estimate the **causal returns of education on earnings** via a twin study. In this notebook, we focus on `PyFixest` estimation functionality and syntax. For details on the question at hand, please take a look at the [OLS with Fixed Effects](tutorials/ols-fixed-effects.md) vignette.

In a first step, we load a synthetic twin-study style data set:

``` python
import pyfixest as pf

twins = pf.get_twin_data(N_pairs=500, seed=42)
twins.head()
```

`pf.get_twin_data()` returns a simulated twin-pair dataset where each row is one individual twin and `twin_pair_id` identifies a pair of twins. We have these other relevant variables:

- `educ`: years of education completed
- `earnings` (or log_wage in transformed specs): labor-market outcome used as the dependent variable
- `experience`: a proxy for labor-market experience
- `twin_pair_id`: twin-pair identifier
- `ability`: an unobserved confounder that leads to both higher earnings and more years of schooling

`PyFixest`’s core estimation function is called `feols()`. As a bare minimum, you need to pass a `pandas` or `polars` data frame and a Wilkinson formula.

We first estimate a naive OLS regression in which we regress earnings on years of education and experience. The resulting coefficient on `educ` will not reflect a causal effect of education on earnings, but will be biased because higher-ability students likely select into more schooling and would also have higher earnings later in life even in the absence of additional education.

We will now estimate a twin fixed-effects model, which aims to control for ability by comparing outcomes within twin pairs. As twins share the same genetic endowment, twin studies hypothesize that controlling for twin fixed effects can remove much of the ability-related confounding that biases cross-sectional OLS estimates. In both models, we cluster standard errors at the twin-pair level.

``` python
fit_naive = pf.feols(
  "log_wage ~ educ + experience",
  data=twins,
  vcov={"CRV1": "twin_pair_id"}
)
fit_fe = pf.feols(
  "log_wage ~ educ + experience | twin_pair_id",
   data=twins,
   vcov={"CRV1": "twin_pair_id"}
)
```

We compare both specifications side by side via `etable()`.

``` python
pf.etable(
    [fit_naive, fit_fe],
    labels={
      "log_wage": "Log Hourly Wage",
      "educ": "Years of Education",
      "experience": "Experience"
    },
    felabels={"twin_pair_id": "Twin Pair FE"},
    caption="Returns to Education: Naive OLS vs Twin Fixed Effects",
)
```

### Accessing fitted-model results

Regression tables are useful for presentation, but fitted `Feols` objects also let you access results directly for downstream analysis. Coefficients, standard errors, and the full tidy coefficient table are available through public methods:

``` python
fit_naive.coef()
fit_naive.se()
fit_naive.tidy()
```

Common model-level fit statistics are stored on the fitted object:

``` python
fit_naive._N        # number of observations
fit_naive._rmse     # root mean squared error
fit_naive._r2       # R-squared
fit_naive._adj_r2   # adjusted R-squared
```

For models with fixed effects, within R-squared measures are also available:

``` python
fit_fe._r2_within
fit_fe._adj_r2_within
```

We see that the estimated return to education is smaller once we include twin fixed effects, consistent with an upward bias in naive OLS of education on earnings from unobserved ability differences across individuals.

Now, what if we had the unobserved confounder at hand? In this case, we could simply control for it in our regression model. In real life, we likely wouldn’t be so lucky to have it, but alas, here we have access to it as we are working with synthetic data:

``` python
fit_latent = pf.feols(
  "log_wage ~ educ + experience + ability",
  data = twins,
  vcov={"CRV1": "twin_pair_id"}
)

pf.coefplot(
  [fit_naive, fit_fe, fit_latent],
  keep = ["educ"],
  coord_flip = False,
  title = "Three Estimates for the Returns of Education on Wages"
)
```

We see that the fixed effect design gives us estimates that are much closer to the model in which we correctly control for the unoberserved confounder. By controlling for twin fixed effects, we have managed to control for an unobserved confounder, ability.

## Where to Go Next

Now that we’ve fit our first regression, we can jump right into one of the next tutorials that showcases core PyFixest workflows for estimation, inference, and reporting of regression models with (and without) fixed effects.

| Tutorial | Description |
|----|----|
| [OLS with Fixed Effects](tutorials/ols-fixed-effects.md) | We provide more examples of fixed effects designs, including twin studies, worker-firm panels, and difference-in-differences models. We also provide some intuition on how the demeaning behind `PyFixest` works via the Frisch-Waugh-Lovell Theorem. |
| [Formula Syntax](tutorials/formula-syntax.md) | We explain `PyFixest`’s formula interface in all of its detail, including special operators as `i()` for interactions and multiple estimation syntax. |
| [Standard Errors & Inference](https://pyfixest.org/reference/estimation.models.feols_.Feols.vcov.llms.md) | Here we showcase different options to conduct inference with `PyFixest`, via iid, heteroskedastic, cluster robust errors, and more. |
| [Regression Tables](tutorials/regression-tables.md) | We show how to produce publication-ready tables via the `pf.etable()` function and `maketables`. |
| [Difference-in-Differences](tutorials/difference-in-differences.md) | TWFE, Gardner’s two-stage DID2S, local projections, and event study designs with heterogeneous treatment effects. |
| [Quantile Regression](tutorials/quantile-regression.md) | Interior-point quantile regression: model the full conditional distribution, not just the mean, with an example from software observability (p99 latency). |

You can browse all tutorials in the [Tutorial Gallery](tutorials/index.md), or see the [How-To Guides](how-to/marginaleffects.md) for task-oriented recipes. The [Function Reference](https://pyfixest.org/reference/index.llms.md) has all details around functions and their arguments, classes, methods, and attributes.
