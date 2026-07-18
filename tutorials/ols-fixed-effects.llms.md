# OLS with Fixed Effects

Core Estimation

Twin studies and ‘AKM’ regressions in worker-firm panels: two cases studies in controlling for unobserved heterogeneity with fixed effects.

> **NOTE:**
>
> You should have read the [Getting Started](../getting-started.llms.md) page and have `pyfixest` installed.

## What Are Fixed Effects?

A fixed effect model includes group-specific intercepts that absorb unobserved heterogeneity.

The canonical panel data model is:

\\ Y\_{it} = \beta X\_{it} + \alpha_i + \psi_t + \varepsilon\_{it} \\

where \\\alpha_i\\ is an individual fixed effect (constant across time) and \\\psi_t\\ is a time fixed effect (constant across individuals). Fixed effects are not limited to panel data - any categorical grouping variable can serve as a fixed effect (for example, wage regressions with worker and firm FE) - more on that topic later!).

`PyFixest` efficiently estimates fixed effects models by applying the [Frisch-Waugh-Lovell](https://mattblackwell.github.io/gov2002-book/least_squares.html#residual-regression) , which, among other things, avoids the need to create hundreds of dummy variables.

In the following section, we introduce two chanonical use cases of fixed effects regression.

## Application 1: Twin Studies and the Returns to Education

One of the most foundational question in the economics of education is “wow much does an extra year of education raise wages”? If we were to simply regress years of education on realized wages, we would likely overstates the return to education as there is a selection bias: *ability* drives both education and wages. Or, in other words, kids with high innate (but unobserved) ability end up with more years of education, but also higher wages! The relation between education and wages might be spurios, as both are driven by the same latent factor.

Twin studies aim to correct for this selection effect by comparing twins who share the same genetic endowment. If the latent innate ability is encoded in the genome, then twins with identical gene should have the same latent ability. Under this assumption, any difference in educational attainment between twins is not driven by innate ability. As a result any within‑twin difference in wages can be attributed to differences in schooling rather than unobserved ability.

In practice, twin fixed‑effects regressions compare each twin to their sibling, netting out shared genes and family background. The estimated coefficient on schooling then captures the causal return to education under the assumption that the only remaining differences between twins are not systematically related to both schooling and wages.

``` python
import pyfixest as pf

twins = pf.get_twin_data(N_pairs=500, seed=42)
twins.head()
```

|     | twin_pair_id | twin_id | ability   | educ      | age  | experience | log_wage |
|-----|--------------|---------|-----------|-----------|------|------------|----------|
| 0   | 1            | 1       | 0.304717  | 14.880083 | 38.0 | 17.119917  | 3.241823 |
| 1   | 1            | 2       | 0.304717  | 13.942729 | 49.0 | 29.057271  | 3.379130 |
| 2   | 2            | 1       | -1.039984 | 10.041047 | 33.0 | 16.958953  | 2.303006 |
| 3   | 2            | 2       | -1.039984 | 8.475001  | 32.0 | 17.524999  | 2.057258 |
| 4   | 3            | 1       | 0.750451  | 8.000000  | 35.0 | 21.000000  | 3.449381 |

### Naive OLS (biased)

In a first step, we estimate the “naive” regression and fit the relation between education and wages.

Without controlling for ability, the coefficient on `educ` captures both the true return to education and the selection effect:

``` python
fit_naive = pf.feols("log_wage ~ educ + experience", data=twins)
fit_naive.summary()
```

    ###

    Estimation:  OLS
    Dep. var.: log_wage
    sample: None = all
    Inference:  iid
    Observations:  1000

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | Intercept     |      1.113 |        0.091 |    12.163 |      0.000 |  0.933 |   1.292 |
    | educ          |      0.114 |        0.007 |    17.548 |      0.000 |  0.101 |   0.127 |
    | experience    |      0.019 |        0.001 |    12.851 |      0.000 |  0.016 |   0.022 |
    ---
    RMSE: 0.407 R2: 0.283 

### Twin-Pair Fixed Effects

In the next regression, we include a fixed effect for each twin pair. This controls for everything the twins share, including genes and environment, so the estimate uses only differences in education between the twins.

``` python
fit_fe = pf.feols("log_wage ~ educ + experience | twin_pair_id", data=twins)
fit_fe.summary()
```

    ###

    Estimation:  OLS
    Dep. var.: log_wage, Fixed effects: twin_pair_id
    sample: None = all
    Inference:  iid
    Observations:  1000

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | educ          |      0.088 |        0.007 |    11.863 |      0.000 |  0.073 |   0.103 |
    | experience    |      0.020 |        0.002 |    12.516 |      0.000 |  0.016 |   0.023 |
    ---
    RMSE: 0.214 R2: 0.801 R2 Within: 0.34 

### Compare Side by Side

The FE estimate (~0.08) is smaller than the “naive” OLS estimate. Indeed, part of the correlation between education and wages is that higher ability students obtain more years of education.

``` python
pf.etable(
    [fit_naive, fit_fe],
    labels={"log_wage": "Log Hourly Wage", "educ": "Years of Education", "experience": "Experience"},
    felabels={"twin_pair_id": "Twin Pair FE"},
    caption="Returns to Education: Naive OLS vs Twin FE",
)
```

[TABLE]

``` python
pf.coefplot([fit_naive, fit_fe], keep="educ")
```

## Application 2: AKM Worker-Firm Regressions

Wages of workers depend on both *worker characteristics* and *workplace characteristics*. Higher-skill worker might earn more, but there might also be workplace-premia. A two-way fixed effects model as formulated in Abowd, Kramarz & Margolis (AKM, 1999) separates these unobserved effects.

For some background reading on AKM models and their application, take a look at this slide deck: [AKM Lecture Slides](https://eml.berkeley.edu/~cle/e250a_f14/lecture11.pdf).

``` python
panel = pf.get_worker_panel(N_workers=500, N_firms=50, N_years=11, seed=42)
panel.head()
```

|     | worker_id | firm_id | year | female | experience | tenure | log_wage  | worker_fe | firm_fe   |
|-----|-----------|---------|------|--------|------------|--------|-----------|-----------|-----------|
| 0   | 0         | 48      | 2000 | 1      | 1          | 1      | 0.406292  | 0.152359  | 0.212055  |
| 1   | 1         | 30      | 2000 | 0      | 2          | 1      | -0.382227 | -0.519992 | -0.108531 |
| 2   | 2         | 18      | 2000 | 1      | 2          | 1      | 0.411605  | 0.375226  | 0.269345  |
| 3   | 3         | 22      | 2000 | 1      | 0          | 1      | 1.232769  | 0.470282  | 0.400735  |
| 4   | 4         | 8       | 2000 | 1      | 2          | 1      | -1.335275 | -0.975518 | -0.167872 |

### One-Way FE: Worker Fixed Effects Only

``` python
fit_worker = pf.feols("log_wage ~ experience + tenure + female | worker_id", data=panel)
fit_worker.summary()
```

    ###

    Estimation:  OLS
    Dep. var.: log_wage, Fixed effects: worker_id
    sample: None = all
    Inference:  iid
    Observations:  5500

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | experience    |      0.021 |        0.001 |    15.047 |      0.000 |  0.018 |   0.024 |
    | tenure        |      0.013 |        0.002 |     5.330 |      0.000 |  0.008 |   0.017 |
    ---
    RMSE: 0.299 R2: 0.776 R2 Within: 0.08 

### Two-Way FE: Worker + Firm Fixed Effects

``` python
fit_twoway = pf.feols("log_wage ~ experience + tenure + female | worker_id + firm_id", data=panel)
fit_twoway.summary()
```

    ###

    Estimation:  OLS
    Dep. var.: log_wage, Fixed effects: worker_id + firm_id
    sample: None = all
    Inference:  iid
    Observations:  5500

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | experience    |      0.021 |        0.001 |    22.547 |      0.000 |  0.019 |   0.023 |
    | tenure        |      0.011 |        0.002 |     6.830 |      0.000 |  0.008 |   0.014 |
    ---
    RMSE: 0.192 R2: 0.908 R2 Within: 0.159 

### Adding Year Fixed Effects

``` python
fit_full = pf.feols("log_wage ~ experience + tenure + female | worker_id + firm_id + year", data=panel)
fit_full.summary()
```

    ###

    Estimation:  OLS
    Dep. var.: log_wage, Fixed effects: worker_id + firm_id + year
    sample: None = all
    Inference:  iid
    Observations:  5500

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | experience    |      0.020 |        0.002 |     9.897 |      0.000 |  0.016 |   0.024 |
    | tenure        |      0.010 |        0.002 |     6.125 |      0.000 |  0.007 |   0.013 |
    ---
    RMSE: 0.191 R2: 0.908 R2 Within: 0.027 

Here is an interesting historical fact: the first paper (to our knowledge) that fitted a three-way high-dimensional fixed effects model in an unbalanced panel was published in 2013, only four years before the transformer was invented!. Before that, economists simply did not know how to fit 3-way fixed effects regression models on unbalanced panels efficiently. See Guimarães, Portugal, and Torres, [“The Sources of Wage Variation: A Three-Way High-Dimensional Fixed Effects Regression Model”](https://ideas.repec.org/p/ptu/wpaper/w201309.html), who more or less fit the model above on Portuguese data.

### Compare All Specifications

``` python
pf.etable(
    [fit_worker, fit_twoway, fit_full],
    labels={"log_wage": "Log Wage", "experience": "Experience", "tenure": "Tenure", "female": "Female"},
    felabels={"worker_id": "Worker FE", "firm_id": "Firm FE", "year": "Year FE"},
    caption="Worker-Firm Panel: Adding Fixed Effects Progressively",
)
```

[TABLE]

> **TIP:**
>
> Another very relevant class of fixed effects models are used in [Difference-in-Differences designs](../tutorials/difference-in-differences.llms.md).
