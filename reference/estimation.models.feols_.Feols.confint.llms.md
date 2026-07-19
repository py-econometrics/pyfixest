# Feols.confint

``` python
confint(
    alpha=0.05,
    keep=None,
    drop=None,
    exact_match=False,
    joint=False,
    seed=None,
    reps=10000,
    *,
    inference_type='regular',
    mixture_precision=1.0,
)
```

Fitted model confidence intervals.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| alpha | float | The significance level for confidence intervals. Defaults to 0.05. keep: str or list of str, optional | `0.05` |
| joint | bool | Deprecated. Use `inference_type="simult"` instead. Whether to compute simultaneous confidence intervals for the joint null of the parameters selected by `keep` and `drop`. Defaults to False. See https://www.causalml-book.org/assets/chapters/CausalML_chap_4.pdf, Remark 4.4.1 for details. | `False` |
| keep | list \| str \| None | The pattern for retaining coefficient names. You can pass a string (one pattern) or a list (multiple patterns). Default is keeping all coefficients. You should use regular expressions to select coefficients. “age”, \# would keep all coefficients containing age r”^tr”, \# would keep all coefficients starting with tr r”\d\$“, \# would keep all coefficients ending with number Output will be in the order of the patterns. | `None` |
| drop | list \| str \| None | The pattern for excluding coefficient names. You can pass a string (one pattern) or a list (multiple patterns). Syntax is the same as for `keep`. Default is keeping all coefficients. Parameter `keep` and `drop` can be used simultaneously. | `None` |
| exact_match | bool \| None | Whether to use exact match for `keep` and `drop`. Default is False. If True, the pattern will be matched exactly to the coefficient name instead of using regular expressions. | `False` |
| reps | int | The number of bootstrap iterations to run for joint confidence intervals. Defaults to 10_000. Only used if `joint` is True. | `10000` |
| seed | int | The seed for the random number generator. Defaults to None. Only used when `inference_type="simult"`. | `None` |
| inference_type | (regular, simult, savi) | Type of confidence interval to compute. “regular” returns pointwise intervals; “simult” returns simultaneous (joint) intervals for the coefficients selected by `keep` and `drop`; “savi” returns coefficient-wise asymptotic SAVI confidence sequences. Defaults to “regular”. Supersedes the deprecated `joint` argument. | `"regular"` |
| mixture_precision | float | Only relevant for `inference_type="savi"`. Controls the mixing weight of the prior in the SAVI e-value. Larger values produce wider confidence sequences early on but narrow faster as the sample grows. Defaults to 1. Use `pyfixest.optimal_mixture_precision()` to minimize confidence-sequence width at a target sample size. | `1.0` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | pd.DataFrame | A pd.DataFrame with confidence intervals of the estimated regression model for the selected coefficients. |

## Notes

SAVI currently supports unweighted, non-IV `feols` models without absorbed fixed effects. The covariance estimator must be iid or heteroskedasticity robust (`hetero`, `HC1`, `HC2`, or `HC3`). With `HC2`/`HC3`, pyfixest’s default small-sample correction scales the variance by `n / (n - k)`. You need to pass `ssc(k_adj=False)` to reproduce `avlm`, which applies no such correction.

## Examples

``` python
from pyfixest.utils import get_data
from pyfixest.estimation import feols

data = get_data()
fit = feols("Y ~ C(f1)", data=data)
fit.confint(alpha=0.10).head()
fit.confint(alpha=0.10, inference_type="simult", reps=9999).head()

savi_fit = feols("Y ~ X1 + X2", data=data, vcov="hetero")
savi_fit.confint(alpha=0.10, inference_type="savi").head()
```

|           | 5.0%      | 95.0%     |
|-----------|-----------|-----------|
| Intercept | 0.522146  | 1.255412  |
| X1        | -1.264565 | -0.721307 |
| X2        | -0.250161 | -0.102524 |
