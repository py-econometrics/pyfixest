# Capabilities

``` python
Capabilities(
    covariance_update,
    crv3_inference,
    hac_inference,
    multiway_clustering,
    wildboottest,
    cluster_causal_variance,
    decomposition,
    prediction,
    prediction_inference,
    fixed_effect_recovery,
    randomization_inference,
    sherman_morrison_update,
    anytime_valid_inference,
)
```

Inference and post-estimation features a fitted model supports.

Each model class declares one value, so a method can reject an unsupported estimator before it reads any estimation state rather than reinterpreting another estimator’s arrays.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| covariance_update | bool | Whether post-estimation `vcov()` updates are available. `did2s()` fits require refitting with the desired cluster variable instead. | *required* |
| crv3_inference | bool | Whether `vcov()` accepts `"CRV3"`. The jackknife refits the model on leave-one-cluster-out samples, so it is restricted to estimators whose refit replays the original estimation contract. | *required* |
| hac_inference | bool | Whether `vcov()` accepts the Newey-West and Driscoll-Kraay estimators. | *required* |
| multiway_clustering | bool | Whether `vcov()` accepts more than one cluster variable. | *required* |
| wildboottest | bool | Whether `wildboottest()` can resample the fit. Unweighted OLS only. | *required* |
| cluster_causal_variance | bool | Whether `ccv()`, the causal cluster variance estimator, is available. | *required* |
| decomposition | bool | Whether `decompose()`, the Gelbach decomposition, is available. | *required* |
| prediction | bool | Whether `predict()` is available. IV and `did2s()` fits do not support it. | *required* |
| prediction_inference | bool | Whether `predict(se_fit=True)` and `predict(interval="prediction")` are available. Only OLS fits support them, additionally requiring no weights or fixed effects. | *required* |
| fixed_effect_recovery | bool | Whether the model class can recover fixed-effect estimates through `fixef()`. IV, `quantreg()`, and `did2s()` fits do not support it. `fixef()` additionally requires the fit to have fixed effects. | *required* |
| randomization_inference | bool | Whether `ritest()`, randomization inference, is available. Only `feols()` fits without instruments and `fepois()` fits support it; `did2s()` and `event_study()` fits do not. | *required* |
| sherman_morrison_update | bool | Whether `update()` can append observations to the fit through a Sherman-Morrison update of the OLS coefficients. Only `feols()` fits without instruments support it; `did2s()` and `event_study()` fits do not. | *required* |
| anytime_valid_inference | bool | Whether `evalue()`, `pvalue_savi()`, and `confint(inference_type="savi")`, safe anytime-valid inference, are available. Only `feols()` fits without instruments support it; `did2s()` and `event_study()` fits do not. SAVI additionally requires no weights, no fixed effects, and an iid or heteroskedasticity-robust covariance. | *required* |

## Examples

``` python
import pyfixest as pf

pf.feols("Y ~ X1 | f1", pf.get_data()).capabilities
```

    Capabilities(covariance_update=True, crv3_inference=True, hac_inference=True, multiway_clustering=True, wildboottest=True, cluster_causal_variance=True, decomposition=True, prediction=True, prediction_inference=True, fixed_effect_recovery=True, randomization_inference=True, sherman_morrison_update=True, anytime_valid_inference=True)

``` python
pf.fepois("Y ~ X1 | f1", pf.get_data(model="Fepois")).capabilities
```

    Capabilities(covariance_update=True, crv3_inference=True, hac_inference=True, multiway_clustering=True, wildboottest=False, cluster_causal_variance=False, decomposition=False, prediction=True, prediction_inference=False, fixed_effect_recovery=True, randomization_inference=True, sherman_morrison_update=False, anytime_valid_inference=False)
