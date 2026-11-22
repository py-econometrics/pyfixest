# Preserve the inference contract in post-estimation

Reuse the fitted model's covariance specification and small-sample
correction when a second calculation needs to reproduce its inference.
If the operation uses different clusters, compute its covariance locally
so it leaves the model's published inference intact.

## Covariance and hypothesis tests

- HC2 and HC3 use leverage from the weighted design. For frequency weights,
  the leverage of an individual repeated observation is the row's leverage
  divided by its frequency, and the aggregated score uses the square root
  of that frequency. GLMs use the final IRLS weights with their
  unpremultiplied design. Weighted RMSE normalizes by the sum of weights,
  as in `fixest`. Gaussian GLM inference follows `feols` where specified in
  the [compatibility ledger](../developer/fixest-compatibility.md#gaussian-glm-inference).
- Inference builds unadjusted covariance terms per cluster dimension and
  applies the small-sample factors before combining them. Heteroskedastic
  inference can use the sample size for the cluster adjustment. HAC uses
  the number of time periods where applicable. The fixed-effect degree-of-freedom
  convention is discussed in [fixest issue 554](https://github.com/lrberge/fixest/issues/554).
- Confidence intervals use the fitted t degrees of freedom. The default Wald
  F case is the joint zero restriction on all coefficients. A fit without
  coefficients has no inference rows. The wild bootstrap API tests a single
  named coefficient against zero.
- Prediction uncertainty is supported for unweighted OLS without fixed effects.
  Fixed-effect models still support point prediction.
- The CCV comparison covariance and cluster component must use the same
  clusters as the split scores (Abadie et al., 2023). Use local refits with
  the fitted small-sample correction. Where only coefficients are needed,
  request iid covariance to avoid unnecessary cluster work.
- Randomization-t centers the coefficient at the null before studentizing.
  Permuted statistics use the fitted covariance and small-sample correction,
  as does the observed statistic (Wu and Ding, JASA 2021). Permutations preserve
  the rows and replace the assignment on a shallow copy. The fast solver
  applies only to a bare assignment column. Use formula refits for
  studentization or when assignment changes interactions, transformations,
  or the response. An assignment-cluster override belongs to the test.
- Romano–Wolf step-down uses the same bootstrap draws on each iteration.
  Reset the seed so adjusted p-values compare the same resamples.

## Staged and decomposition inference

- Gardner's DID2S cluster scores must include each estimation observation.
  Validate cluster values before fitting so missing clusters cannot be
  retained in the model and then omitted from its covariance. Remove the
  second-stage intercept when the first stage includes fixed effects.
  Use the sparse solver's CSC input. Factorization of a Series yields an
  index even where the pandas stubs admit an array. Normalize plot treatment
  indicators to Boolean to handle integer treatment columns.
- Gelbach decomposition aligns the short design with its requested regressor
  order and treats decomposition weights as frequency weights. The
  Share of Full Effect table omits the direct-effect interval where the
  normalization fixes it to `[1, 1]`.
