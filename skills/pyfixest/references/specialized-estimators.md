# Specialized estimators

Use fepois or feglm for fixed-effects count and GLM models. feglm accepts the
documented families; offsets are supported only for Poisson. Validate
separation, weights, and fixed-effects support before generalizing an OLS
workflow.

Use quantreg with one quantile or a list of floats strictly between zero and
one. A list returns FixestMulti; pf.qplot visualizes the process. Do not use
fixed-effects or IV formula parts with quantile regression.

For difference-in-differences, use pf.event_study, pf.did2s, or pf.lpdid. Use
pf.panelview to inspect treatment timing. The packaged pyfixest.did/data/df_het.csv
fixture supports offline examples through importlib.resources.files(pyfixest.did).

For family-wise adjustments, use pf.bonferroni, pf.rwolf, or pf.wyoung. Read the
installed DiD, Poisson/GLM, and quantile tutorials before combining specialized
estimators with nonstandard inference.
