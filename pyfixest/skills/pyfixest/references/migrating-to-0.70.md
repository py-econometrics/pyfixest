# Migrating code to pyfixest 0.70

Code written for pyfixest 0.60 or earlier, and much code found online, uses
spellings that 0.70 deprecates or removes. Deprecated spellings still run and
warn; removed ones raise. Full notes: https://pyfixest.org/changelog.html and
https://pyfixest.org/how-to/migrate-fit-attributes.html

To find deprecated calls, run your code or tests with warnings as errors:
`python -W error::DeprecationWarning -W error::FutureWarning script.py`, or in
pytest `filterwarnings = ["error::DeprecationWarning", "error::FutureWarning"]`.

## Formula spellings

Deprecated (`DeprecationWarning`):

```text
Y ~ X1 | f1^f2                ->  Y ~ X1 | f1:f2
Y ~ X1 | f1 | X2 ~ Z1         ->  Y ~ X1 + [X2 ~ Z1] | f1
Y ~ X1 | X2 ~ Z1              ->  Y ~ X1 + [X2 ~ Z1]
Y + Y2 ~ X1                   ->  sw(Y, Y2) ~ X1
```

Rules that old code often breaks:

- Squares are `I(X1**2)`; `^` is not a power.
- `log()` is `np.log`: zeros give `-inf` and are dropped as infinite values
  with a warning, not as missing values.
- `i(f, k)` with a numeric `k` gives slopes; write `i(f, C(k))` for
  factor-by-factor dummies. `ref2=` or `bin2=` on a numeric `k` raise.
- Fixed effects must be data columns or `:` interactions with identifier
  names; `C(f1)` after `|` and `object` dtype columns raise.
- In IV models the main formula sets the intercept of both stages.
- Multiple-estimation model names are canonical: `"Y ~ sw(X1, X2)"` gives
  keys `"Y ~ 1 + X1"` and `"Y ~ 1 + X2"`. Update `rename_models` mappings and
  lookups into `all_fitted_models`.
- Some malformed formulas raise `formulaic.errors.FormulaSyntaxError`, a
  different class from `pyfixest.errors.FormulaSyntaxError`; catch both.

## Fitted-model attributes

Most private attributes are gone; accessing one such as `fit._se` raises
`AttributeError` with a link to the migration guide.

```python
import numpy as np
import pyfixest as pf

data = pf.get_data()
fit = pf.feols("Y ~ X1 + X2 | f1", data, vcov={"CRV1": "f1"})

fit.coef(), fit.se(), fit.tstat(), fit.pvalue()  # Series; were _beta_hat, _se, _tstat, _pvalue
fit.coeftable.conf_int  # was _conf_int; also coeftable.estimate, .se, .tstat, .pvalue
fit.variance_covariance.vcov  # was _vcov
fit.variance_covariance.spec.vcov_type_detail  # was _vcov_type_detail, e.g. "CRV1"
fit.variance_covariance.spec.clustervar  # was _clustervar
fit.sample_info.n_obs  # was _N
fit.sample_info.dropped_by_stage  # dropped rows: missing, infinite, singleton, ...
fit.fitstat.r2, fit.fitstat.adj_r2, fit.fitstat.r2_within, fit.fitstat.rmse  # were _r2, ...
fit.model.formula, fit.model.depvar, fit.model.fixef  # were _fml, _depvar, _fixef
fit.fixef_counts  # was _k_fe, _n_fe
fit.options.ssc  # an Ssc; was the _ssc_dict dict
fit.resid()  # was _u_hat, which weighted fits multiplied by sqrt(weights)
fit.fitted_values.response  # was _Y_hat_response; also .link
fit.capabilities  # was _support_* and _supports_*
```

| 0.60 | 0.70 |
|---|---|
| `fit.FixestFormula` | removed: `fit.model.parsed_formula` |
| `fit.na_index` | removed: `fit.sample_info.dropped_row_index` |
| `fit._fixef_dict`, `fit.fixef()` as dict | `fit.fixef()` returns a DataFrame (`variable`, `code`, `level`, `coefficient`) |
| `fit.get_performance()` | deprecated: read `fit.fitstat` |
| `fit.n_separation_na` | deprecated: `fit.sample_info.dropped_by_stage.separation` |
| `fit.maxiter`, `fit.tol`, `fit.separation_check` (GLM) | deprecated: `fit.options.maxiter`, ... |
| `fit._convergence` | `fit.convergence` (GLM, a bool) |
| `fit._weights` | `fit.observation_weights` |
| `fit._dfn`, `fit._f_statistic` | `fit.wald` |
| `multi.FixestFormulaDict` | deprecated: `fit.model.parsed_formula` on each model |

IV fits:

```python
iv = pf.feols("Y ~ X2 + [X1 ~ Z1] | f1", data)
iv.first_stage.model  # was _model_1st_stage (deprecated) or first_stage()
iv.first_stage.diagnostics.f_stat  # was _f_stat_1st_stage
iv.IV_Diag()
iv.first_stage.diagnostics.eff_f  # was _eff_F (deprecated); set after IV_Diag()
```

`fit.first_stage` is a component now, so `fit.first_stage()` raises
`TypeError`.

## Functions and arguments

Deprecated (warns):

```text
fit.confint(joint=True)                       ->  fit.confint(inference_type="simult", seed=1)
pf.ssc(adj=, fixef_k=, cluster_adj=, cluster_df=)  ->  pf.ssc(k_adj=, k_fixef=, G_adj=, G_df=)
```

Removed or changed (raises or returns something new):

```text
pf.feols(..., demeaner_backend="rust")  ->  pf.feols(..., demeaner=pf.MapDemeaner())
pf.feols(..., fixef_tol=1e-8, fixef_maxiter=100)
                                        ->  demeaner=pf.MapDemeaner(fixef_tol=1e-8, fixef_maxiter=100)
                                            or pf.LsmrDemeaner(fixef_atol=..., fixef_btol=...)
JAX, SciPy and CuPy demeaner backends   ->  pf.LsmrDemeaner(backend="torch", device="cuda") on GPU
pf.dtable()                             ->  maketables.DTable()
fit.vcov("hetero", data=df)             ->  fit.vcov("hetero"): data= raises ValueError
from pyfixest.estimation.feols_ import Feols
                                        ->  from pyfixest.estimation import Feols, Feiv, Fepois
```

- `pf.ssc()` returns a frozen `Ssc`, not a dict; dicts are still accepted as
  `ssc=` input.
- `wald_test()` returns a `WaldTest` (`stat`, `pvalue`, ...) instead of a
  Series.
- `feglm().resid()` returns response residuals `y - mu`;
  `resid(type="working")` gives the old working residuals.
- Methods that used to return wrong numbers on unsupported models now raise
  (`NotImplementedError` or `VcovTypeNotSupportedError`); check
  `fit.capabilities` first.
- Methods that need data dropped by `lean=True` or `store_data=False` raise
  `MissingModelDataError`.
- pyfixest no longer uses or reseeds NumPy's global random state: `ccv()`
  and `confint(inference_type="simult")` need `seed=` to be reproducible, and
  `ritest()` takes `rng=np.random.default_rng(seed)`.
- The default demeaner is LSMR from the `within` package with
  `preconditioner="auto"`, which now means adaptive, so results can differ
  slightly from 0.60. `pf.LsmrDemeaner(preconditioner="additive")` restores
  the 0.60 preconditioner.
- Python 3.11 or newer is required.
