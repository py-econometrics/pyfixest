<!-- Generated from docs/troubleshooting.md; do not edit. -->

# Troubleshooting

Start with the exception message: public-boundary errors name the invalid argument,
the received value, and the accepted alternatives. Use the focused pages below
before changing model syntax or inference assumptions.

- Formula parsing and multiple estimation: [Formula syntax](tutorials/formula-syntax.md)
- Variance estimators and clustering: [Standard errors and inference](tutorials/standard-errors.md)
- Small-sample corrections: [Small sample corrections](explanation/ssc.md)
- Fixed-effects convergence: [Choosing a demeaner](how-to/demeaner-backends.md)
- Result tables and plots: [Regression tables](tutorials/regression-tables.md)

## Missing stored data

Methods such as prediction, fixed-effect recovery, and some post-estimation routines
need the estimation data. Refit with `store_data=True` and without `lean=True` when
the model was created with storage disabled.

## Unsupported combinations

Not every covariance estimator or post-estimation method supports weights, fixed
effects, instrumental variables, or every model family. Preserve the requested
econometric design and choose a supported method listed in the relevant API
reference instead of silently dropping part of the model.

## Optional dependencies

Install the extra named in the exception message only when using that feature. For
example, plotting uses the `plots` extra and optional accelerated paths can use the
`numba` or `torch` extras.
