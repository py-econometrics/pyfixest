## Readme

- [Check How close `PyFixest` reproduces standard errors produced via `fixest` and `stats::glm`](https://github.com/py-econometrics/pyfixest/tree/master/tests/check-crv-diffs-fixest-pyfixest-glm.qmd)
- [Test `PyFixest` against `fixest`](https://github.com/py-econometrics/pyfixest/tree/master/tests/test_vs_fixest.py)
- `pandas` needs to be a version lower than `1.5.3` to be compatible with `rpy2`, else you'll run into [this error](https://stackoverflow.com/questions/76404811/attributeerror-dataframe-object-has-no-attribute-iteritems). The github actions for testing ensures that `pandas` is of a version lower than `1.5.3`.
