# News

## PyFixest `0.9.11`

- Bump required `formulaic` version to `0.6.5`.
- Stop copying the data frame in `fixef()`.

## PyFixest `0.9.10`

- Fixes a big in the `wildboottest` method (see [#158](https://github.com/s3alfisc/pyfixest/issues/158)).
- Allows to run a wild bootstrap after fixed effect estimation.

## PyFixest `0.9.9`

- Adds support for `wildboottest` for Python `3.11`.

## PyFixest `0.9.8`

- Fixes a couple more bugs in the `predict()` and `fixef()` methods.
- The `predict()` argument `data` is renamed to `newdata`.

## PyFixest `0.9.7`

Fixes a bug in `predict()` produced when multicollinear variables are dropped.

## PyFixest `0.9.6`

Improved Collinearity handling. See [#145](https://github.com/s3alfisc/pyfixest/issues/145)

## PyFixest `0.9.5`


- Moves plotting from `matplotlin` to `lets-plot`.
- Fixes a few minor bugs in plotting and the `fixef()` method.


## PyFixest `0.9.1`

### Breaking API changes

It is no longer required to initiate an object of type `Fixest` prior to running `feols` or `fepois`. Instead,
you can now simply use `feols()` and `fepois()` as functions, just as in `fixest`. Both function can be found in an
`estimation` module and need to obtain a `pd.DataFrame` as a function argument:

```py
from pyfixest.estimation import fixest, fepois
from pyfixest.utils import get_data

data = get_data()
fit = feols("Y ~ X1 | f1", data = data, vcov = "iid")
```

Calling `feols()` will return an instance of class `Feols`, while calling `fepois()` will return an instance of class `Fepois`.
Multiple estimation syntax will return an instance of class `FixestMulti`.

Post processing works as before via `.summary()`, `.tidy()` and other methods.

### New Features

A summary function allows to compare multiple models:

```py
from pyfixest.summarize import summary
fit2 = feols("Y ~ X1 + X2| f1", data = data, vcov = "iid")
summary([fit, fit2])
```

Visualization is possible via custom methods (`.iplot()` & `.coefplot()`), but a new module allows to visualize
  a list of `Feols` and/or `Fepois` instances:

```py
from pyfixest.visualize import coefplot, iplot
coefplot([fit, fit2])
```

The documentation has been improved (though there is still room for progress), and the code has been cleaned up a
bit (also lots of room for improvements).