# report.iplot { #pyfixest.report.iplot }

```python
report.iplot(
    models,
    alpha=0.05,
    figsize=None,
    yintercept=None,
    xintercept=None,
    rotate_xticks=0,
    title=None,
    coord_flip=True,
    keep=None,
    drop=None,
    exact_match=False,
    plot_backend='lets_plot' if _HAS_LETS_PLOT else 'matplotlib',
    labels=None,
    cat_template=None,
    rename_models=None,
    ax=None,
    joint=None,
    seed=None,
)
```

Plot model coefficients for variables interacted via "i()" syntax, with
confidence intervals.

## Parameters {.doc-section .doc-section-parameters}

| Name          | Type                                                                     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Default                                           |
|---------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| models        | A supported model object (Feols, Fepois, Feiv, FixestMulti) or a list of | Feols, Fepois & Feiv models.                                                                                                                                                                                                                                                                                                                                                                                                                                                | _required_                                        |
| figsize       | tuple or None                                                            | The size of the figure. If None, the default size is used.                                                                                                                                                                                                                                                                                                                                                                                                                  | `None`                                            |
| alpha         | float                                                                    | The significance level for the confidence intervals.                                                                                                                                                                                                                                                                                                                                                                                                                        | `0.05`                                            |
| yintercept    | int or None                                                              | The value at which to draw a horizontal line on the plot.                                                                                                                                                                                                                                                                                                                                                                                                                   | `None`                                            |
| xintercept    | int or None                                                              | The value at which to draw a vertical line on the plot.                                                                                                                                                                                                                                                                                                                                                                                                                     | `None`                                            |
| rotate_xticks | float                                                                    | The angle in degrees to rotate the xticks labels. Default is 0 (no rotation).                                                                                                                                                                                                                                                                                                                                                                                               | `0`                                               |
| title         | str                                                                      | The title of the plot.                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `None`                                            |
| coord_flip    | bool                                                                     | Whether to flip the coordinates of the plot. Default is True.                                                                                                                                                                                                                                                                                                                                                                                                               | `True`                                            |
| keep          | Optional\[Union\[list, str\]\]                                           | The pattern for retaining coefficient names. You can pass a string (one pattern) or a list (multiple patterns). Default is keeping all coefficients. You should use regular expressions to select coefficients.     "age",            # would keep all coefficients containing age     r"^tr",           # would keep all coefficients starting with tr     r"\\d$",          # would keep all coefficients ending with number Output will be in the order of the patterns. | `None`                                            |
| drop          | Optional\[Union\[list, str\]\]                                           | The pattern for excluding coefficient names. You can pass a string (one pattern) or a list (multiple patterns). Syntax is the same as for `keep`. Default is keeping all coefficients. Parameter `keep` and `drop` can be used simultaneously.                                                                                                                                                                                                                              | `None`                                            |
| exact_match   | bool                                                                     | Whether to use exact match for `keep` and `drop`. Default is False. If True, the pattern will be matched exactly to the coefficient name instead of using regular expressions.                                                                                                                                                                                                                                                                                              | `False`                                           |
| plot_backend  | str                                                                      | The plotting backend to use. Options are "lets_plot" (default if installed) and "matplotlib". If "lets_plot" is specified but not installed, an ImportError will be raised with instructions to install it or use "matplotlib" instead.                                                                                                                                                                                                                                     | `'lets_plot' if _HAS_LETS_PLOT else 'matplotlib'` |
| rename_models | dict                                                                     | A dictionary to rename the models. The keys are the original model names and the values the new names.                                                                                                                                                                                                                                                                                                                                                                      | `None`                                            |
| labels        | Optional\[dict\]                                                         | A dictionary to relabel the variables. The keys in this dictionary are the original variable names, which correspond to the names stored in the `_coefnames` attribute of the model. The values in the dictionary are the new  names you want to assign to these variables. Note that interaction terms will also be relabeled using the labels of the individual variables. The renaming is applied after the selection of the coefficients via `keep` and `drop`.         | `None`                                            |
| cat_template  | Optional\[str\]                                                          | Template to relabel categorical variables. None by default, which applies no relabeling. Other options include combinations of "{variable}" and "{value}", e.g. "{variable}::{value}" to mimic fixest encoding. But "{variable}--{value}" or "{variable}{value}" or just "{value}" are also possible.                                                                                                                                                                       | `None`                                            |
| joint         | Optional\[Union\[str, bool\]\]                                           | Whether to plot simultaneous confidence bands for the coefficients. If True, simultaneous confidence bands are plotted. If False, "standard" confidence intervals are plotted. If "both", both are plotted in one figure. Default is None, which returns the standard confidence intervals. Note that this option is not available for objects of type `FixestMulti`, i.e. multiple estimation.                                                                             | `None`                                            |
| seed          | Optional\[int\]                                                          | The seed for the random number generator. Default is None. Only required / used when `joint` is True.                                                                                                                                                                                                                                                                                                                                                                       | `None`                                            |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                               |
|--------|--------|-------------------------------------------|
|        | object | A plot figure from the specified backend. |

## Examples {.doc-section .doc-section-examples}

```{python}
import pyfixest as pf
from pyfixest.report.utils import rename_categoricals

df = pf.get_data()
fit1 = pf.feols("Y ~ i(f1)", data = df)
fit2 = pf.feols("Y ~ i(f1) + X2", data = df)
fit3 = pf.feols("Y ~ i(f1) + X2 | f2", data = df)

pf.iplot([fit1, fit2, fit3], labels = rename_categoricals(fit1._coefnames))
pf.iplot(
    models = [fit1, fit2, fit3],
    labels = rename_categoricals(fit1._coefnames)
)
pf.iplot(
    models = [fit1, fit2, fit3],
    rename_models = {
        fit1._model_name_plot: "Model 1",
        fit2._model_name_plot: "Model 2",
        fit3._model_name_plot: "Model 3"
    },
)
pf.iplot(
    models = [fit1, fit2, fit3],
    rename_models = {
        "Y~i(f1)": "Model 1",
        "Y~i(f1)+X2": "Model 2",
        "Y~i(f1)+X2|f2": "Model 3"
    },
)
pf.iplot([fit1], joint = "both")
```