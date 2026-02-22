# report.summary { #pyfixest.report.summary }

```python
report.summary(models, digits=3)
```

Print a summary of estimation results for each estimated model.

For each model, this method prints a header indicating the fixed-effects and the
dependent variable, followed by a table of coefficient estimates with standard
errors, t-values, and p-values.

## Parameters {.doc-section .doc-section-parameters}

| Name   | Type                                                                     | Description                                                                    | Default    |
|--------|--------------------------------------------------------------------------|--------------------------------------------------------------------------------|------------|
| models | A supported model object (Feols, Fepois, Feiv, FixestMulti) or a list of | Feols, Fepois & Feiv models.                                                   | _required_ |
| digits | int                                                                      | The number of decimal places to round the summary statistics to. Default is 3. | `3`        |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description   |
|--------|--------|---------------|
|        | None   |               |

## Examples {.doc-section .doc-section-examples}

```{python}
import pyfixest as pf

# load data
df = pf.get_data()
fit1 = pf.feols("Y~X1 + X2 | f1", df)
fit2 = pf.feols("Y~X1 + X2 | f1 + f2", df)
fit3 = pf.feols("Y~X1 + X2 | f1 + f2 + f3", df)

pf.summary([fit1, fit2, fit3])
```