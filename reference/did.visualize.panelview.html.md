# did.visualize.panelview { #pyfixest.did.visualize.panelview }

```python
did.visualize.panelview(
    data,
    unit,
    time,
    treat,
    outcome=None,
    collapse_to_cohort=False,
    subsamp=None,
    units_to_plot=None,
    sort_by_timing=False,
    xlab=None,
    ylab=None,
    figsize=(11, 3),
    noticks=False,
    title=None,
    legend=False,
    ax=None,
    xlim=None,
    ylim=None,
)
```

Generate a panel view of the treatment variable over time for each unit.

## Parameters {.doc-section .doc-section-parameters}

| Name               | Type                   | Description                                                                                   | Default    |
|--------------------|------------------------|-----------------------------------------------------------------------------------------------|------------|
| data               | pandas.DataFrame       | The input dataframe containing the data.                                                      | _required_ |
| unit               | str                    | The column name representing the unit identifier.                                             | _required_ |
| time               | str                    | The column name representing the time identifier.                                             | _required_ |
| treat              | str                    | The column name representing the treatment variable.                                          | _required_ |
| outcome            | str                    | The column name representing the outcome variable. If not None, an outcome plot is generated. | `None`     |
| collapse_to_cohort | bool                   | Whether to collapse units into treatment cohorts.                                             | `False`    |
| subsamp            | int                    | The number of samples to draw from data set for display (default is None).                    | `None`     |
| sort_by_timing     | bool                   | Whether to sort the treatment cohorts by the number of treated periods.                       | `False`    |
| xlab               | str                    | The label for the x-axis. Default is None, in which case default labels are used.             | `None`     |
| ylab               | str                    | The label for the y-axis. Default is None, in which case default labels are used.             | `None`     |
| figsize            | tuple                  | The figure size for the outcome plot. Default is (11, 3).                                     | `(11, 3)`  |
| noticks            | bool                   | Whether to display ticks on the plot. Default is False.                                       | `False`    |
| title              | str                    | The title for the plot. Default is None, in which case no title is displayed.                 | `None`     |
| legend             | bool                   | Whether to display a legend. Default is False (since binary treatments are self-explanatory). | `False`    |
| ax                 | matplotlib.pyplot.Axes | The axes on which to draw the plot. Default is None, in which case a new figure is created.   | `None`     |
| xlim               | tuple                  | The limits for the x-axis of the plot. Default is None.                                       | `None`     |
| ylim               | tuple                  | The limits for the y-axis of the plot. Default is None.                                       | `None`     |
| units_to_plot      | list                   | A list of unit to include in the plot. If None, all units in the dataset are plotted.         | `None`     |

## Returns {.doc-section .doc-section-returns}

| Name   | Type                   | Description   |
|--------|------------------------|---------------|
| ax     | matplotlib.pyplot.Axes |               |

## Examples {.doc-section .doc-section-examples}

```{python}
import pandas as pd
import numpy as np
import pyfixest as pf

url = "https://raw.githubusercontent.com/py-econometrics/pyfixest/master/pyfixest/did/data/df_het.csv"
df_het = pd.read_csv(url)

# Inspect treatment assignment
pf.panelview(
    data = df_het,
    unit = "unit",
    time = "year",
    treat = "treat",
    subsamp = 50,
    title = "Treatment Assignment"
)

# Outcome plot
pf.panelview(
    data = df_het,
    unit = "unit",
    time = "year",
    outcome = "dep_var",
    treat = "treat",
    subsamp = 50,
    title = "Outcome Plot"
)
```