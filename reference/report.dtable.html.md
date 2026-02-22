# report.dtable { #pyfixest.report.dtable }

```python
report.dtable(
    df,
    vars,
    stats=None,
    bycol=None,
    byrow=None,
    type='gt',
    labels=None,
    stats_labels=None,
    digits=2,
    notes='',
    counts_row_below=False,
    hide_stats=False,
    **kwargs,
)
```

Generate descriptive statistics tables and create a booktab style table in
the desired format (gt or tex).

.. deprecated:: 0.41.0
    This function is deprecated and will be removed in a future version.
    Please use `maketables.DTable()` directly instead.
    See https://py-econometrics.github.io/maketables/ for documentation.

## Parameters {.doc-section .doc-section-parameters}

| Name             | Type         | Description                                                                                                                                                                                               | Default    |
|------------------|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| df               | pd.DataFrame | DataFrame containing the table to be displayed.                                                                                                                                                           | _required_ |
| vars             | list         | List of variables to be included in the table.                                                                                                                                                            | _required_ |
| stats            | list         | List of statistics to be calculated. The default is None, that sets ['count','mean', 'std']. All pandas aggregation functions are supported.                                                              | `None`     |
| bycol            | list         | List of variables to be used to group the data by columns. The default is None.                                                                                                                           | `None`     |
| byrow            | str          | Variable to be used to group the data by rows. The default is None.                                                                                                                                       | `None`     |
| type             | str          | Type of table to be created. The default is 'gt'. Type can be 'gt' for great_tables, 'tex' for LaTeX or 'df' for dataframe.                                                                               | `'gt'`     |
| labels           | dict         | Dictionary containing the labels for the variables. The default is None.                                                                                                                                  | `None`     |
| stats_labels     | dict         | Dictionary containing the labels for the statistics. The default is None. The function uses a default labeling which will be replaced by the labels in the dictionary.                                    | `None`     |
| digits           | int          | Number of decimal places to round the statistics to. The default is 2.                                                                                                                                    | `2`        |
| notes            | str          | Table notes to be displayed at the bottom of the table.                                                                                                                                                   | `''`       |
| counts_row_below | bool         | Whether to display the number of observations at the bottom of the table. Will only be carried out when each var has the same number of obs and when byrow is None. The default is False                  | `False`    |
| hide_stats       | bool         | Whether to hide the names of the statistics in the table header. When stats are hidden and the user provides no notes string the labels of the stats are listed in the table notes. The default is False. | `False`    |
| kwargs           | dict         | Additional arguments to be passed to maketables.DTable.                                                                                                                                                   | `{}`       |

## Returns {.doc-section .doc-section-returns}

| Name   | Type                             | Description   |
|--------|----------------------------------|---------------|
|        | A table in the specified format. |               |

## Examples {.doc-section .doc-section-examples}

For more examples, take a look at the [regression tables and summary statistics vignette](https://pyfixest.org/table-layout.html).

```{python}
import pyfixest as pf

# load data
df = pf.get_data()
pf.dtable(df, vars = ["Y", "X1", "X2", "f1"])
```