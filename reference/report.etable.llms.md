# etable

``` python
etable(
    models,
    type='gt',
    signif_code=None,
    coef_fmt=None,
    custom_stats=None,
    custom_model_stats=None,
    keep=None,
    drop=None,
    exact_match=False,
    labels=None,
    cat_template=None,
    show_fe=True,
    felabels=None,
    fe_present='x',
    fe_absent='-',
    notes='',
    model_heads=None,
    head_order='dh',
    file_name=None,
    docx_style=None,
    **kwargs,
)
```

Generate a table summarizing the results of multiple regression models.

This function uses the maketables package internally to create publication-ready regression tables. It supports various output formats including HTML (via Great Tables), markdown, LaTeX, Typst, and Word.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| models | Feols, Fepois, Feiv, FixestMulti, or list | A fitted model object, or a list of Feols, Fepois, and Feiv models. The models to be summarized in the table. | *required* |
| type | str | Type of output. Either “df” for pandas DataFrame, “md” for markdown, “gt” for great_tables, “tex” for LaTeX, “typst” for Typst, or “docx” for an editable Word document. Default is “gt”. | `'gt'` |
| signif_code | list | Significance levels for the stars. Default is None, which sets \[0.001, 0.01, 0.05\]. Note that stars are only rendered if `coef_fmt` includes `*` on a statistic token, for example `b*` or `b:.3f*`. | `None` |
| coef_fmt | str | The format of the coefficient (b), standard error (se), t-stats (t), and p-value (p). Default is inherited from maketables. To render significance stars, include `*` on the relevant token, following maketables syntax, for example `"b:.3f* \n (se:.3f)"` with 3 decimal places. Spaces , parentheses `()`, brackets `[]`, newlines `\n` are supported. For further available formatting rules, see `maketables.ETable`. | `None` |
| custom_stats | dict \| None | A dictionary of custom statistics that can be used in the coef_fmt string to be displayed in the coefficuent cells analogously to “b”, “se” etc. The keys are the names of the custom statistics, and the values are lists of lists, where each inner list contains the custom statistic values for all coefficients each model. Note that “b”, “se”, “t”, or “p” are reserved and cannot be used as keys. | `None` |
| custom_model_stats | dict \| None | A dictionary of custom model statistics or model information displayed in a new line in the bottom panel of the table. The keys are the names of the statistics (i.e. entry in the first column) and the values are a lists of the same length as the number of models. Default is None. | `None` |
| keep | list \| str \| None | The pattern for retaining coefficient names. You can pass a string (one pattern) or a list (multiple patterns). Default is keeping all coefficients. You should use regular expressions to select coefficients. “age”, \# would keep all coefficients containing age r”^tr”, \# would keep all coefficients starting with tr r”\d\$“, \# would keep all coefficients ending with number Output will be in the order of the patterns. | `None` |
| drop | list \| str \| None | The pattern for excluding coefficient names. You can pass a string (one pattern) or a list (multiple patterns). Syntax is the same as for `keep`. Default is keeping all coefficients. Parameter `keep` and `drop` can be used simultaneously. | `None` |
| exact_match | bool \| None | Whether to use exact match for `keep` and `drop`. Default is False. If True, the pattern will be matched exactly to the coefficient name instead of using regular expressions. | `False` |
| labels | dict \| None | A dictionary to relabel the variables. The keys in this dictionary are the original variable names, which correspond to the names stored in the `_coefnames` attribute of the model. The values in the dictionary are the new names you want to assign to these variables. Note that interaction terms will also be relabeled using the labels of the individual variables. The command is applied after the `keep` and `drop` commands. | `None` |
| cat_template | str \| None | Template to relabel categorical variables. None by default, which applies no relabeling. Other options include combinations of “{variable}” and “{value}”, e.g. “{variable}::{value}” to mimic fixest encoding. But “{variable}–{value}” or “{variable}{value}” or just “{value}” are also possible. | `None` |
| show_fe | bool \| None | Whether to show the rows with fixed effects markers. Default is True. | `True` |
| felabels | dict \| None | A dictionary to relabel the fixed effects. Only needed if you want to relabel the FE lines with a different label than the one specied for the respective variable in the labels dictionary. The command is applied after the `keep` and `drop` commands. | `None` |
| fe_present | str | Symbol to use when a fixed effect is present in a model. Default is “x”. Common alternatives include “Y”, “YES”, “✓”, “✅”, or any custom string. | `'x'` |
| fe_absent | str | Symbol to use when a fixed effect is absent from a model. Default is “-”. Common alternatives include “N”, “NO”, “✗”, ““, or any custom string. | `'-'` |
| digits |  | The number of digits to round to. | *required* |
| thousands_sep |  | The thousands separator. Default is False. | *required* |
| scientific_notation |  | Whether to use scientific notation. Default is True. | *required* |
| scientific_notation_threshold |  | The threshold for using scientific notation. Default is 10_000. | *required* |
| notes | str | Custom table notes. Default shows the significance levels and the format of the coefficient cell. | `''` |
| model_heads | list \| None | Add custom headlines to models when output as df or latex. Length of list must correspond to number of models. Default is None. | `None` |
| head_order | str \| None | String to determine the display of the table header when output as df or latex. Allowed values are “dh”, “hd”, “d”, “h”, or ““. When head_order is”dh”, the dependent variable is displayed first, followed by the custom model_heads (provided the user has specified them). With “hd” it is the other way around. When head_order is “d”, only the dependent variable and model numbers are displayed and with “” only the model numbers. Default is “dh”. | `'dh'` |
| file_name | str \| None | The name/path of the file to save when type is “gt”, “tex”, “typst”, or “docx”. Default is None, which returns the output without saving it. | `None` |
| docx_style | dict\[str, object\] \| None | Word style overrides, used only when type is “docx”. Passed to `maketables.MTable.make` as `docx_style`. For example, `{"font_name": "Arial", "font_size_pt": 12, "notes_font_size_pt": 9}`. See `maketables.MTable.DEFAULT_DOCX_STYLE` for supported options. Default is None, which uses maketables’ Word style defaults. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | pandas.DataFrame, great_tables.GT, docx.document.Document, str, or None | A DataFrame for “df”, a GT object for “gt”, a string for “html”, “tex”, or “typst”, or a Word Document for “docx”. Markdown output is printed and returns None. Word output is returned even when saved via `file_name`, so it can be edited further with python-docx. |

## Examples

For more examples, see the [regression tables tutorial](../tutorials/regression-tables.llms.md).

``` python
import pyfixest as pf

# load data
df = pf.get_data()
fit1 = pf.feols("Y~X1 + X2 | f1", df)
fit2 = pf.feols("Y~X1 + X2 | f1 + f2", df)

pf.etable([fit1, fit2])
```

[TABLE]

``` python
document = pf.etable(
    [fit1, fit2],
    type="docx",
    docx_style={"font_name": "Arial", "font_size_pt": 12},
)
# Save directly with file_name="results.docx", or edit the document first.
document.add_paragraph("Additional discussion of the estimates.")
```

    <docx.text.paragraph.Paragraph at 0x7ff6f830a690>
