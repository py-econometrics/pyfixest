import re
import warnings
from typing import Optional


def _check_label_keys_in_covars(label_keys: list[str], covariate_names: list[str]):
    for label_key in label_keys:
        if label_key not in covariate_names:
            warnings.warn(f"The label key '{label_key}' is not in the covariate names.")


def _relabel_expvar(
    varname: str, labels: dict, interaction_symbol: str, cat_template=""
):
    """
    Relabel a variable name using the labels dictionary
    Also automatically relabel interaction terms using the labels of the individual variables
    and categorical variables using the cat_template.

    Parameters
    ----------
    varname: str
        The varname in the regression.
    labels: dict
        A dictionary to relabel the variables. The keys are the original variable names and the values the new names.
    interaction_symbol: str
        The symbol to use for displaying the interaction term.
    cat_template: str
        Template to relabel categorical variables. When empty, the function will not relabel categorical variables.
        You can use {variable}, {value}, or {value_int} placeholders.
        e.g. "{variable}::{value_int}" if you want to force integer format when possible.

    Returns
    -------
    str
        The relabeled variable
    """
    if not labels and cat_template != "" and ("C(" in varname or "[" in varname):
        v = _rename_categorical(varname, template=cat_template, labels=labels)
    else:
        v = labels.get(varname, varname)

    return v.replace(interaction_symbol, ":")


def _rename_categorical(
    col_name, template="{variable}::{value}", labels: Optional[dict] = None
):
    """
    Rename categorical variables, optionally converting floats to ints in the category label.

    Parameters
    ----------
    col_name : str
        A single coefficient string (e.g. "C(var)[T.1]").
    template: str, optional
        String template for formatting. You can use {variable}, {value}, or {value_int} placeholders.
        e.g. "{variable}::{value_int}" if you want to force integer format when possible.
    labels: dict, optional
        Dictionary that replaces variable names with user-specified labels.

    Returns
    -------
    str
        The renamed categorical variable.
    """
    # Here two patterns are used to extract the variable and level
    # Note the second pattern matches the notation when the variable is categorical at the outset
    if col_name.startswith("C("):
        pattern = r"C\(([^,]+)(?:,[^]]+)?\)\[(?:T\.)?([^]]+)\]"
    else:
        pattern = r"([^[]+)\[(?:T\.)?([^]]+)\]"

    # Replace labels with empty dictionary if not provided
    if labels is None:
        labels = {}
    # Apply the regex to extract the variable and value
    match = re.search(pattern, col_name)
    if match:
        variable = match.group(1)
        variable = labels.get(variable, variable)  # apply label if any
        value_raw = match.group(2)

        # Try parsing as float so that e.g. "2.0" can become "2"
        value_int = value_raw
        try:
            numeric_val = float(value_raw)
            value_int = int(numeric_val) if numeric_val.is_integer() else numeric_val
        except ValueError:
            # If not numeric at all, we'll leave it as-is
            pass

        return template.format(variable=variable, value=value_raw, value_int=value_int)
    else:
        return col_name


def rename_categoricals(
    coef_names_list: list, template="{variable}::{value}", labels: Optional[dict] = None
) -> dict:
    """
    Rename the categorical variables in the coef_names_list.

    Parameters
    ----------
    coef_names_list: list
        The list of coefficient names.
    template: str
        The template to use for renaming the categorical variables.
        You can use {variable}, {value}, or {value_int} placeholders.
        e.g. "{variable}::{value_int}" if you want to force integer format when possible.
        or "{value_int}" if you only want to display integers
    labels: dict
        A dictionary with the variable names as keys and the new names as values.

    Returns
    -------
    dict
        A dictionary with the renamed variables.

    Examples
    --------
    rename_categorical(["C(var)[T.1]", "C(var)[T.2]", "C(var2)[T.1]", "C(var2)[T.2]"])
    {'C(var)[T.1]': 'var::1', 'C(var)[T.2]': 'var::2', 'C(var2)[T.1]': 'var2::1', 'C(var2)[T.2]': 'var2::2'}
    """
    return {
        col: _rename_categorical(col, template=template, labels=labels)
        for col in coef_names_list
    }


def _rename_event_study_coefs(col_name: str):
    pattern = r"C\(([^,]+)(?:,[^]]+)?\)\[T\.(-?\d+(?:\.\d+)?)\]"
    match = re.search(pattern, col_name)

    if match:
        variable = match.group(1)
        level = match.group(2)
        result = f"{variable}::{level}"
        return result
    else:
        return col_name


def rename_event_study_coefs(coef_names_list: list):
    """
    Rename the event study coefficients in the coef_names_list.

    Parameters
    ----------
    coef_names_list: list
        The list of coefficient names to rename.

    Returns
    -------
    dict
        A dictionary with the renamed variables ready to be passed
        to the "labels" argument of coefplot() and iplot().

    Examples
    --------
    rename_event_study_coefs(["C(rel_year, contr.treatment(base=-1.0))[T.-20.0]", "C(rel_year, contr.treatment(base=-1.0))[T.-19.0]", "Intercept"])
        {
            'C(rel_year, contr.treatment(base=-1.0))[T.-20.0]': 'rel_year::-20.0',
            'C(rel_year, contr.treatment(base=-1.0))[T.-19.0]': 'rel_year::-19.0',
            'Intercept': 'Intercept'
        }
    """
    # add a deprecation warning
    warnings.warn(
        "rename_event_study_coefs is deprecated and will be removed in a future version. Please use the cat_template argument instead."
    )

    return {col: _rename_event_study_coefs(col) for col in coef_names_list}
