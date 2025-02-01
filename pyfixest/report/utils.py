import re
from typing import Optional

import pandas as pd


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

    Returns
    -------
    str
        The relabeled variable
    """
    # First split the variable name by the interaction symbol
    # Note: will just be equal to varname when no interaction term
    vars = varname.split(":")
    # Loop over the variables and relabel them
    for i in range(len(vars)):
        # Check whether template for categorical variables is provided &
        # whether the variable is a categorical variable
        v = vars[i]
        if cat_template != "" and ("C(" in v or "[T." in v):
            vars[i] = _rename_categorical(v, template=cat_template, labels=labels)
        else:
            vars[i] = labels.get(v, v)
    # Finally join the variables using the interaction symbol
    return interaction_symbol.join(vars)


def _rename_categorical(
    col_name, template="{variable}::{value}", labels: Optional[dict] = None
):
    # Here two patterns are used to extract the variable and level
    # Note the second pattern matches the notation when the variable is categorical at the outset
    if col_name.startswith("C("):
        pattern = r"C\(([^,]+)(?:,[^]]+)?\)\[T\.([^]]+)\]"
    else:
        pattern = r"([^[]+)\[T\.([^]]+)\]"
    # Replace labels with empty dictionary if not provided
    if labels is None:
        labels = {}
    # Apply the regex to extract the variable and value
    match = re.search(pattern, col_name)
    if match:
        variable = match.group(1)
        # Relabel the variable if it is in the labels dictionary
        variable = labels.get(variable, variable)
        value = match.group(2)
        result = template.format(variable=variable, value=value)
        return result
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
        It can contain the placeholders {variable} and {level}.
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


def set_first_cat(df, var_value_dict):
    """
    Set the first category of the categorical variables in a DataFrame.
    This function is useful to change the reference categories of categorical variables.
    Regressions of the form "y ~ c" will use the first category of c as reference.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame containing the categorical variables.
    var_value_dict: dict
        A dictionary with the variable names as keys and the value to set as first category as values.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the first category set for the categorical variables.
    """
    assert all(column in df.columns for column in var_value_dict), (
        "All keys in var_value_dict must be columns in df."
    )
    assert all(df[column].dtype.name == "category" for column in var_value_dict), (
        "All variables in var_value_dict must be categorical."
    )
    assert all(
        value in df[column].cat.categories for column, value in var_value_dict.items()
    ), (
        "All values in var_value_dict must be in the categories of the corresponding column."
    )
    for column, value in var_value_dict.items():
        df[column] = pd.Categorical(
            df[column],
            categories=[
                value,
                *list(x for x in df[column].cat.categories if x != value),
            ],
            ordered=True,
        )
    return df


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
    return {col: _rename_event_study_coefs(col) for col in coef_names_list}
