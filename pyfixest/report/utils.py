import re


def _relabel_expvar(varname: str, labels: dict, interaction_symbol: str):
    """
    Relabel a variable name using the labels dictionary
    Also automatically relabel interaction terms using the labels of the individual variables.

    Parameters
    ----------
    varname: str
        The varname in the regression.
    labels: dict
        A dictionary to relabel the variables. The keys are the original variable names and the values the new names.
    interaction_symbol: str
        The symbol to use for displaying the interaction term.

    Returns
    -------
    str
        The relabeled variable
    """
    # When varname in labels dictionary, then relabel (note: this allows also to manually rename interaction terms in the dictionary)
    # Otherwise: When interaction term, then split by vars, relabel, and join using interaction symbol
    if varname in labels:
        return labels[varname]
    elif ":" in varname:
        vars = varname.split(":")
        relabeled_vars = [labels.get(v, v) for v in vars]
        return interaction_symbol.join(relabeled_vars)
    return varname


def _rename_categorical(col_name):
    pattern = r"C\(([^,]+)(?:,[^]]+)?\)\[T\.([^]]+)\]"

    # Apply the regex to extract the variable and level
    match = re.search(pattern, col_name)

    if match:
        variable = match.group(1)
        level = match.group(2)
        result = f"{variable}::{level}"
        return result
    else:
        return col_name


def rename_categoricals(coef_names_list: list) -> dict:
    """
    Rename the categorical variables in the coef_names_list.

    Parameters
    ----------
    coef_names_list: list
        The list of coefficient names.

    Returns
    -------
    dict
        A dictionary with the renamed variables.

    Examples
    --------
    rename_categorical(["C(var)[T.1]", "C(var)[T.2]", "C(var2)[T.1]", "C(var2)[T.2]"])
    {'C(var)[T.1]': 'var::1', 'C(var)[T.2]': 'var::2', 'C(var2)[T.1]': 'var2::1', 'C(var2)[T.2]': 'var2::2'}
    """
    return {col: _rename_categorical(col) for col in coef_names_list}


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
