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
    match = re.match(r"C\(([^)]+)\)\[T\.([\d\.]+)\]", col_name)
    if match:
        return f"{match.group(1)}::{int(float(match.group(2)))}"
    return col_name


def rename_categorical(coef_names_list: list) -> dict:
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
