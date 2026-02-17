import re
import warnings
from collections import Counter
from collections.abc import ValuesView
from typing import Optional, Union

from pyfixest.estimation.models.feiv_ import Feiv
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.models.fepois_ import Fepois
from pyfixest.estimation.FixestMulti_ import FixestMulti

ModelInputType = Union[
    FixestMulti, Feols, Fepois, Feiv, list[Union[Feols, Fepois, Feiv]]
]


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
    warnings.warn(
        "The function `_relabel_expvar` is deprecated, please rely on the `labels` and `cat_template` arguments of `pf.etable()` instead. ",
        DeprecationWarning,
        stacklevel=2,
    )

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
    warnings.warn(
        "The function `_relabel_expvar` is deprecated, please rely on the `labels` and `cat_template` arguments of `pf.etable()` instead. ",
        DeprecationWarning,
        stacklevel=2,
    )

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
    warnings.warn(
        "The function `_relabel_expvar` is deprecated as we have adjusted the naming of variables interacted via the i() operator with pyfixest 0.50. "
        "For regression tables, please rely on the `labels` and `cat_template` arguments of `pf.etable()` instead. ",
        DeprecationWarning,
        stacklevel=2,
    )

    return {
        col: _rename_categorical(col, template=template, labels=labels)
        for col in coef_names_list
    }


def _rename_event_study_coefs(col_name: str):
    warnings.warn(
        "The function `_relabel_expvar` is deprecated as we have adjusted the naming of variables interacted via the i() operator with pyfixest 0.50. "
        "For regression tables, please rely on the `labels` and `cat_template` arguments of `pf.etable()` instead. ",
        DeprecationWarning,
        stacklevel=2,
    )

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
    warnings.warn(
        "The function `_relabel_expvar` is deprecated as we have adjusted the naming of variables interacted via the i() operator with pyfixest 0.50. "
        "For regression tables, please rely on the `labels` and `cat_template` arguments of `pf.etable()` instead. ",
        DeprecationWarning,
        stacklevel=2,
    )

    return {col: _rename_event_study_coefs(col) for col in coef_names_list}


def _post_processing_input_checks(
    models: ModelInputType,
    check_duplicate_model_names: bool = False,
    rename_models: Optional[dict[str, str]] = None,
) -> list[Union[Feols, Fepois, Feiv]]:
    """
    Perform input checks for post-processing models.

    Parameters
    ----------
        models : Union[List[Union[Feols, Fepois, Feiv]], FixestMulti]
                The models to be checked. This can either be a list of models
                (Feols, Fepois, Feiv) or a single FixestMulti object.
        check_duplicate_model_names : bool, optional
                Whether to check for duplicate model names. Default is False.
                Mostly used to avoid overlapping models in plots created via
                pf.coefplot() and pf.iplot().
        rename_models : dict, optional
                A dictionary to rename the models. The keys are the original model names
                and the values are the new model names.

    Returns
    -------
        List[Union[Feols, Fepois]]
            A list of checked and validated models. The returned list contains only
            Feols and Fepois types.

    Raises
    ------
        TypeError: If the models argument is not of the expected type.

    """
    models_list: list[Union[Feols, Fepois, Feiv]] = []

    if isinstance(models, (Feols, Fepois, Feiv)):
        models_list = [models]
    elif isinstance(models, FixestMulti):
        models_list = models.to_list()
    elif isinstance(models, (list, ValuesView)):
        if all(isinstance(m, (Feols, Fepois, Feiv)) for m in models):
            models_list = models
        else:
            raise TypeError(
                "All elements in the models list must be instances of Feols, Feiv, or Fepois."
            )
    else:
        raise TypeError("Invalid type for models argument.")

    if check_duplicate_model_names or rename_models is not None:
        all_model_names = [model._model_name for model in models_list]

    if check_duplicate_model_names:
        # create model_name_plot attribute to differentiate between models with the
        # same model_name / model formula
        for model in models_list:
            model._model_name_plot = model._model_name

        counter = Counter(all_model_names)
        duplicate_model_names = [item for item, count in counter.items() if count > 1]

        for duplicate_model in duplicate_model_names:
            duplicates = [
                model for model in models_list if model._model_name == duplicate_model
            ]
            for i, model in enumerate(duplicates):
                model._model_name_plot = f"Model {i}: {model._model_name}"
                warnings.warn(
                    f"The _model_name attribute {model._model_name}' is duplicated for models in the `models` you provided. To avoid overlapping model names / plots, the _model_name_plot attribute has been changed to '{model._model_name_plot}'."
                )

        if rename_models is not None:
            model_name_diff = set(rename_models.keys()) - set(all_model_names)
            if model_name_diff:
                warnings.warn(
                    f"""
                    The following model names specified in rename_models are not found in the models:
                    {model_name_diff}
                    """
                )

    return models_list
