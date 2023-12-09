from pyfixest.feols import Feols
from pyfixest.fepois import Fepois
from pyfixest.feiv import Feiv

import numpy as np
import pandas as pd
from typing import Union, List, Optional
from tabulate import tabulate


def etable(
    models: Union[Feols, Fepois, Feiv, List],
    digits: Optional[int] = 3,
    type: Optional[str] = "md",
) -> Union[pd.DataFrame, str]:
    """
    Create an esttab-like table from a list of models#
    Args:
        models: A list of models of type Feols, Feiv, Fepois.
        digits: Number of digits to round to.
        type: Type of output. Either "df" for pandas DataFrame, "md" for markdown, or "tex" for LaTeX table. "md" by default.
    Returns:
        A pandas DataFrame with the coefficients and standard errors of the models.
    """

    models = _post_processing_input_checks(models)

    assert digits >= 0, "digits must be a positive integer"
    assert type in ["df", "tex", "md"], "type must be either 'df', 'md' or 'tex'"

    dep_var_list = []
    nobs_list = []
    fixef_list = []
    nobs_list = []
    n_coefs = []
    se_type_list = []
    r2_list = []
    r2_within_list = []

    for i, model in enumerate(models):
        dep_var_list.append(model._fml.split("~")[0])
        n_coefs.append(len(model._coefnames))
        nobs_list.append(model._N)

        if model._vcov_type == "CRV":
            se_type_list.append("by: " + "+".join(model._clustervar))
        else:
            se_type_list.append(model._vcov_type)

        if model._fixef is not None:
            fixef_list += model._fixef.split("+")

    # find all fixef variables
    fixef_list = list(set(fixef_list))
    n_fixef = len(fixef_list)
    max_coefs = max(n_coefs)

    # create a pd.dataframe with the depvar, nobs, and fixef as keys
    nobs_fixef_df = pd.DataFrame({"Observations": nobs_list, "S.E. type": se_type_list})

    if fixef_list:  # only when at least one model has a fixed effect
        for fixef in fixef_list:
            nobs_fixef_df[fixef] = "-"

            for i, model in enumerate(models):
                if model._fixef is not None:
                    if fixef in model._fixef.split("+"):
                        nobs_fixef_df.loc[i, fixef] = "x"

    colnames = nobs_fixef_df.columns.tolist()
    colnames_reordered = colnames[2:] + colnames[:2]
    nobs_fixef_df = nobs_fixef_df[colnames_reordered].T.reset_index()

    etable_list = []
    for i, model in enumerate(models):
        model = model.tidy().reset_index().round(digits)
        model["stars"] = np.where(
            model["Pr(>|t|)"] < 0.001,
            "***",
            np.where(
                model["Pr(>|t|)"] < 0.01,
                "**",
                np.where(model["Pr(>|t|)"] < 0.05, "*", ""),
            ),
        )
        model["Estimate (Std. Error)"] = pd.Categorical(
            model.apply(
                lambda row: f"{row['Estimate']}{row['stars']} ({row['Std. Error']})",
                axis=1,
            )
        )
        model = model[["Coefficient", "Estimate (Std. Error)"]]
        model = pd.melt(
            model, id_vars=["Coefficient"], var_name="Metric", value_name=f"est{i+1}"
        )
        model = model.drop("Metric", axis=1).set_index("Coefficient")
        etable_list.append(model)

    res = pd.concat(etable_list, axis=1).fillna("").reset_index()

    res.rename(columns={"Coefficient": "index"}, inplace=True)
    nobs_fixef_df.columns = res.columns

    depvars = pd.DataFrame({"depvar": dep_var_list}).T.reset_index()
    depvars.columns = res.columns

    res_all = pd.concat([depvars, res, nobs_fixef_df], ignore_index=True)
    res_all.columns = [""] + list(res_all.columns[1:])

    if type == "tex":
        return res_all.to_latex()
    elif type == "md":
        res_all = _tabulate_etable(res_all, len(models), max_coefs, n_fixef)
        print(res_all)
        print("Significance levels: * p < 0.05, ** p < 0.01, *** p < 0.001")
    else:
        return res_all


def summary(
    models: Union[Feols, Fepois, Feiv, List], digits: Optional[int] = 3
) -> None:
    """
    # Summary

    Prints a summary of the feols() estimation results for each estimated model.

    For each model, the method prints a header indicating the fixed-effects and the
    dependent variable, followed by a table of coefficient estimates with standard
    errors, t-values, and p-values.

    Args:
        digits (int, optional): The number of decimal places to round the summary statistics to. Default is 3.

    Returns:
        None
    """

    models = _post_processing_input_checks(models)

    for fxst in list(models):
        depvar = fxst._depvar

        df = fxst.tidy().round(digits)

        if fxst._method == "feols":
            if fxst._is_iv:
                estimation_method = "IV"
            else:
                estimation_method = "OLS"
        elif fxst._method == "fepois":
            estimation_method = "Poisson"
        elif fxst._method == "twfe":
            estimation_method = "TWFE"
        elif fxst._method == "did2s":
            estimation_method = "DID2S"
        else:
            raise ValueError("Unknown estimation method.")

        print("###")
        print("")
        print("Estimation: ", estimation_method)
        depvar_fixef = f"Dep. var.: {depvar}"
        if fxst._fixef is not None:
            depvar_fixef += f", Fixed effects: {fxst._fixef}"
        print(depvar_fixef)
        print("Inference: ", fxst._vcov_type_detail)
        print("Observations: ", fxst._N)
        print("")
        print(df.to_markdown(floatfmt=f".{digits}f"))
        print("---")
        if fxst._method == "feols":
            if not fxst._is_iv:
                if fxst._has_fixef:
                    print(
                        f"RMSE: {np.round(fxst._rmse, digits)}   R2: {np.round(fxst._r2, digits)}   R2 Within: {np.round(fxst._r2_within, digits)}"
                    )
                else:
                    print(
                        f"RMSE: {np.round(fxst._rmse, digits)}   R2: {np.round(fxst._r2, digits)}"
                    )
        elif fxst._method == "fepois":
            print(f"Deviance: {np.round(fxst.deviance[0], digits)}")
        else:
            pass


def _post_processing_input_checks(models):
    # check if models instance of Feols or Fepois
    if isinstance(models, (Feols, Fepois)):
        models = [models]
    else:
        if isinstance(models, list):
            for model in models:
                if not isinstance(model, (Feols, Fepois)):
                    raise TypeError(
                        """
                        The models argument must be either a list of Feols or Fepois instances,
                        a dict of Feols or Fepois instances, or simply a Feols or Fepois instance.
                        """
                    )
        elif isinstance(models, dict):
            for model in models.keys():
                if not isinstance(models[model], (Feols, Fepois)):
                    raise TypeError(
                        "The models argument must be a list of Feols or Fepois instances."
                    )

    return models


def _tabulate_etable(df, n_models, max_covariates, n_fixef):
    # Format the DataFrame for tabulate
    table = tabulate(
        df, headers="keys", showindex=False, colalign=["left"] + n_models * ["right"]
    )

    # Split the table into header and body
    header, body = table.split("\n", 1)

    # Add separating line after the third row
    body_lines = body.split("\n")
    body_lines.insert(2, "-" * len(body_lines[0]))
    body_lines.insert(-2 - n_fixef, "-" * len(body_lines[0]))
    body_lines.insert(-2, "-" * len(body_lines[0]))
    body_lines.append("-" * len(body_lines[0]))

    # Join the lines back together
    formatted_table = "\n".join([header, "\n".join(body_lines)])

    # Print the formatted table
    return formatted_table
