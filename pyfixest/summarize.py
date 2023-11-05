from pyfixest.feols import Feols
from pyfixest.fepois import Fepois
from pyfixest.feiv import Feiv

import numpy as np
import pandas as pd
from typing import Union, List, Optional


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

    res = pd.concat(etable_list, axis=1).fillna("")
    # separator_row = pd.DataFrame(['_' * 15, '_' * 15, '_' * 15], columns=res.columns)
    # res = pd.concat([res, separator_row], ignore_index=True)

    if type == "tex":
        return res.to_latex()
    elif type == "md":
        res = res.to_markdown()
        print(res)
        print("Significance levels: * p < 0.05, ** p < 0.01, *** p < 0.001")
    else:
        return res


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
        fml = fxst._fml
        split = fml.split("|")

        depvar = split[0].split("~")[0]
        # fxst = [x]

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
                print(
                    f"RMSE: {np.round(fxst._rmse, digits)}  Adj. R2: {np.round(fxst._adj_r2, digits)}  Adj. R2 Within: {np.round(fxst._adj_r2_within, digits)}"
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
