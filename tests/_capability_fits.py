from __future__ import annotations

import pandas as pd
import pytest

import pyfixest as pf

_DID_DATA_PATH = "pyfixest/did/data/df_het.csv"


def capability_fit(model: str, *, fixed_effects: bool = True, **kwargs):
    """Fit one model class for capability tests.

    ``fixed_effects`` adds ``| f1`` to the models that accept fixed effects;
    ``update()`` rejects fits with fixed effects before it reads capabilities.
    """
    data = pf.get_data()
    fe = " | f1" if fixed_effects else ""
    if model == "feols":
        return pf.feols(f"Y ~ X1{fe}", data, **kwargs)
    if model == "feols-iv":
        return pf.feols(f"Y ~ 1{fe} | X1 ~ Z1", data, **kwargs)
    if model == "fepois":
        return pf.fepois(f"Y ~ X1{fe}", pf.get_data(model="Fepois"), **kwargs)
    if model in {"feglm-gaussian", "feglm-logit", "feglm-probit"}:
        data = data.dropna()
        data["Y"] = (data["Y"] > data["Y"].median()).astype(int)
        family = model.removeprefix("feglm-")
        return pf.feglm(f"Y ~ X1{fe}", data, family=family, **kwargs)
    if model == "quantreg":
        with pytest.warns(FutureWarning, match="experimental"):
            return pf.quantreg("Y ~ X1", data, **kwargs)
    did_data = pd.read_csv(_DID_DATA_PATH)
    if model == "did2s":
        return pf.did2s(
            did_data,
            yname="dep_var",
            first_stage="~ 0 | state + year",
            second_stage="~ treat",
            treatment="treat",
            cluster="state",
        )
    if model in {"twfe", "saturated"}:
        return pf.event_study(
            did_data,
            yname="dep_var",
            idname="unit",
            tname="year",
            gname="g",
            estimator=model,
        )
    raise ValueError(model)
