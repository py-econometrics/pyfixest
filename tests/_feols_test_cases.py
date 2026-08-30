from __future__ import annotations

import numpy as np
import pandas as pd

ols_fmls = (
    "Y~X1",
    "Y~X1+X2",
    "Y~X1|f2",
    "Y~X1|f2+f3",
    "Y ~ X1 + exp(X2)",
    "Y ~ X1 + C(f1)",
    "Y ~ X1 + i(f1, ref = 1)",
    "Y ~ X1 + i(f2, ref = 2.0)",
    "Y ~ X1 + C(f1) + C(f2)",
    "Y ~ X1 + C(f1) | f2",
    "Y ~ X1 + i(f1, ref = 3.0) | f2",
    "Y ~ X1 + C(f1) | f2 + f3",
    "Y ~ X1 + i(f1, ref = 1) | f2 + f3",
    "Y ~ X1 + i(f1) + i(f2)",
    "Y ~ X1 + i(f1, ref = 1) + i(f2, ref = 2)",
    # C():C() translation is not implemented yet.
    # "Y ~ X1 + C(f1):C(fe2)",
    # "Y ~ X1 + C(f1):C(fe2) | f3",
    "Y ~ X1 + X2:f1",
    "Y ~ X1 + X2:f1 | f3",
    "Y ~ X1 + X2:f1 | f3 + f1",
    # These currently make fepois prohibitively slow.
    # "log(Y) ~ X1:X2 | f3 + f1",
    # "log(Y) ~ log(X1):X2 | f3 + f1",
    # "Y ~  X2 + exp(X1) | f3 + f1",
    "Y ~ X1 + i(f1,X2)",
    "Y ~ X1 + i(f1,X2) + i(f2, X2)",
    "Y ~ X1 + i(f1,X2, ref =1) + i(f2)",
    "Y ~ X1 + i(f1,X2, ref =1) + i(f2, X1, ref =2)",
    "Y ~ X1 + i(f2,X2)",
    "Y ~ X1 + i(f1,X2) | f2",
    "Y ~ X1 + i(f1,X2) | f2 + f3",
    "Y ~ X1 + i(f1,X2, ref=1.0)",
    "Y ~ X1 + i(f2,X2, ref=2.0)",
    "Y ~ X1 + i(f1,X2, ref=3.0) | f2",
    "Y ~ X1 + i(f1,X2, ref=4.0) | f2 + f3",
    # C():X and C():C() translation are not implemented yet.
    # "Y ~ C(f1):X2",
    # "Y ~ C(f1):C(f2)",
    "Y ~ X1 + I(X2 ** 2)",
    "Y ~ X1 + I(X1 ** 2) + I(X2**4)",
    "Y ~ X1*X2",
    "Y ~ X1*X2 | f1+f2",
    # Formulaic does not implement the former X1/X2 translation yet.
    # "Y ~ X1/X2",
    # "Y ~ X1/X2 | f1+f2",
    "Y ~ X1 + poly(X2, 2) | f1",
)


ols_but_not_poisson_fml = (
    "log(Y) ~ X1",
    "Y~X1|f2:f3",
    "Y~X1|f1 + f2:f3",
    "Y~X1|f2:f3:f1",
)


ALL_F3_TYPES = ("str", "object", "int", "categorical", "float")


def _deduplicate(items: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


FEOLS_FORMULAS = _deduplicate(ols_fmls + ols_but_not_poisson_fml)

# Every formula uses the historical string representation. Formulas involving f3
# additionally exercise every supported pandas representation without collecting
# irrelevant combinations that would immediately skip.
FEOLS_FORMULA_F3_CASES = (
    *((fml, "str") for fml in FEOLS_FORMULAS),
    *(
        (fml, f3_type)
        for fml in FEOLS_FORMULAS
        if "f3" in fml
        for f3_type in ALL_F3_TYPES[1:]
    ),
)


def convert_f3(data: pd.DataFrame, f3_type: str) -> pd.DataFrame:
    """Convert f3 to the requested representation."""
    if f3_type == "categorical":
        data["f3"] = pd.Categorical(data["f3"])
    elif f3_type == "int":
        data["f3"] = data["f3"].astype(float).astype(np.int32)
    elif f3_type == "str":
        data["f3"] = data["f3"].astype(str)
    elif f3_type == "object":
        data["f3"] = data["f3"].astype(object)
    elif f3_type == "float":
        data["f3"] = data["f3"].astype(float)
    else:  # pragma: no cover - the case matrix is closed above
        raise ValueError(f"Unsupported f3_type: {f3_type}")
    return data


def build_feols_data_variants(
    base: pd.DataFrame,
) -> dict[tuple[bool, str], pd.DataFrame]:
    """Build the missing-data and factor-type inputs shared by parity suites."""
    variants = {}
    for dropna in (False, True):
        for f3_type in ALL_F3_TYPES:
            data = base.dropna() if dropna else base.copy()
            data.where(data != "nan", np.nan, inplace=True)
            variants[(dropna, f3_type)] = convert_f3(data, f3_type)
    return variants
