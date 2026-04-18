# The following code is adapted from the fixest_benchmarks repository
# by Kyle Butts (https://github.com/kylebutts/fixest_benchmarks?tab=License-1-ov-file)
# Licensed under the MIT License

from __future__ import annotations

import numpy as np
import pandas as pd


def base_dgp(
    n: int = 1000,
    nb_year: int = 10,
    nb_indiv_per_firm: int = 23,
    type_: str = "simple",
    k: int = 1,
    max_k: int = 10,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate the generic modular benchmark panel.

    This DGP is kept for the main benchmark suite only. AKM benchmarks use the
    standalone generator in ``akm_dgp.py``.
    """
    rng = np.random.default_rng(seed)

    nb_indiv = round(n / nb_year)
    nb_firm = round(nb_indiv / nb_indiv_per_firm)

    if nb_indiv < 1 or nb_firm < 1:
        raise ValueError(
            f"n={n} too small for nb_year={nb_year}, "
            f"nb_indiv_per_firm={nb_indiv_per_firm}"
        )

    n_obs = nb_indiv * nb_year

    if k < 1 or k > max_k:
        raise ValueError(f"k={k} must satisfy 1 <= k <= max_k={max_k}")

    indiv_id = np.repeat(np.arange(1, nb_indiv + 1), nb_year)
    year = np.tile(np.arange(1, nb_year + 1), nb_indiv)

    if type_ == "simple":
        firm_id = rng.integers(1, nb_firm + 1, size=n_obs)
    elif type_ == "difficult":
        firm_id = np.tile(np.arange(1, nb_firm + 1), n_obs // nb_firm + 1)[:n_obs]
    else:
        raise ValueError(f"Unknown type of dgp: {type_!r}")

    x = rng.standard_normal((n_obs, max_k))
    betas = 1.0 / np.arange(1, k + 1, dtype=float)

    firm_fe = rng.standard_normal(nb_firm)[firm_id - 1]
    unit_fe = rng.standard_normal(nb_indiv)[indiv_id - 1]
    year_fe = rng.standard_normal(nb_year)[year - 1]
    mu = x[:, :k] @ betas + firm_fe + unit_fe + year_fe
    y = mu + rng.standard_normal(len(mu))

    theta = 0.5
    exp_y = np.exp(y)
    nb_p = theta / (theta + exp_y)
    negbin_y = rng.negative_binomial(n=theta, p=nb_p)

    data = {
        "indiv_id": indiv_id,
        "firm_id": firm_id,
        "year": year,
        "y": y,
        "exp_y": exp_y,
        "negbin_y": negbin_y,
        "binary_y": (y > 0).astype(int),
    }
    for j in range(max_k):
        data[f"x{j + 1}"] = x[:, j]

    return pd.DataFrame(data)
