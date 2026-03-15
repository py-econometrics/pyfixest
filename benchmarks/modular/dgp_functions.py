from __future__ import annotations

import numpy as np
import pandas as pd


def base_dgp(
    n: int = 1000,
    nb_year: int = 10,
    nb_indiv_per_firm: int = 23,
    type_: str = "simple",
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

    indiv_id = np.repeat(np.arange(1, nb_indiv + 1), nb_year)
    year = np.tile(np.arange(1, nb_year + 1), nb_indiv)

    if type_ == "simple":
        firm_id = rng.integers(1, nb_firm + 1, size=n_obs)
    elif type_ == "difficult":
        firm_id = np.tile(np.arange(1, nb_firm + 1), n_obs // nb_firm + 1)[:n_obs]
    else:
        raise ValueError(f"Unknown type of dgp: {type_!r}")

    x1 = rng.standard_normal(n_obs)
    x2 = x1**2

    firm_fe = rng.standard_normal(nb_firm)[firm_id - 1]
    unit_fe = rng.standard_normal(nb_indiv)[indiv_id - 1]
    year_fe = rng.standard_normal(nb_year)[year - 1]
    mu = 1 * x1 + 0.05 * x2 + firm_fe + unit_fe + year_fe
    y = mu + rng.standard_normal(len(mu))

    theta = 0.5
    exp_y = np.exp(y)
    nb_p = theta / (theta + exp_y)
    negbin_y = rng.negative_binomial(n=theta, p=nb_p)

    return pd.DataFrame(
        {
            "indiv_id": indiv_id,
            "firm_id": firm_id,
            "year": year,
            "x1": x1,
            "x2": x2,
            "y": y,
            "exp_y": exp_y,
            "negbin_y": negbin_y,
            "binary_y": (y > 0).astype(int),
        }
    )
