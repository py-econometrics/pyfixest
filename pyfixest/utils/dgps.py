import numpy as np
import pandas as pd


def get_blw():
    """DGP for effect heterogeneity in panel data from Baker, Larcker, and Wang (2022)."""
    n = np.arange(1, 31)
    id_ = np.arange(1, 1001)
    blw = pd.DataFrame(
        [(n_, id__) for n_ in n for id__ in id_], columns=["n", "id"]
    ).eval(
        """
            year = n + 1980 - 1
            state = 1 + (id - 1) // 25
            group = 1 + (state - 1) // 10
            treat_date = 1980 + group * 6
            time_til = year - treat_date
            treat = time_til >= 0
        """
    )
    blw["firms"] = np.random.uniform(0, 5, size=len(blw))
    blw["e"] = np.random.normal(0, 0.5**2, size=len(blw))
    blw["te"] = np.random.normal(10 - 2 * (blw["group"] - 1), 0.2**2, size=len(blw))
    blw.eval(
        """
        y = firms + n + treat * te * (year - treat_date + 1) + e
        y2 = firms + n + te * treat + e
    """,
        inplace=True,
    )
    return blw


# convert this into a single assignment

treat_effect_vector_1 = np.log(2 * np.arange(1, 30 - 15 + 1))
treat_effect_vector_1[8:] = 0


def get_sharkfin(
    num_units=1000,
    num_periods=30,
    num_treated=200,
    treatment_start=15,
    hetfx=False,
    base_treatment_effect=None,
    sigma_unit=2,
    sigma_time=1,
    sigma_epsilon=0.5,
    ar_coef=0.8,
    het_AR=False,
    return_dataframe=True,
    seed=42,
):
    """Sharkfin panel data DGP with heterogeneous treatment effects and effect reversal.

    Args:
        num_units (int, optional): Defaults to 100.
        num_periods (int, optional): Defaults to 30.
        num_treated (int, optional): Defaults to 50.
        treatment_start (int, optional): Defaults to 15.
        hetfx (bool, optional): Heterogeneous effects. Defaults to True.
        base_treatment_effect (_type_, optional): _description_. Defaults to 0.1*np.log(np.arange(1, 30 - 15 + 1)).
        return_dataframe (bool, optional): _description_. Defaults to True.
        sigma_unit (int, optional): _description_. Defaults to 1.
        sigma_time (float, optional): _description_. Defaults to 0.5.
        sigma_epsilon (float, optional): _description_. Defaults to 0.5.
        het_AR (bool, optional): _description_. Defaults to False.
    """
    np.random.seed(seed)
    if base_treatment_effect is None:
        base_treatment_effect = np.where(
            np.arange(1, 30 - 15 + 1) <= 8,
            0.2 * np.log(2 * np.arange(1, 30 - 15 + 1)),
            0,
        )
    unit_intercepts = np.random.normal(0, sigma_unit, num_units)

    # Generate day-of-the-week pattern
    day_effects = np.array(
        [-0.1, 0.1, 0, 0, 0.1, 0.5, 0.5]
    )  # Stronger effects on weekends
    day_pattern = np.tile(day_effects, num_periods // 7 + 1)[:num_periods]

    # Generate autoregressive structure
    ar_coef_time = 0.2
    ar_noise_time = np.random.normal(0, sigma_time, num_periods)
    time_intercepts = np.zeros(num_periods)
    time_intercepts[0] = ar_noise_time[0]
    for t in range(1, num_periods):
        time_intercepts[t] = ar_coef_time * time_intercepts[t - 1] + ar_noise_time[t]
    # Combine day-of-the-week pattern and autoregressive structure
    time_intercepts = day_pattern + time_intercepts - np.mean(time_intercepts)
    # Generate autoregressive noise for each unit
    ar_noise = np.random.normal(0, sigma_epsilon, (num_units, num_periods))
    if het_AR:
        ar_coef = np.random.normal(ar_coef, 0.1, num_units)
    noise = np.zeros((num_units, num_periods))
    noise[:, 0] = ar_noise[:, 0]
    for t in range(1, num_periods):
        noise[:, t] = ar_coef * noise[:, t - 1] + ar_noise[:, t]
    # N X T matrix of potential outcomes under control
    Y0 = unit_intercepts[:, np.newaxis] + time_intercepts[np.newaxis, :] + noise
    # Generate the base treatment effect (concave structure)
    # Generate heterogeneous multipliers for each unit
    if hetfx:
        heterogeneous_multipliers = np.random.uniform(0.5, 1.5, num_units)
    else:
        heterogeneous_multipliers = np.ones(num_units)

    # Create a 2D array to store the heterogeneous treatment effects
    treatment_effect = np.zeros((num_units, num_periods - treatment_start))
    for i in range(num_units):
        treatment_effect[i, :] = heterogeneous_multipliers[i] * base_treatment_effect

    # random assignment
    treated_units = np.random.choice(num_units, num_treated, replace=False)
    treatment_status = np.zeros((num_units, num_periods), dtype=bool)
    treatment_status[treated_units, treatment_start:] = True

    # Apply the heterogeneous treatment effect to the treated units
    Y1 = Y0.copy()
    for t in range(treatment_start, num_periods):
        Y1[:, t][treatment_status[:, t]] += treatment_effect[:, t - treatment_start][
            treatment_status[:, t]
        ]

    result = {
        "Y1": Y1,
        "Y0": Y0,
        "W": treatment_status,
        "unit_intercepts": unit_intercepts,
        "time_intercepts": time_intercepts,
    }

    if return_dataframe:
        # Create a DataFrame
        unit_ids = np.repeat(np.arange(num_units), num_periods)
        time_ids = np.tile(np.arange(num_periods), num_units)
        W_it = treatment_status.flatten()
        Y_it = np.where(W_it, Y1.flatten(), Y0.flatten())
        df = pd.DataFrame(
            {
                "unit": unit_ids,
                "year": time_ids,
                "treat": W_it.astype(int),
                "Y": Y_it,
            }
        )
        # assign units to ever treated if the max of W_it is 1
        df["ever_treated"] = df.groupby("unit")["treat"].transform("max")
        result["dataframe"] = df
        return df
    return result


def get_panel_dgp_stagg(
    num_units=1_000,
    num_periods=30,
    num_treated=None,
    treatment_start_cohorts=None,
    sigma_unit=1,
    sigma_time=0.5,
    sigma_epsilon=0.2,
    hetfx=False,
    base_treatment_effects=None,
    return_dataframe=True,
    ar_coef=0.8,
):
    """Panel DGP with staggered treatment effects and effect heterogeneity."""
    if num_treated is None:
        num_treated = [250, 500, 150]
    if treatment_start_cohorts is None:
        treatment_start_cohorts = [10, 15, 20]
    if base_treatment_effects is None:
        # Cohort 1: mean reversal: big bump that decays to zero within 10 days, then zero
        # Cohort 2: shark-fin - logarithmic for the first week, then 0
        # Cohort 3: sinusoidal
        # effect functions
        treat_effect_vector_1 = np.log(
            2 * np.arange(1, num_periods - treatment_start_cohorts[1] + 1)
        )
        treat_effect_vector_1[8:] = 0  # switch off effects after a week
        base_treatment_effects = [
            np.r_[
                np.linspace(2, 0, num_periods - treatment_start_cohorts[0] - 10),
                np.repeat(0, 10),
            ],
            treat_effect_vector_1,
            np.sin(
                np.arange(1, num_periods - treatment_start_cohorts[2] + 1)
            ),  # Treatment effect function for cohort 2
        ]
    # unit FEs
    unit_intercepts = np.random.normal(0, sigma_unit, num_units)
    ####################################################################
    # time FEs: Generate day-of-the-week pattern
    day_effects = np.array(
        [-0.1, 0.1, 0, 0, 0.1, 0.5, 0.5]
    )  # Stronger effects on weekends
    day_pattern = np.tile(day_effects, num_periods // 7 + 1)[:num_periods]
    # autoregressive structure in time FEs
    ar_coef_time = 0.2
    ar_noise_time = np.random.normal(0, sigma_time, num_periods)
    time_intercepts = np.zeros(num_periods)
    time_intercepts[0] = ar_noise_time[0]
    for t in range(1, num_periods):
        time_intercepts[t] = ar_coef_time * time_intercepts[t - 1] + ar_noise_time[t]
    # Combine day-of-the-week pattern and autoregressive structure
    time_intercepts = day_pattern + time_intercepts - np.mean(time_intercepts)
    ####################################################################
    # Generate autoregressive noise for each unit
    ar_noise = np.random.normal(0, sigma_epsilon, (num_units, num_periods))
    noise = np.zeros((num_units, num_periods))
    noise[:, 0] = ar_noise[:, 0]
    for t in range(1, num_periods):
        noise[:, t] = ar_coef * noise[:, t - 1] + ar_noise[:, t]
    # N X T matrix of potential outcomes under control
    Y0 = unit_intercepts[:, np.newaxis] + time_intercepts[np.newaxis, :] + noise
    ####################################################################
    # Generate heterogeneous multipliers for each unit
    if hetfx:
        heterogeneous_multipliers = np.random.uniform(0.5, 1.5, num_units)
    else:
        heterogeneous_multipliers = np.ones(num_units)
    # random assignment
    treated_units = np.array([], dtype=int)
    treatment_status = np.zeros((num_units, num_periods), dtype=bool)
    ####################################################################
    # Create a 2D array to store the heterogeneous treatment effects
    treatment_effect = np.zeros((num_units, num_periods))
    # iterate over treatment cohorts
    for cohort_idx, (treatment_start, num_treated_cohort) in enumerate(
        zip(treatment_start_cohorts, num_treated)
    ):
        base_treatment_effect = base_treatment_effects[cohort_idx]
        cohort_treatment_effect = np.zeros((num_units, num_periods - treatment_start))

        for i in range(num_units):
            cohort_treatment_effect[i, :] = (
                heterogeneous_multipliers[i] * base_treatment_effect
            )
        cohort_treated_units = np.random.choice(
            np.setdiff1d(np.arange(num_units), treated_units),
            num_treated_cohort,
            replace=False,
        )
        treated_units = np.concatenate((treated_units, cohort_treated_units))
        treatment_status[cohort_treated_units, treatment_start:] = True
        treatment_effect[cohort_treated_units, treatment_start:] += (
            cohort_treatment_effect[cohort_treated_units, :]
        )

    # Apply the heterogeneous treatment effect to the treated units
    Y1 = Y0.copy()
    Y1[treatment_status] += treatment_effect[treatment_status]
    ####################################################################
    result = {
        "Y1": Y1,
        "Y0": Y0,
        "W": treatment_status,
        "unit_intercepts": unit_intercepts,
        "time_intercepts": time_intercepts,
    }

    if return_dataframe:
        # Create a DataFrame
        unit_ids = np.repeat(np.arange(num_units), num_periods)
        time_ids = np.tile(np.arange(num_periods), num_units)
        W_it = treatment_status.flatten().astype(int)
        Y_it = np.where(W_it, Y1.flatten(), Y0.flatten())
        unit_intercepts_flat = np.repeat(unit_intercepts, num_periods)
        time_intercepts_flat = np.tile(time_intercepts, num_units)
        df = pd.DataFrame(
            {
                "unit_id": unit_ids,
                "time_id": time_ids,
                "W_it": W_it,
                "Y_it": Y_it,
                "unit_intercept": unit_intercepts_flat,
                "time_intercept": time_intercepts_flat,
            }
        )
        result["dataframe"] = df
    return result


def gelbach_data(nobs):
    "Create data for testing of Gelbach Decomposition."
    rng = np.random.default_rng(49392)
    df = pd.DataFrame(index=range(nobs))
    df["x1"] = rng.normal(size=nobs)

    df["x21"] = df["x1"] * 1 + rng.normal(loc=0, scale=0.1, size=nobs)
    df["x22"] = df["x1"] * 0.25 + df["x21"] * 0.75 + rng.normal(size=nobs)
    df["x23"] = (
        df["x1"] * 0.4
        + df["x21"] * 0.6
        + df["x22"] * 0.4
        + rng.normal(loc=0, scale=0.1, size=nobs)
    )
    df["y"] = (
        df["x1"] * 1
        + df["x21"] * 2
        + df["x22"] * 0.5
        + df["x23"] * 0.75
        + rng.normal(loc=0, scale=0.1, size=nobs)
    )

    return df
