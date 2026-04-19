import numpy as np
import pandas as pd


def get_ivf_data(N=2000, seed=1234):
    """
    Synthetic data for the motherhood penalty IV application (IVF instrument).

    DGP
    ---
    Unobserved confounder: career_ambition ~ N(0, 1)

    First stage (num_children on ivf_success):
        num_children = 1.2 - 0.4*career_ambition + 0.8*ivf_success + N(0, 0.5)
        → ivf_success is relevant (first-stage coefficient ≈ 0.8, F >> 10)

    Outcome (structural equation):
        earnings = 10 + 0.6*career_ambition + TRUE_EFFECT*num_children + N(0, 1)
        TRUE_EFFECT = -0.15

    OVB formula for naive OLS (earnings ~ num_children):
        bias ≈ gamma_ambition * Cov(num_children, ambition) / Var(num_children)
             ≈ 0.6 * (-0.4) / 0.57 ≈ -0.42
        beta_OLS ≈ -0.15 + (-0.42) ≈ -0.57   (overstates the penalty)
        beta_IV  ≈ -0.15                        (recovers the true effect)

    Parameters
    ----------
    N : int, optional
        Number of observations. Default is 2000.
    seed : int, optional
        Random seed. Default is 1234.

    Returns
    -------
    pandas.DataFrame
        Columns: ``earnings``, ``num_children``, ``ivf_success``.
    """
    # --- DGP parameters ---
    true_effect = -0.15  # causal effect of num_children on earnings
    ambition_on_children = -0.4  # confounder → endogenous var (creates OVB)
    ambition_on_earnings = 0.6  # confounder → outcome (creates OVB)
    ivf_on_children = 0.8  # first-stage strength (instrument → endogenous)

    rng = np.random.default_rng(seed)
    career_ambition = rng.normal(0, 1, N)
    ivf_success = rng.binomial(1, 0.45, N)
    num_children = np.clip(
        1.2
        + ambition_on_children * career_ambition
        + ivf_on_children * ivf_success
        + rng.normal(0, 0.5, N),
        0,
        None,
    )
    earnings = (
        10
        + ambition_on_earnings * career_ambition
        + true_effect * num_children
        + rng.normal(0, 1, N)
    )
    return pd.DataFrame(
        {
            "earnings": earnings,
            "num_children": num_children,
            "ivf_success": ivf_success,
        }
    )


def get_bartik_data(N=300, seed=1234):
    """
    Synthetic data for a Bartik (shift-share) IV application on immigration and wages.

    DGP
    ---
    Unobserved confounder: local_demand ~ N(0, 1)

    First stage (immigration on bartik_instrument, conditional on log_population):
        immigration = 0.5 + 0.7*bartik_instrument + 0.9*local_demand + N(0, 0.5)
        → bartik_instrument is relevant; bartik ⊥ local_demand (exogenous)

    Outcome (structural equation):
        wages = 8 + 0.5*local_demand + TRUE_EFFECT*immigration + 0.2*log_population + N(0, 1)
        TRUE_EFFECT = -0.3

    OVB for naive OLS (wages ~ immigration + log_population):
        Partial bias from local_demand ≈ 0.5 * 0.9/Var(immigration|log_pop) > 0
        β_OLS on immigration ≈ -0.3 + positive_bias → attenuated (less negative or positive)
        β_IV  on immigration ≈ -0.3  (recovers the true effect)

    Parameters
    ----------
    N : int, optional
        Number of observations (regions). Default is 300.
    seed : int, optional
        Random seed. Default is 1234.

    Returns
    -------
    pandas.DataFrame
        Columns: ``wages``, ``immigration``, ``log_population``, ``bartik_instrument``.
    """
    # --- DGP parameters ---
    true_effect = -0.3  # causal effect of immigration on wages
    demand_on_immig = 0.9  # confounder → endogenous var
    demand_on_wages = 0.5  # confounder → outcome (creates positive OVB)
    bartik_on_immig = 0.7  # first-stage strength

    rng = np.random.default_rng(seed)
    local_demand = rng.normal(0, 1, N)
    bartik_instrument = rng.normal(0, 1, N)
    log_population = 2 + 0.1 * local_demand + rng.normal(0, 0.3, N)
    immigration = (
        0.5
        + bartik_on_immig * bartik_instrument
        + demand_on_immig * local_demand
        + rng.normal(0, 0.5, N)
    )
    wages = (
        8
        + demand_on_wages * local_demand
        + true_effect * immigration
        + 0.2 * log_population
        + rng.normal(0, 1, N)
    )
    return pd.DataFrame(
        {
            "wages": wages,
            "immigration": immigration,
            "log_population": log_population,
            "bartik_instrument": bartik_instrument,
        }
    )


def get_encouragement_data(N=4000, seed=1234):
    """
    Synthetic data for an A/B encouragement design IV application.

    DGP
    ---
    Instrument: assigned_treatment ~ Bernoulli(0.5)  [randomized, exogenous]
    Fixed effect: user_type ∈ {0, 1, 2}

    First stage (compliance):
        P(adopt | encouraged)     = 0.70  (compliers + always-takers)
        P(adopt | not encouraged) = 0.15  (always-takers only)
        First-stage coefficient   = 0.70 - 0.15 = 0.55

    Outcome (structural equation):
        revenue = 5 + user_type_FE + TRUE_LATE*adopted_feature + N(0, 1)
        TRUE_LATE = 2.0  (effect on compliers)

    Wald identity (exact by construction):
        ITT  = E[Y|Z=1] - E[Y|Z=0] = 2.0 * 0.55 = 1.10
        LATE = ITT / first_stage   = 1.10 / 0.55 = 2.0  ✓

    Parameters
    ----------
    N : int, optional
        Number of observations (users). Default is 4000.
    seed : int, optional
        Random seed. Default is 1234.

    Returns
    -------
    pandas.DataFrame
        Columns: ``revenue``, ``assigned_treatment``, ``adopted_feature``, ``user_type``.
    """
    # --- DGP parameters ---
    true_late = 2.0  # LATE: causal effect of adoption on revenue for compliers
    p_adopt_encouraged = 0.70  # P(adopt | Z=1): compliers + always-takers
    p_adopt_control = 0.15  # P(adopt | Z=0): always-takers only
    # first_stage = p_adopt_encouraged - p_adopt_control = 0.55
    # ITT         = true_late * first_stage              = 1.10
    # LATE        = ITT / first_stage                    = 2.0

    rng = np.random.default_rng(seed)
    user_type = rng.choice([0, 1, 2], size=N)
    user_type_effect = np.array([0.0, 1.0, -0.5])[user_type]
    assigned_treatment = rng.binomial(1, 0.5, N)
    p_adopt = np.where(assigned_treatment == 1, p_adopt_encouraged, p_adopt_control)
    adopted_feature = rng.binomial(1, p_adopt, N)
    revenue = 5 + user_type_effect + true_late * adopted_feature + rng.normal(0, 1, N)
    return pd.DataFrame(
        {
            "revenue": revenue,
            "assigned_treatment": assigned_treatment,
            "adopted_feature": adopted_feature,
            "user_type": pd.Categorical(user_type),
        }
    )


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
        zip(treatment_start_cohorts, num_treated, strict=True)
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


def get_twin_data(N_pairs=500, seed=42):
    """Generate twin study data for returns to education.

    Inspired by Ashenfelter & Krueger (1994). Each twin pair shares an
    unobserved ``ability`` component. The true return to education is 0.08
    log-points per year; naive OLS is biased upward because ability is
    correlated with both education and wages.

    Parameters
    ----------
    N_pairs : int
        Number of twin pairs.  Total observations = 2 * N_pairs.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: twin_pair_id, twin_id, ability, educ, age, experience,
        log_wage.
    """
    rng = np.random.default_rng(seed)

    twin_pair_id = np.repeat(np.arange(1, N_pairs + 1), 2)
    twin_id = np.tile([1, 2], N_pairs)

    # Shared unobserved ability within each pair
    ability_pair = rng.normal(0, 1, size=N_pairs)
    ability = np.repeat(ability_pair, 2)

    # Education: correlated with ability (source of OV bias)
    educ = 12 + 0.5 * ability + rng.normal(0, 2, size=2 * N_pairs)
    educ = np.clip(educ, 8, 20)  # realistic bounds

    # Age and experience
    age = rng.integers(25, 55, size=2 * N_pairs).astype(float)
    experience = np.maximum(0, age - educ - 6)

    # Log wage: true return to education = 0.08
    log_wage = (
        1.5
        + 0.08 * educ
        + 0.3 * ability
        + 0.02 * experience
        + rng.normal(0, 0.3, size=2 * N_pairs)
    )

    return pd.DataFrame(
        {
            "twin_pair_id": twin_pair_id,
            "twin_id": twin_id,
            "ability": ability,
            "educ": educ,
            "age": age,
            "experience": experience,
            "log_wage": log_wage,
        }
    )


def get_worker_panel(N_workers=500, N_firms=50, N_years=11, seed=42):
    """Generate a worker-firm panel dataset with two-way fixed effects.

    Inspired by Abowd, Kramarz & Margolis (1999).  Workers switch firms
    with ~20 % probability each year.  Both worker and firm fixed effects
    contribute to wages.

    Parameters
    ----------
    N_workers : int
        Number of workers.
    N_firms : int
        Number of firms.
    N_years : int
        Number of years in the panel (starting from 2000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: worker_id, firm_id, year, female, experience, tenure,
        log_wage, worker_fe, firm_fe.
    """
    rng = np.random.default_rng(seed)

    # Worker and firm fixed effects
    worker_fe = rng.normal(0, 0.5, size=N_workers)
    firm_fe = rng.normal(0, 0.3, size=N_firms)

    # Time-invariant worker characteristics
    female = rng.binomial(1, 0.5, size=N_workers)

    # Initial firm assignment
    firm_assignment = rng.integers(0, N_firms, size=N_workers)

    records = []
    tenure_counter = np.ones(N_workers, dtype=int)

    for t in range(N_years):
        year = 2000 + t

        # ~20% of workers switch firms each year (after year 0)
        if t > 0:
            switchers = rng.random(N_workers) < 0.20
            firm_assignment[switchers] = rng.integers(0, N_firms, size=switchers.sum())
            tenure_counter[switchers] = 1
            tenure_counter[~switchers] += 1

        experience = t + rng.integers(0, 5, size=N_workers)

        log_wage = (
            worker_fe
            + firm_fe[firm_assignment]
            + 0.02 * experience
            + 0.01 * tenure_counter
            - 0.05 * female
            + rng.normal(0, 0.2, size=N_workers)
        )

        for i in range(N_workers):
            records.append(
                {
                    "worker_id": i,
                    "firm_id": firm_assignment[i],
                    "year": year,
                    "female": female[i],
                    "experience": experience[i],
                    "tenure": tenure_counter[i],
                    "log_wage": log_wage[i],
                    "worker_fe": worker_fe[i],
                    "firm_fe": firm_fe[firm_assignment[i]],
                }
            )

    return pd.DataFrame(records)


def get_motherhood_event_study_data(
    n_per_country=280,
    start_year=2000,
    end_year=2020,
    seed=2026,
):
    """Generate a fertility-timing panel for motherhood-penalty event studies.

    The DGP encodes:
    - stronger post-birth penalties in DACH than in Scandinavia
    - endogenous fertility timing: slower career trajectories lead to earlier births
    - nontrivial share of never-treated units

    Parameters
    ----------
    n_per_country : int
        Number of units per country.
    start_year : int
        First year in the panel.
    end_year : int
        Last year in the panel (inclusive).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: unit, country, region, year, g, treat, rel_year, log_earnings
    """
    rng = np.random.default_rng(seed)

    years = np.arange(start_year, end_year + 1)
    countries_dach = ["DE", "AT", "CH"]
    countries_scand = ["DK", "SE", "NO"]
    all_countries = countries_dach + countries_scand

    country_fe = {
        "DE": 0.00,
        "AT": -0.04,
        "CH": 0.03,
        "DK": 0.02,
        "SE": 0.01,
        "NO": 0.04,
    }

    records = []
    unit = 0

    for country in all_countries:
        region = "DACH" if country in countries_dach else "Scandinavia"

        for _ in range(n_per_country):
            unit += 1

            ability = rng.normal(0, 0.25)
            career_slowdown = rng.normal(0, 1)
            trend_i = 0.020 - 0.010 * career_slowdown

            # Most women eventually have a first birth, but never-treated units remain.
            has_birth = rng.random() < 0.78
            if has_birth:
                base_g = rng.integers(2006, 2015)
                g = int(np.clip(np.round(base_g - 0.9 * career_slowdown), 2003, 2017))
            else:
                g = 0

            for year in years:
                rel_year = year - g if g > 0 else -999
                treat = int(g > 0 and year >= g)

                # Event-time dip before birth: fertility occurs as career growth slows.
                pretrend = 0.0
                if g > 0 and -4 <= rel_year <= -1:
                    pretrend = -0.025 * (rel_year + 4)

                # Region-specific motherhood penalties.
                post = 0.0
                if g > 0 and rel_year >= 0:
                    if region == "DACH":
                        post = -0.22 - 0.06 * min(rel_year, 5)
                    else:
                        post = -0.10 - 0.03 * min(rel_year, 5)

                year_fe = 0.010 * (year - years.min()) + 0.005 * np.sin(
                    (year - years.min()) / 2
                )
                eps = rng.normal(0, 0.11)

                log_earnings = (
                    10
                    + country_fe[country]
                    + ability
                    + trend_i * (year - years.min())
                    + year_fe
                    + pretrend
                    + post
                    + eps
                )

                records.append(
                    {
                        "unit": unit,
                        "country": country,
                        "region": region,
                        "year": year,
                        "g": g,
                        "treat": treat,
                        "rel_year": rel_year,
                        "log_earnings": log_earnings,
                    }
                )

    return pd.DataFrame(records)


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
