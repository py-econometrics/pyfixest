"""Tasks for the fresh-agent eval.

Each task is a function whose docstring is the prompt. It takes a seeded
generator and returns the data and a reference function that fits the intended
model with pyfixest. Run this file with the interpreter under test: without
arguments it lists the task ids; with them it writes one task's data and prints
the task as JSON.

    <python> -I scripts/agent_eval/tasks.py [<task id> <output dir>]
"""

import json
import sys
import sysconfig
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import pyfixest as pf

TASKS = {}


def task(fn):
    TASKS[fn.__name__] = fn
    return fn


def panel(r, n_units, years, unit="firm"):
    """Balanced panel with unit and year effects."""
    df = pd.DataFrame(
        [(i, t) for i in range(n_units) for t in years], columns=[unit, "year"]
    )
    unit_effect = r.normal(size=n_units)[df[unit]]
    return df, unit_effect, r.normal(size=len(years))[df.year - years[0]]


def coefs(fit, **keys):
    return {
        key: fit.coef().iloc[i] if isinstance(i, int) else fit.coef()[i]
        for key, i in keys.items()
    }


# Silent traps: a plausible spelling runs but fits another model or warns.


@task
def quadratic(r):
    """Regress y on x and the square of x, with state fixed effects. Report the
    coefficients on x and on x squared.
    """
    n, s = 1500, r.integers(0, 30, 1500)
    x = r.normal(1, 1, n)
    y = 0.8 * x - 0.3 * x**2 + r.normal(size=30)[s] + r.normal(size=n)
    df = pd.DataFrame(dict(state=s, x=x, y=y))
    return df, lambda df: coefs(
        pf.feols("y ~ x + I(x**2) | state", df), b_x=0, b_x_squared=1
    )


@task
def quadratic_integer(r):
    """Estimate a Mincer wage equation: regress log_wage on schooling, age and age
    squared, with region fixed effects. Report the three coefficients.
    """
    n, region = 2000, r.integers(1, 9, 2000)
    age, school = r.integers(18, 66, n), r.integers(8, 21, n)
    lw = 1 + 0.08 * school + 0.05 * age - 0.0005 * age**2 + r.normal(0, 0.2, 9)[region]
    lw += r.normal(0, 0.4, n)
    df = pd.DataFrame(dict(region=region, age=age, schooling=school, log_wage=lw))
    fml = "log_wage ~ schooling + age + I(age**2) | region"
    return df, lambda df: coefs(
        pf.feols(fml, df), b_schooling=0, b_age=1, b_age_squared=2
    )


@task
def region_by_year(r):
    """Regress sales on price and a full set of region-by-year indicators, one for
    every combination of region and year, both treated as categories. Report the
    price coefficient and the estimated difference in expected sales between
    region 3 and region 1 in 2019 at the same price.
    """
    n, region, year = 2400, r.integers(1, 5, 2400), r.integers(2015, 2021, 2400)
    cell = 2 * r.normal(size=(4, 6))[region - 1, year - 2015]
    price = 10 + 0.5 * cell + r.normal(0, 2, n)
    sales = 50 - 1.5 * price + cell + r.normal(size=n)

    def reference(df):
        b = pf.feols("sales ~ price | region:year", df).coef()["price"]
        # With a dummy per cell, each cell's intercept is its mean of sales - b * price.
        cell_mean = (df.sales - b * df.price).groupby([df.region, df.year]).mean()
        return {
            "b_price": b,
            "gap_region3_vs_1_2019": cell_mean[3, 2019] - cell_mean[1, 2019],
        }

    return pd.DataFrame(
        {"region": region, "year": year, "price": price, "sales": sales}
    ), reference


@task
def slope_by_level(r):
    """In one pooled regression, regress output on hours, letting both the
    intercept and the slope on hours differ by sector. Report the slope of output
    with respect to hours in the retail sector.
    """
    n, sec = 1500, r.choice(["manufacturing", "retail", "services"], 1500)
    hours = r.normal(40, 8, n)
    slope = pd.Series(sec).map({"manufacturing": 1.2, "retail": 0.4, "services": 0.8})
    out = 5 * (sec == "retail") + slope.to_numpy() * hours + r.normal(0, 4, n)
    df = pd.DataFrame({"sector": sec, "hours": hours, "output": out})
    # A fully interacted pooled fit has the same slope as the retail subsample.
    return df, lambda df: coefs(
        pf.feols("output ~ hours", df[df.sector == "retail"]), slope_retail="hours"
    )


@task
def cluster_with_fe(r):
    """Regress y on x with firm and year fixed effects. Report the coefficient on x
    and its standard error clustered by firm.
    """
    df, a, g = panel(r, 100, range(2010, 2020))
    t = df.year - 2014.5
    df["x"] = a + r.normal(size=100)[df.firm] * t + r.normal(size=len(df))
    df["y"] = (
        0.5 * df.x + a + g + r.normal(size=100)[df.firm] * t + r.normal(size=len(df))
    )

    def reference(df):
        fit = pf.feols("y ~ x | firm + year", df, vcov={"CRV1": "firm"})
        return {"b_x": fit.coef()["x"], "se_x": fit.se()["x"]}

    return df, reference


@task
def iv_with_fe(r):
    """Estimate the effect of price on quantity by two-stage least squares, using
    cost as the instrument for price, controlling for income and market fixed
    effects. Report the coefficient on price.
    """
    n, m = 3000, r.integers(0, 50, 3000)
    cost, income, u = r.normal(size=n), r.normal(size=n), r.normal(size=n)
    market_effect = r.normal(size=50)[m]
    price = 1 + 0.8 * cost + 0.3 * income + market_effect + 0.5 * u + r.normal(size=n)
    q = 10 - 1.2 * price + 0.5 * income + r.normal(size=50)[m] + u
    df = pd.DataFrame(
        {"market": m, "cost": cost, "income": income, "price": price, "quantity": q}
    )
    return df, lambda df: coefs(
        pf.feols("quantity ~ income + [price ~ cost] | market", df), b_price="price"
    )


@task
def interacted_fe(r):
    """Regress log_wage on tenure with a fixed effect for every firm-by-year
    combination. Report the tenure coefficient.
    """
    n, firm, year = 3000, r.integers(0, 40, 3000), r.integers(2015, 2020, 3000)
    fy = r.normal(size=(40, 5))[firm, year - 2015]
    tenure = np.round(5 + 3 * fy + r.exponential(4, n), 1)
    lw = 2 + 0.03 * tenure + 0.5 * fy + r.normal(0, 0.3, n)
    df = pd.DataFrame({"firm": firm, "year": year, "tenure": tenure, "log_wage": lw})
    return df, lambda df: coefs(
        pf.feols("log_wage ~ tenure | firm:year", df), b_tenure="tenure"
    )


@task
def two_outcomes(r):
    """With a single pyfixest estimation call that fits both models, regress
    exports and, separately, imports on tariff and gdp with country fixed effects.
    Report the tariff coefficient from each model.
    """
    df, a, g = panel(r, 60, range(2005, 2020), unit="country")
    df["tariff"] = 5 + 2 * a + r.normal(size=len(df))
    df["gdp"] = 10 + a + r.normal(size=len(df))
    df["exports"] = 3 - 0.2 * df.tariff + 0.5 * df.gdp + a + r.normal(size=len(df))
    df["imports"] = 2 - 0.4 * df.tariff + 0.7 * df.gdp + g + r.normal(size=len(df))
    fit = lambda df, y: pf.feols(f"{y} ~ tariff + gdp | country", df)  # noqa: E731
    return df, lambda df: {
        f"b_tariff_{y}": fit(df, y).coef()["tariff"] for y in ("exports", "imports")
    }


@task
def n_and_r2(r):
    """Regress output on capital and labor with plant fixed effects. Report the
    number of observations used in the estimation and the R squared of the model
    (the ordinary R squared, not the within R squared).
    """
    plant = np.concatenate(
        [np.repeat(np.arange(120), r.integers(2, 12, 120)), np.arange(120, 150)]
    )
    n = len(plant)
    capital, labor = r.normal(5, 1, n), r.normal(3, 1, n)
    out = 0.3 * capital + 0.6 * labor + r.normal(size=150)[plant] + r.normal(0, 0.5, n)
    capital[r.random(n) < 0.05] = np.nan  # missing values and singleton plants shrink N
    df = pd.DataFrame(
        {"plant": plant, "capital": capital, "labor": labor, "output": out}
    )
    fit = lambda df: pf.feols("output ~ capital + labor | plant", df)  # noqa: E731
    return df, lambda df: {"n_obs": fit(df).sample_info.n_obs, "r2": fit(df).fitstat.r2}


# Errors whose message may not lead to the fix.


@task
def factor_covariate(r):
    """Regress wage on experience (exper) and a set of indicators for the education
    category (educ is a category code from 1 to 4), with region fixed effects.
    Report the coefficient on exper and the coefficient for education category 4
    relative to category 1.
    """
    n, region, educ = 2000, r.integers(0, 10, 2000), r.integers(1, 5, 2000)
    exper = 20 - 2 * educ + r.normal(0, 6, n)
    educ_effect = np.array([0, 0.2, 0.5, 1.4])[educ - 1]
    wage = 2 + 0.03 * exper + educ_effect + r.normal(0, 0.3, 10)[region]
    wage = wage + r.normal(0, 0.3, n)
    df = pd.DataFrame(dict(region=region, educ=educ, exper=exper, wage=wage))
    fit = lambda df: pf.feols("wage ~ exper + C(educ) | region", df)  # noqa: E731
    return df, lambda df: coefs(fit(df), b_exper="exper", b_educ_4="C(educ)[T.4]")


@task
def lagged_regressor(r):
    """Regress investment on cash_flow and cash_flow lagged by one year within
    firm, with firm and year fixed effects. Report both coefficients.
    """
    df, a, g = panel(r, 60, range(2005, 2020))
    df["cash_flow"] = a + r.normal(size=len(df))
    prev = df.groupby("firm").cash_flow.shift(1).fillna(0)
    df["investment"] = 0.3 * df.cash_flow + 0.2 * prev + a + g + r.normal(size=len(df))

    def reference(df):
        df = df.sort_values(["firm", "year"]).assign(
            lag=lambda d: d.groupby("firm").cash_flow.shift(1)
        )
        fit = pf.feols("investment ~ cash_flow + lag | firm + year", df)
        return coefs(fit, b_cash_flow="cash_flow", b_cash_flow_lag="lag")

    return df, reference


@task
def messy_columns(r):
    """Regress log_sales on rd_intensity in this firm panel, with firm and
    fiscal-year fixed effects. Report the coefficient on rd_intensity and its
    standard error clustered by firm.
    """
    df, a, g = panel(r, 80, range(2010, 2020))
    df["rd_intensity"] = 0.1 + 0.05 * a + 0.02 * r.normal(size=len(df))
    log_sales = 3 + 4 * df.rd_intensity + a + g + r.normal(size=80)[df.firm]
    df["log_sales"] = log_sales + r.normal(size=len(df))
    df["firm"] = [f"F-{17 * f + 3:04d}" for f in df.firm]
    df = df.rename(columns={"firm": "firm_id", "year": "fiscal.year"})

    def reference(df):
        df = df.rename(columns={"fiscal.year": "fiscal_year"}).astype(
            {"firm_id": "category"}
        )
        fml = "log_sales ~ rd_intensity | firm_id + fiscal_year"
        fit = pf.feols(fml, df, vcov={"CRV1": "firm_id"})
        return {
            "b_rd_intensity": fit.coef()["rd_intensity"],
            "se_rd_intensity": fit.se()["rd_intensity"],
        }

    return df, reference


@task
def first_stage_f(r):
    """Estimate the effect of x on y by two-stage least squares, instrumenting x
    with z and controlling for w (no fixed effects, classical iid standard
    errors). Report the coefficient on x and the first-stage F statistic of the
    excluded instrument.
    """
    n = 1000
    z, w, u = r.normal(size=n), r.normal(size=n), r.normal(size=n)
    x = 0.15 * z + 0.5 * w + u + r.normal(size=n)
    df = pd.DataFrame(
        {"z": z, "w": w, "x": x, "y": 1 + 0.7 * x + 0.5 * w + u + r.normal(size=n)}
    )

    def reference(df):
        # One instrument and iid errors: F is the squared t statistic of z in the first stage.
        fit = pf.feols("y ~ w + [x ~ z]", df)
        return {
            "b_x": fit.coef()["x"],
            "first_stage_f": fit.first_stage.diagnostics.f_stat,
        }

    return df, reference


@task
def firm_trends(r):
    """Regress y on treated with firm fixed effects, year fixed effects and
    firm-specific linear time trends. Report the coefficient on treated.
    """
    df, a, g = panel(r, 60, range(2005, 2017))
    slope = r.normal(0, 0.3, 60)
    adopt = np.where(slope > 0.2, 2008, np.where(slope > -0.1, 2012, 2030))
    df["treated"] = (df.year >= adopt[df.firm]).astype(int)
    trend = slope[df.firm] * (df.year - 2005)
    df["y"] = df.treated + a + g + trend + r.normal(0, 0.5, len(df))
    return df, lambda df: coefs(
        pf.feols("y ~ treated | firm[year] + year", df), b_treated="treated"
    )


@task
def poisson_offset(r):
    """Fit a Poisson regression of accidents on speed_limit with city fixed
    effects, treating population as the exposure (log population enters as an
    offset with its coefficient fixed at one). Report the speed_limit
    coefficient.
    """
    df, a, _ = panel(r, 40, range(2010, 2020), unit="city")
    growth, t = r.normal(0, 0.1, 40)[df.city], df.year - 2010
    df["population"] = np.round(np.exp(r.normal(11, 1, 40)[df.city] + growth * t))
    df["speed_limit"] = r.choice([30, 40, 50], len(df)) + 10 * (growth > 0) * (t > 4)
    df["accidents"] = r.poisson(np.exp(-9 + 0.02 * df.speed_limit + a) * df.population)

    def reference(df):
        df = df.assign(log_pop=np.log(df.population))
        fit = pf.fepois("accidents ~ speed_limit | city", df, offset="log_pop")
        return coefs(fit, b_speed_limit="speed_limit")

    return df, reference


def main(task_id=None, out_dir=None):
    if task_id is None:
        return print(json.dumps(list(TASKS)))
    fn = TASKS[task_id]
    data, reference = fn(np.random.default_rng(zlib.crc32(task_id.encode())))
    path = Path(out_dir) / "data.csv"
    data.to_csv(path, index=False)
    ref = {k: float(v) for k, v in reference(pd.read_csv(path)).items()}
    site = sysconfig.get_paths()["purelib"]
    prompt = " ".join(fn.__doc__.split())
    task = dict(
        id=task_id, prompt=prompt, reference=ref, pyfixest=pf.__file__, site=site
    )
    print(json.dumps(task))


if __name__ == "__main__":
    main(*sys.argv[1:])
