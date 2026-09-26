import contextlib
import os

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro

# rpy2 imports
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.estimation import fepois

fixest = importr("fixest")
stats = importr("stats")
sandwich = importr("sandwich")


def test_separation():
    """Test separation detection."""
    example1 = pd.DataFrame.from_dict(
        {
            "Y": [0, 0, 0, 1, 2, 3],
            "fe1": ["a", "a", "b", "b", "b", "c"],
            "fe2": ["c", "c", "d", "d", "d", "e"],
            "X": np.random.normal(0, 1, 6),
        }
    )
    with pytest.warns(
        UserWarning, match="2 observations removed because of separation."
    ):
        fepois("Y ~ X  | fe1", data=example1, vcov="hetero", separation_check=["fe"])

    if False:
        # this example is taken from ppmlhdfe's primer on separation https://github.com/sergiocorreia/ppmlhdfe/blob/master/guides/separation_primer.md
        # disabled because we currently do not perform separation checks if no fixed effects are provided
        # TODO: enable once separation checks without fixed effects are enabled
        example2 = pd.DataFrame.from_dict(
            {
                "Y": [0, 0, 0, 1, 2, 3],
                "X1": [2, -1, 0, 0, 5, 6],
                "X2": [5, 10, 0, 0, -10, -12],
            }
        )

        with pytest.warns(
            UserWarning, match="2 observations removed because of separation."
        ):
            fepois("Y ~ X1 + X2", data=example2, vcov="hetero", separation_check=["ir"])

    # ppmlhdfe test data sets (check readme in data/ppmlhdfe_separation_examples)
    path = os.path.dirname(os.path.abspath(__file__))
    folder = r"data/ppmlhdfe_separation_examples"
    fns = sorted(
        [fn for fn in os.listdir(os.path.join(path, folder)) if fn.endswith(".csv")]
    )
    for fn in fns:
        if fn in ["07.csv"]:
            # this case fails but is not tested in ppmlhdfe
            # https://github.com/sergiocorreia/ppmlhdfe/blob/master/test/validate_tagsep.do#L27
            continue
        data = pd.read_csv(os.path.join(path, folder, fn))
        # build formula dynamically from dataframe
        # datasets have fixed structure of the form (y, x1, ..., xN, id1, ..., idM, separated)
        fml = "y"  # dependent variable y
        regressors = data.columns[
            data.columns.str.startswith("x")
        ]  # regressors x1,...,xN
        fixed_effects = data.columns[
            data.columns.str.startswith("id")
        ]  # fixed effects id1,...,id2

        if regressors.empty:
            # TODO: formulae with just a constant term and fixed effects throw error in FIT.get_fit(), e.g., for 03.csv and Y ~ 1 | id1 + id2 + id3?
            continue
        fml += f" ~ {' + '.join(regressors)}"

        if fixed_effects.empty:
            # TODO: separation checks are currently disabled if no fixed effects are specified; enable tests once we run separation check without fixed effects
            continue
        else:
            fml += f" | {' + '.join(fixed_effects)}"

        with (
            pytest.warns(
                UserWarning,
                match=f"{data.separated.sum()} observations removed because of separation.",
            ) as record,
            contextlib.suppress(Exception),
        ):
            pf.fepois(fml, data=data, separation_check=["ir"])

        # if no separation, no warning is raised
        if data.separated.sum() == 0:
            assert len(record) == 0


@pytest.mark.against_r_core
@pytest.mark.parametrize("fml", ["Y ~ X1", "Y ~ X1 | f1"])
def test_against_fixest(fml):
    data = pf.get_data(model="Fepois")
    iwls_tol = 1e-12

    # vcov = "hetero"
    vcov = "hetero"
    fit = pf.fepois(fml, data=data, vcov=vcov, iwls_tol=iwls_tol)
    fit_r = fixest.fepois(ro.Formula(fml), data=data, vcov=vcov, glm_tol=iwls_tol)

    np.testing.assert_allclose(
        fit_r.rx2("irls_weights"),
        fit.working_state.working_weights,
        atol=1e-08,
        rtol=1e-07,
    )
    np.testing.assert_allclose(
        fit_r.rx2("linear.predictors").reshape(-1, 1),
        fit.working_state.eta.reshape(-1, 1),
        atol=1e-08,
        rtol=1e-07,
    )
    np.testing.assert_allclose(
        fit_r.rx2("scores").reshape(-1, 1),
        fit.sandwich.scores.reshape(-1, 1),
        atol=1e-08,
        rtol=1e-07,
    )

    np.testing.assert_allclose(
        fit_r.rx2("hessian"), fit.sandwich.hessian, atol=1e-08, rtol=1e-07
    )

    np.testing.assert_allclose(
        fit_r.rx2("deviance"), fit.fitstat.deviance, atol=1e-08, rtol=1e-07
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    ("fml", "r_fml"),
    [
        ("Y ~ X1 + X2", "Y ~ X1 + X2 + offset(off)"),
        ("Y ~ X1 + X2 | f3", "Y ~ X1 + X2 + factor(f3) + offset(off)"),
    ],
)
def test_crv3_offset_vs_sandwich_vcovjk(fml, r_fml):
    "The CRV3 leave-one-cluster-out refits must keep the offset (and IRLS tolerance)."
    data = pf.get_data(model="Fepois").dropna()[["Y", "X1", "X2", "f1", "f3"]]
    data = data.reset_index(drop=True)
    rng = np.random.default_rng(8123)
    data["off"] = np.log(rng.uniform(0.5, 2.0, size=len(data)))
    iwls_tol = 1e-12
    n_clusters = data["f1"].nunique()

    # without small-sample corrections, CRV3 is the unscaled sum of outer
    # products of the leave-one-cluster-out deviations from the full estimate
    fit = pf.fepois(
        fml,
        data=data,
        offset="off",
        vcov={"CRV3": "f1"},
        ssc=pf.ssc(k_adj=False, G_adj=False),
        iwls_tol=iwls_tol,
        # glm() keeps singleton fixed-effect levels
        fixef_rm="none",
    )
    fit_r = stats.glm(
        ro.Formula(r_fml),
        family=stats.poisson(),
        data=data,
        control=stats.glm_control(epsilon=iwls_tol, maxit=100),
    )
    with ro.default_converter.context():
        coef_r = fit_r.rx2("coefficients")
        coef_r = pd.Series(np.asarray(coef_r), index=list(coef_r.names))
        coef_r = coef_r.rename({"(Intercept)": "Intercept"})
        # vcovJK scales the same sum of outer products by (G - 1) / G
        vcov_r = sandwich.vcovJK(fit_r, cluster=ro.Formula("~f1"), center="estimate")
        vcov_r = pd.DataFrame(
            np.asarray(vcov_r) * n_clusters / (n_clusters - 1),
            index=list(ro.r["rownames"](vcov_r)),
            columns=list(ro.r["colnames"](vcov_r)),
        ).rename(
            index={"(Intercept)": "Intercept"}, columns={"(Intercept)": "Intercept"}
        )

    coefnames = list(fit.coef().index)
    vcov_py = pd.DataFrame(
        fit.variance_covariance.vcov, index=coefnames, columns=coefnames
    )
    np.testing.assert_allclose(
        fit.coef().to_numpy(),
        coef_r[coefnames].to_numpy(),
        rtol=1e-8,
        atol=0,
        err_msg="coefficients",
    )
    # each refit converges only to the IRLS tolerance, and the vcov sums G
    # differences of such refits
    np.testing.assert_allclose(
        vcov_py.to_numpy(),
        vcov_r.loc[coefnames, coefnames].to_numpy(),
        rtol=1e-6,
        atol=1e-12,
        err_msg="CRV3 vcov",
    )
