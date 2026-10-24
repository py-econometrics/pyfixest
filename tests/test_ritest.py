from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.post_estimation.ritest import _assignment_units

matplotlib.use("Agg")  # Use a non-interactive backend


@pytest.mark.parametrize("algorithm", ["auto", "fast", "slow"])
@pytest.mark.parametrize(
    "fml",
    [
        "y ~ D*x",
        "y ~ D + {D*x}",
        "y ~ D + I(D**2) + x",
        "y ~ D*x | g",
        "I(y-D) ~ D + x",
    ],
)
def test_ritest_rebuilds_treatment_dependencies(fml, algorithm):
    """Every draw must evaluate treatment-dependent terms on its new assignment."""
    rng = np.random.default_rng(4321)
    n = 240
    data = pd.DataFrame(
        {
            "D": rng.normal(size=n),
            "x": rng.normal(size=n),
            "g": np.repeat(np.arange(12), 20),
        }
    )
    if "D**2" not in fml:
        data["D"] = (data.D > 0).astype(float)
    data["y"] = 0.4 * data.D + data.D * data.x + rng.normal(size=n)
    fit = pf.feols(fml, data=data)
    original = fit._data.copy(deep=True)
    reps = 7
    expected = []
    rng = np.random.default_rng(31)
    # Direct formula fits rebuild every column without ritest's algorithm selection.
    for _ in range(reps):
        draw = data.assign(D=rng.permutation(data.D.to_numpy()))
        expected.append(pf.feols(fml, data=draw).coef().loc["D"])

    fit.ritest(
        "D",
        reps=reps,
        choose_algorithm=algorithm,
        rng=np.random.default_rng(31),
        store_ritest_statistics=True,
    )
    np.testing.assert_allclose(
        fit.ritest_statistics.statistics,
        expected,
        rtol=0,
        atol=1e-9,
        err_msg="RI coefficients differ from direct feols formula refits",
    )
    pd.testing.assert_frame_equal(fit._data, original)


def test_fast_ritest_requires_numba(monkeypatch):
    """Fast ritest must surface a clear ImportError when numba is missing."""
    import pyfixest.estimation.post_estimation.ritest as ritest

    monkeypatch.setattr(ritest, "nb", None)

    fit = pf.feols("Y ~ X1", data=pf.get_data(N=50))

    with pytest.raises(ImportError, match=r"pyfixest\[numba\]"):
        fit.ritest("X1", reps=10, choose_algorithm="fast")


def test_assignment_units():
    """Keep each unit's treatment and map observations to their assignment unit."""
    D = np.repeat([1.0, 0.0], [10, 90])
    cluster = np.repeat(np.arange(20), 5)
    D_cluster = np.repeat(np.repeat([1.0, 0.0], [3, 17]), 5)

    assignment, unit_codes = _assignment_units(resampvar_arr=D)
    np.testing.assert_array_equal(assignment, D)
    np.testing.assert_array_equal(unit_codes, np.arange(len(D)))

    assignment, unit_codes = _assignment_units(
        resampvar_arr=D_cluster, clustervar_arr=cluster
    )
    np.testing.assert_array_equal(assignment, np.repeat([1.0, 0.0], [3, 17]))
    np.testing.assert_array_equal(unit_codes, cluster)
    np.testing.assert_array_equal(assignment[unit_codes], D_cluster)


@pytest.mark.extended
@pytest.mark.parametrize(
    "fml", ["Y~X1+D_cluster", "Y~X1+D_cluster|f1", "Y~X1+D_cluster|f1+f2"]
)
@pytest.mark.parametrize(
    "resampvar, cluster",
    [("X1", None), ("D_cluster", None), ("D_cluster", "group_id")],
)
@pytest.mark.parametrize("reps", [111, 212])
def test_algos_internally(data, fml, resampvar, reps, cluster):
    fit = pf.feols(fml, data=data)

    rng1 = np.random.default_rng(1234)
    rng2 = np.random.default_rng(1234)

    kwargs = {
        "resampvar": resampvar,
        "reps": reps,
        "type": "randomization-c",
        "store_ritest_statistics": True,
        "cluster": cluster,
    }

    kwargs1 = kwargs.copy()
    kwargs2 = kwargs.copy()

    kwargs1["choose_algorithm"] = "slow"
    kwargs1["rng"] = rng1
    kwargs2["choose_algorithm"] = "fast"
    kwargs2["rng"] = rng2

    res1 = fit.ritest(**kwargs1)
    ritest_stats1 = fit.ritest_statistics.statistics.copy()

    res2 = fit.ritest(**kwargs2)
    ritest_stats2 = fit.ritest_statistics.statistics.copy()

    assert np.allclose(res1.Estimate, res2.Estimate, atol=1e-8, rtol=1e-8)
    assert np.allclose(res1["Pr(>|t|)"], res2["Pr(>|t|)"], atol=1e-8, rtol=1e-8)
    assert np.allclose(ritest_stats1, ritest_stats2, atol=1e-8, rtol=1e-8)


@pytest.mark.extended
@pytest.mark.parametrize("fml", ["Y~X1+f3", "Y~X1+f3|f1"])
@pytest.mark.parametrize("resampvar", ["X1"])
@pytest.mark.parametrize("cluster", [None, "group_id"])
def test_randomization_t_vs_c(fml, resampvar, cluster):
    data = pf.get_data(N=300)
    if cluster is not None:
        # cluster random assignment needs a cluster-level treatment
        data["X1"] = data.groupby("group_id").X1.transform("first")

    fit1 = pf.feols(fml, data=data)
    fit2 = pf.feols(fml, data=data)

    rng1 = np.random.default_rng(12354)
    rng2 = np.random.default_rng(12354)

    fit1.ritest(
        resampvar="X1",
        type="randomization-c",
        rng=rng1,
        cluster=cluster,
        store_ritest_statistics=True,
        reps=100,
    )
    fit2.ritest(
        resampvar="X1",
        type="randomization-t",
        rng=rng2,
        cluster=cluster,
        store_ritest_statistics=True,
        reps=100,
    )

    # just weak test that both are somewhat close
    assert (
        np.abs(fit1.ritest_statistics.pvalue - fit2.ritest_statistics.pvalue) < 0.03
        if cluster is None
        else 0.06
    ), (
        f"P-values are too different for randomization-c and randomization-t tests for {fml} and {resampvar} and {cluster}."
    )


@pytest.fixture
def ritest_results():
    # Load the CSV file into a pandas DataFrame
    file_path = "tests/data/ritest_results.csv"
    results_df = pd.read_csv(file_path)
    results_df.set_index(["formula", "resampvar", "cluster"], inplace=True)
    return results_df


@pytest.fixture
def data():
    data = pf.get_data(N=1000, seed=2999)
    # unbalanced treatments, as in tests/r_test_comparisons.R
    data["D_unit"] = (np.arange(len(data)) % 5 == 0).astype(float)
    data["D_cluster"] = (data.group_id % 4 == 0).astype(float)
    return data


_FIXEF = ["", "|f1", "|f1+f2"]
_R_CASES = (
    [
        (f"Y~X1+f3{fe}", resampvar, None)
        for resampvar in ["X1", "f3", "X1=-0.75", "f3>0.05"]
        for fe in _FIXEF
    ]
    + [
        (f"Y~D_unit+f3{fe}", resampvar, None)
        for resampvar in ["D_unit", "D_unit=0.2"]
        for fe in _FIXEF
    ]
    + [
        (f"Y~D_cluster+f3{fe}", resampvar, "group_id")
        for resampvar in ["D_cluster", "D_cluster=0.2"]
        for fe in _FIXEF
    ]
)


@pytest.mark.extended
@pytest.mark.parametrize("fml, resampvar, cluster", _R_CASES)
def test_vs_r(data, fml, resampvar, cluster, ritest_results):
    fit = pf.feols(fml, data=data)
    reps = 10_000

    rng1 = np.random.default_rng(1234)

    kwargs = {
        "resampvar": resampvar,
        "reps": reps,
        "type": "randomization-c",
        "cluster": cluster,
    }

    kwargs1 = kwargs.copy()

    kwargs1["choose_algorithm"] = "fast"
    kwargs1["rng"] = rng1

    res1 = fit.ritest(**kwargs1)

    expected = ritest_results.loc[(fml, resampvar, cluster or "none")]
    pval = expected["pval"]
    # four Monte Carlo standard errors of the difference of two estimates,
    # each from 10,000 draws
    atol = 4 * np.sqrt(pval * (1 - pval) * 2 / reps) + 1e-3

    assert np.isclose(res1["Pr(>|t|)"], pval, rtol=0, atol=atol)
    assert np.isclose(res1["Std. Error (Pr(>|t|))"], expected["se"], rtol=0, atol=atol)
    assert np.isclose(res1["2.5% (Pr(>|t|))"], expected["ci_lower"], rtol=0, atol=atol)


@pytest.mark.extended
def test_fepois_ritest():
    data = pf.get_data(model="Fepois")
    fit = pf.fepois("Y ~ X1*f3", data=data)
    fit.ritest(resampvar="f3", reps=2000, store_ritest_statistics=True)

    assert fit.ritest_statistics.statistics is not None
    assert np.allclose(
        fit.pvalue().xs("f3"), fit.ritest_statistics.pvalue, rtol=0.01, atol=0.01
    )


@pytest.fixture
def data_r_vs_t():
    return pf.get_data(N=5000, seed=2999)


@pytest.mark.parametrize("statistic", ["randomization-c", "randomization-t"])
@pytest.mark.parametrize("estimator", [pf.feols, pf.fepois])
def test_ritest_preserves_treatment_names(estimator, statistic, monkeypatch):
    """No-intercept RI retains controls whose names contain the treatment name."""
    data = pf.get_data(N=300, model="Fepois").dropna()
    data["D"] = (data.X1 > 0).astype(float)
    data["D_control"] = data.X2
    # Neither a pre-existing resampled column nor a substring in a control
    # should be overwritten or substituted in the formula.
    data["D_resampled"] = data.X1
    fml = "Y ~ D + D_control + {D * D_control} - 1"
    fit = estimator(fml, data=data)
    original = fit._data.copy(deep=True)
    reps = 5
    rng = np.random.default_rng(31)
    expected = []
    for _ in range(reps):
        resampled = original.copy(deep=False)
        resampled["D"] = rng.permutation(original.D.to_numpy())
        reference = estimator(
            fml,
            data=resampled,
            vcov="iid" if statistic == "randomization-c" else "hetero",
        )
        values = (
            reference.coef() if statistic == "randomization-c" else reference.tstat()
        )
        expected.append(values["D"])

    monkeypatch.setattr(
        Formula, "render", lambda self: "descriptive text, not a formula"
    )
    assert fit.model.formula == fit._fml == "descriptive text, not a formula"
    monkeypatch.setattr(Formula, "parse", _fail_on_reparse)
    fit.ritest(
        "D",
        reps=reps,
        type=statistic,
        choose_algorithm="slow",
        rng=np.random.default_rng(31),
        store_ritest_statistics=True,
    )
    np.testing.assert_allclose(fit.ritest_statistics.statistics, expected, rtol=1e-10)
    pd.testing.assert_frame_equal(fit._data, original)


def _fail_on_reparse(*args, **kwargs):
    raise AssertionError("Randomization inference must reuse the parsed formula.")


@pytest.mark.parametrize(
    "initial_vcov", ["iid", {"CRV1": "g1"}, {"CRV1": "g2"}, {"CRV3": "g2"}]
)
@pytest.mark.parametrize("statistic", ["randomization-c", "randomization-t"])
def test_ritest_cluster_override_preserves_fit(initial_vcov, statistic):
    """Direct CRV1 fits bypass the override and leave its numerical contract explicit."""
    rng = np.random.default_rng(9)
    assignments = rng.integers(0, 2, 12).astype(float)
    data = pd.DataFrame(
        {
            "D": np.repeat(assignments, 10),
            "x": rng.normal(size=120),
            "g1": np.repeat(np.arange(12), 10),
            "g2": np.tile(np.arange(10), 12),
        }
    )
    data["y"] = 0.4 * data.D + data.x + rng.normal(size=12)[data.g1]
    data["y"] += rng.normal(size=len(data))
    correction = pf.ssc(k_adj=False, G_adj=False)
    fit = pf.feols("y ~ D + x", data=data, vcov=initial_vcov, ssc=correction)
    covariance = fit.variance_covariance
    original = fit.tidy().copy(deep=True)

    def reference(draw_data):
        draw_fit = pf.feols(
            "y ~ D + x", data=draw_data, vcov={"CRV1": "g1"}, ssc=correction
        )
        values = draw_fit.tstat() if statistic == "randomization-t" else draw_fit.coef()
        return values.loc["D"]

    draw_rng = np.random.default_rng(1)
    expected = [
        reference(data.assign(D=np.repeat(draw_rng.permutation(assignments), 10)))
        for _ in range(5)
    ]
    fit.ritest(
        "D",
        cluster="g1",
        type=statistic,
        choose_algorithm="slow",
        reps=5,
        rng=np.random.default_rng(1),
        store_ritest_statistics=True,
    )
    assert fit.variance_covariance is covariance
    pd.testing.assert_frame_equal(fit.tidy(), original)
    # The same estimator and rows differ only by cluster-score accumulation rounding.
    np.testing.assert_allclose(
        fit.ritest_statistics.statistics,
        expected,
        rtol=0,
        atol=1e-10,
        err_msg="Cluster override permutation statistics differ from direct CRV1 fits",
    )
    np.testing.assert_allclose(
        fit.ritest_statistics.sample_stat,
        reference(data),
        rtol=0,
        atol=1e-10,
        err_msg="Cluster override observed statistic differs from the direct CRV1 fit",
    )
