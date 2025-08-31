import numpy as np
import pandas as pd
import pytest
from formulaic.errors import FactorEvaluationError

import pyfixest as pf
from pyfixest.errors import (
    DuplicateKeyError,
    EndogVarsAsCovarsError,
    InstrumentsAsCovarsError,
    NanInClusterVarError,
    UnderDeterminedIVError,
    VcovTypeNotSupportedError,
)
from pyfixest.estimation.estimation import feols, fepois
from pyfixest.estimation.FormulaParser import FixestFormulaParser
from pyfixest.estimation.multcomp import rwolf
from pyfixest.report.summarize import etable, summary
from pyfixest.utils.dgps import gelbach_data
from pyfixest.utils.utils import get_data, ssc


@pytest.fixture
def data():
    return pf.get_data()


def test_formula_parser2():
    with pytest.raises(DuplicateKeyError):
        FixestFormulaParser("y ~ sw(a, b) +  sw(c, d)| sw(X3, X4))")


def test_formula_parser3():
    with pytest.raises(DuplicateKeyError):
        FixestFormulaParser("y ~ sw(a, b) +  csw(c, d)| sw(X3, X4))")


# def test_formula_parser2():
#    with pytest.raises(FixedEffectInteractionError):
#        FixestFormulaParser('y ~ X1 + X2 | X3:X4')

# def test_formula_parser3():
#    with pytest.raises(CovariateInteractionError):
#        FixestFormulaParser('y ~ X1 + X2^X3')


def test_cluster_na():
    """Test if a nan value in a cluster variable raises an error."""
    data = get_data()
    data = data.dropna()
    data["f3"] = data["f3"].astype("int64")
    data["f3"][5] = np.nan

    with pytest.raises(NanInClusterVarError):
        feols(fml="Y ~ X1", data=data, vcov={"CRV1": "f3"})


def test_cluster_but_no_data():
    """Test if AttributeError if self._data is not stored."""
    data = get_data()
    fit = feols("Y ~ X1", data=data, store_data=False)
    with pytest.raises(AttributeError):
        fit.vcov({"CRV1": "f2"})


def test_error_hc23_fe():
    """
    Test if HC2 & HC3 inference with fixed effects regressions raises an error.

    Notes
    -----
    Currently not supported.
    """
    data = get_data().dropna()

    with pytest.raises(VcovTypeNotSupportedError):
        feols(fml="Y ~ X1 | f2", data=data, vcov="HC2")

    with pytest.raises(VcovTypeNotSupportedError):
        feols(fml="Y ~ X1 | f2", data=data, vcov="HC3")


def test_depvar_numeric():
    """Test if feols() throws an error when the dependent variable is not numeric."""
    data = get_data()
    data["Y"] = data["Y"].astype("str")
    data["Y"] = pd.Categorical(data["Y"])

    with pytest.raises(TypeError):
        feols(fml="Y ~ X1", data=data)


def test_iv_errors():
    data = get_data()

    # under determined
    with pytest.raises(UnderDeterminedIVError):
        feols(fml="Y ~ X1 | Z1 + Z2 ~ 24 ", data=data)
    # instrument specified as covariate
    with pytest.raises(InstrumentsAsCovarsError):
        feols(fml="Y ~ X1 | Z1  ~ X1 + X2", data=data)
    # endogenous variable specified as covariate
    with pytest.raises(EndogVarsAsCovarsError):
        feols(fml="Y ~ Z1 | Z1  ~ X1", data=data)

    # instrument specified as covariate
    # with pytest.raises(InstrumentsAsCovarsError):
    #    fixest.feols('Y ~ X1 | Z1 + Z2 ~ X3 + X4')
    # underdetermined IV
    # with pytest.raises(UnderDeterminedIVError):
    #    fixest.feols('Y ~ X1 + X2 | X1 + X2 ~ X4 ')
    # with pytest.raises(UnderDeterminedIVError):
    #    fixest.feols('Y ~ X1 | Z1 + Z2 ~ X2 + X3 ')
    # CRV3 inference
    with pytest.raises(VcovTypeNotSupportedError):
        feols(fml="Y ~ 1 | Z1 ~ X1 ", vcov={"CRV3": "group_id"}, data=data)
    # wild bootstrap
    with pytest.raises(NotImplementedError):
        feols(fml="Y ~ 1 | Z1 ~ X1 ", data=data).wildboottest(param="Z1", reps=999)
    # multi estimation error
    with pytest.raises(NotImplementedError):
        feols(fml="Y + Y2 ~ 1 | Z1 ~ X1 ", data=data)
    with pytest.raises(NotImplementedError):
        feols(fml="Y  ~ 1 | sw(f2, f3) | Z1 ~ X1 ", data=data)
    with pytest.raises(NotImplementedError):
        feols(fml="Y  ~ 1 | csw(f2, f3) | Z1 ~ X1 ", data=data)
    # unsupported HC vcov
    with pytest.raises(VcovTypeNotSupportedError):
        feols(fml="Y  ~ 1 | Z1 ~ X1", vcov="HC2", data=data)
    with pytest.raises(VcovTypeNotSupportedError):
        feols(fml="Y  ~ 1 | Z1 ~ X1", vcov="HC3", data=data)


@pytest.mark.skip("Not yet implemented.")
def test_poisson_devpar_count():
    """Check that the dependent variable is a count variable."""
    data = get_data()
    # under determined
    with pytest.raises(AssertionError):
        fepois(fml="Y ~ X1 | X4", data=data)


def test_feols_errors():
    data = pf.get_data()

    with pytest.raises(TypeError):
        pf.feols(fml=1, data=data)

    with pytest.raises(TypeError):
        pf.feols(fml="Y ~ X1", data=1)

    with pytest.raises(TypeError):
        pf.feols(fml="Y ~ X1", data=data, vcov=1)

    with pytest.raises(TypeError):
        pf.feols(fml="Y ~ X1", data=data, fixef_rm=1)

    with pytest.raises(ValueError):
        pf.feols(fml="Y ~ X1", data=data, fixef_rm="f1")

    with pytest.raises(TypeError):
        pf.feols(fml="Y ~ X1", data=data, collin_tol=2)

    with pytest.raises(TypeError):
        pf.feols(fml="Y ~ X1", data=data, collin_tol="2")

    with pytest.raises(ValueError):
        pf.feols(fml="Y ~ X1", data=data, collin_tol=-1.0)

    with pytest.raises(TypeError):
        pf.feols(fml="Y ~ X1", data=data, lean=1)

    with pytest.raises(TypeError):
        pf.feols(fml="Y ~ X1", data=data, fixef_tol="a")

    with pytest.raises(ValueError):
        pf.feols(fml="Y ~ X1", data=data, fixef_tol=1.0)

    with pytest.raises(ValueError):
        pf.feols(fml="Y ~ X1", data=data, fixef_tol=0.0)

    with pytest.raises(ValueError):
        pf.feols(fml="Y ~ X1", data=data, weights_type="qweights", weights="weights")


def test_poisson_errors():
    data = pf.get_data(model="Fepois")
    # iv not supported
    with pytest.raises(NotImplementedError):
        pf.fepois("Y ~ 1 | X1 ~ Z1", data=data)


def test_all_variables_multicollinear():
    data = get_data()
    with pytest.raises(ValueError):
        fit = feols("Y ~ f1 | f1", data=data)  # noqa: F841


def test_wls_errors():
    data = get_data()

    with pytest.raises(AssertionError):
        feols(fml="Y ~ X1", data=data, weights="weights2")

    with pytest.raises(ValueError):
        feols("Y ~ X1", data=data, weights=[1, 2])

    data.loc[0, "weights"] = np.nan
    with pytest.raises(NotImplementedError):
        feols("Y ~ X1", data=data, weights="weights").wildboottest(
            cluster="f1", param="X1", reps=999, seed=12
        )

    # test for ValueError when weights are not positive
    data.loc[10, "weights"] = -1
    with pytest.raises(ValueError):
        feols("Y ~ X1", data=data, weights="weights", vcov="iid")

    # test for ValueError when weights are not numeric
    data["weights"] = data["weights"].astype("str")
    data.loc[10, "weights"] = "a"
    with pytest.raises(ValueError):
        feols("Y ~ X1", data=data, weights="weights", vcov="iid")

    data = get_data()
    with pytest.raises(NotImplementedError):
        feols("Y ~ X1", data=data, weights="weights", vcov="iid").wildboottest(reps=999)


def test_multcomp_errors():
    data = get_data().dropna()

    # param not in model
    fit1 = feols("Y + Y2 ~ X1 | f1", data=data)
    with pytest.raises(ValueError):
        rwolf(fit1.to_list(), param="X2", reps=999, seed=92)


def test_multcomp_sampling_errors():
    data = get_data().dropna()
    # Sampling method not supported in "rwolf"
    fit1 = feols("Y + Y2 ~ X1 | f1", data=data)
    with pytest.raises(ValueError):
        rwolf(fit1.to_list(), param="X1", reps=999, seed=92, sampling_method="abc")


def test_rwolf_error():
    rng = np.random.default_rng(123)

    data = get_data()
    data["f1"] = rng.choice(range(5), len(data), True)
    fit = feols("Y + Y2 ~ X1 | f1", data=data)

    # test for full enumeration warning
    with pytest.warns(UserWarning):
        pf.rwolf(fit.to_list(), "X1", reps=9999, seed=123)


def test_predict_dtype_error():
    data = get_data()
    fit = feols("Y ~ X1 | f1", data=data)

    data["f1"] = data["f1"].fillna(0).astype(int)
    with pytest.warns(UserWarning):
        fit.predict(newdata=data.iloc[0:100])


def test_wildboottest_errors():
    data = get_data()
    fit = feols("Y ~ X1", data=data)
    with pytest.raises(ValueError):
        fit.wildboottest(param="X2", reps=999, seed=213)


def test_summary_errors():
    "Test for appropriate errors when providing objects of type FixestMulti."
    data = get_data()
    fit1 = feols("Y + Y2 ~ X1 | f1", data=data)
    fit2 = feols("Y ~ X1 + X2 | f1", data=data)

    with pytest.raises(TypeError):
        etable([fit1, fit2])
    with pytest.raises(TypeError):
        summary([fit1, fit2])


def test_errors_etable():
    data = get_data()
    fit1 = feols("Y ~ X1", data=data)
    fit2 = feols("Y ~ X1 + X2 | f1", data=data)

    with pytest.raises(AssertionError):
        etable([fit1, fit2], signif_code=[0.01, 0.05])

    with pytest.raises(AssertionError):
        etable([fit1, fit2], signif_code=[0.2, 0.05, 0.1])

    with pytest.raises(AssertionError):
        etable([fit1, fit2], signif_code=[0.1, 0.5, 1.5])

    with pytest.raises(AssertionError):
        etable(
            models=[fit1, fit2],
            custom_stats={
                "conf_int_lb": [
                    fit2._conf_int[0]
                ],  # length of customized statistics not equal to the number of models
                "conf_int_ub": [fit2._conf_int[1]],
            },
            coef_fmt="b se\n[conf_int_lb, conf_int_ub]",
        )

    with pytest.raises(AssertionError):
        etable(
            models=[fit1, fit2],
            custom_stats={
                "conf_int_lb": [
                    [0.1, 0.1, 0.1],
                    fit2._conf_int[0],
                ],  # length of customized statistics not equal to length of model
                "conf_int_ub": [fit1._conf_int[1], fit2._conf_int[1]],
            },
            coef_fmt="b [conf_int_lb, conf_int_ub]",
        )

    with pytest.raises(ValueError):
        etable(
            models=[fit1, fit2],
            custom_stats={
                "b": [
                    fit2._conf_int[0],
                    fit2._conf_int[0],
                ],  # preserved keyword cannot be used as a custom statistic
            },
            coef_fmt="b [se]",
        )


def test_errors_ccv():
    data = get_data().dropna()
    data["D"] = np.random.choice([0, 1], size=len(data))

    # error when D is not binary
    fit = feols("Y ~ X1", data=data, vcov={"CRV1": "f1"})
    with pytest.raises(AssertionError):
        fit.ccv(treatment="X1", pk=0.05, qk=0.5, n_splits=10, seed=929)

    # error when fixed effects in estimation
    fit = feols("Y ~ D | f1", data=data)
    with pytest.raises(NotImplementedError):
        fit.ccv(treatment="D", pk=0.05, qk=0.5, n_splits=10, seed=929)

    # error when treatment not found
    fit = feols("Y ~ D", data=data)
    with pytest.raises(ValueError):
        fit.ccv(treatment="X2", pk=0.05, qk=0.5, n_splits=10, seed=929)

    # error when no cluster variable found
    fit = feols("Y ~ D", data=data)
    with pytest.raises(ValueError):
        fit.ccv(treatment="D", pk=0.05, qk=0.5, n_splits=10, seed=929)

    # error when cluster variable not in data.frame
    with pytest.raises(ValueError):
        fit.ccv(treatment="D", pk=0.05, qk=0.5, n_splits=10, seed=929, cluster="e8")

    # error when two-way clustering
    fit = feols("Y ~ D", data=data, vcov={"CRV1": "f1+f2"})
    with pytest.raises(ValueError):
        fit.ccv(treatment="D", pk=0.05, qk=0.5, n_splits=10, seed=929)

    # error when ccv is attempted on fepois
    pois_data = get_data(model="Fepois").dropna()
    pois_data["D"] = np.random.choice([0, 1], size=len(pois_data))
    fit = fepois("Y ~ D", data=pois_data)
    with pytest.raises(AssertionError):
        fit.ccv(treatment="D", pk=0.05, qk=0.5, n_splits=10, seed=929)

    # same for IV
    fit = feols("Y ~ 1 | D ~ Z1", data=data)
    with pytest.raises(AssertionError):
        fit.ccv(treatment="D", pk=0.05, qk=0.5, n_splits=10, seed=929)


def test_errors_confint():
    data = get_data()
    fit = feols("Y ~ X1", data=data)
    with pytest.raises(ValueError):
        fit.confint(alpha=0.5, keep=["abababa"])


def test_i_error():
    data = get_data()
    data["f2"] = pd.Categorical(data["f2"])

    with pytest.raises(ValueError):
        feols("Y ~ i(f1, f2)", data)

    data["f2"] = data["f2"].astype("object")
    with pytest.raises(ValueError):
        feols("Y ~ i(f1, f2)", data)

    with pytest.raises(FactorEvaluationError):
        feols("Y ~ i(f1, X1, ref=a)", data)


def test_plot_error():
    df = get_data()
    fit = feols("Y ~ X1", data=df)
    with pytest.raises(
        ValueError, match="plot_backend must be either 'lets_plot' or 'matplotlib'."
    ):
        fit.coefplot(plot_backend="plotnine")

    fit_multi = feols("Y + Y2 ~ i(f1)", data=df)
    with pytest.raises(ValueError):
        fit_multi.coefplot(joint=True)

    with pytest.raises(ValueError):
        fit_multi.iplot(joint="both")


def test_ritest_error(data):
    fit = pf.feols("Y ~ X1", data=data)

    with pytest.raises(ValueError, match="X2 not found in the model's coefficients."):
        fit.ritest(resampvar="X2", reps=1000)

    with pytest.raises(ValueError, match="CLUST is not found in the data"):
        fit.ritest(resampvar="X1", reps=1000, cluster="CLUST")

    with pytest.raises(
        ValueError, match="type must be 'randomization-t' or 'randomization-c."
    ):
        fit.ritest(resampvar="X1", reps=1000, type="a")

    with pytest.raises(AssertionError):
        fit.ritest(resampvar="X1", reps=100.4)

    with pytest.raises(ValueError):
        fit.ritest(resampvar="X1", cluster="f1", reps=100)

    with pytest.raises(NotImplementedError):
        fit_iv = pf.feols("Y ~ 1 | X1 ~ Z1", data=data)
        fit_iv.ritest(resampvar="X1", reps=100)

    with pytest.raises(NotImplementedError):
        fit_wls = pf.feols("Y ~ X1", data=data, weights="weights")
        fit_wls.ritest(resampvar="X1", reps=100)

    with pytest.raises(ValueError):
        "Incorrect plot backend."
        fit.ritest(resampvar="X1", reps=100, store_ritest_statistics=True)
        fit.plot_ritest(plot_backend="a")

    with pytest.raises(ValueError):
        "No test_statistics found in the model."
        fit = pf.feols("Y ~ X1", data=data)
        fit.ritest(resampvar="X1", reps=100)
        fit.plot_ritest()


def test_wald_test_invalid_distribution():
    data = pd.read_csv("pyfixest/did/data/df_het.csv")
    data = data.iloc[1:3000]

    fml = "dep_var ~ treat"
    fit = feols(fml, data, vcov={"CRV1": "year"}, ssc=ssc(adj=False))

    with pytest.raises(ValueError):
        fit.wald_test(R=np.array([[1, -1]]), distribution="abc")


def test_wald_test_R_q_column_consistency():
    data = pd.read_csv("pyfixest/did/data/df_het.csv")
    data = data.iloc[1:3000]
    fml = "dep_var ~ treat"
    fit = feols(fml, data, vcov={"CRV1": "year"}, ssc=ssc(adj=False))

    # Test with R.size[1] == number of coeffcients
    with pytest.raises(ValueError):
        fit.wald_test(R=np.array([[1, 0, 0]]))

    # Test with q type
    with pytest.raises(ValueError):
        fit.wald_test(R=np.array([[1, 0]]), q="invalid type q")

    # Test with q being a one-dimensional array or a scalar.
    with pytest.raises(ValueError):
        fit.wald_test(R=np.array([[1, 0], [0, 1]]), q=np.array([[0, 1]]))

    # q must have the same number of rows as R
    with pytest.raises(ValueError):
        fit.wald_test(
            R=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), q=np.array([[0, 1]])
        )


def setup_feiv_instance():
    # Setup necessary data for Feiv

    data = pf.get_data()
    return pf.feols("Y ~ 1 | X1 ~ Z1", data=data)


def test_IV_first_stage_invalid_model_type():
    class NotFeols:
        # Dummy class for testing invalid model type
        pass

    invalid_model = NotFeols()

    with pytest.raises(TypeError):
        feiv_instance = setup_feiv_instance()
        feiv_instance.first_stage(invalid_model)  # This should raise TypeError


def test_IV_Diag_unsupported_statistics():
    feiv_instance = setup_feiv_instance()

    unsupported_statistics = ["unsupported_stat"]

    with pytest.raises(ValueError):
        feiv_instance.IV_Diag(statistics=unsupported_statistics)


def test_errors_compressed():
    data = pf.get_data()

    # no more than two fixed effects
    with pytest.raises(NotImplementedError):
        pf.feols("Y ~ X1 | f1 + f2 + f3", data=data, use_compression=True)

    # cluster variables not in model
    with pytest.raises(NotImplementedError):
        pf.feols("Y ~ X1", vcov={"CRV1": "f1"}, data=data, use_compression=True)

    with pytest.raises(NotImplementedError):
        pf.feols("Y ~ C(f1)", vcov={"CRV1": "f1"}, data=data, use_compression=True)

    with pytest.raises(NotImplementedError):
        pf.feols("Y ~ X1 | f1", data=data, use_compression=True, vcov={"CRV1": "f1+f2"})

    # only CVR supported for Mundlak
    with pytest.raises(NotImplementedError):
        pf.feols("Y ~ X1 | f1", data=data, use_compression=True, vcov="iid")

    # crv3 inference:
    with pytest.raises(NotImplementedError):
        pf.feols("Y ~ X1 | f1", vcov={"CRV3": "f1"}, data=data, use_compression=True)

    # prediction:
    with pytest.raises(NotImplementedError):
        pf.feols("Y ~ X1 | f1", data=data, use_compression=True).predict()

    # argument errors
    with pytest.raises(TypeError):
        pf.feols("Y ~ X1", data=data, use_compression=True, vcov="iid", reps=1.2)

    with pytest.raises(ValueError):
        pf.feols("Y ~ X1", data=data, use_compression=True, vcov="iid", reps=-1)

    with pytest.raises(TypeError):
        pf.feols("Y ~ X1", data=data, use_compression=True, vcov="iid", seed=1.2)

    # no support for IV
    with pytest.raises(NotImplementedError):
        pf.feols("Y ~ 1 | X1 ~ Z1", data=data, use_compression=True)

    # no support for WLS
    with pytest.raises(NotImplementedError):
        pf.feols("Y ~ X1", data=data, weights="weights", use_compression=True)


def test_errors_panelview():
    """Test all ValueError conditions in panelview."""
    sample_df = pd.DataFrame(
        {
            "unit": [1, 2, 3, 4],
            "year": [2001, 2002, 2003, 2004],
            "treat": [0, 1, 1, 0],
            "dep_var": [10, 20, 30, 40],
        }
    )

    # 1. Test for missing 'unit' column
    with pytest.raises(ValueError, match="Column 'unit' not found in data."):
        pf.panelview(sample_df.drop(columns=["unit"]), "unit", "year", "treat")

    # 2. Test for missing 'year' column (time)
    with pytest.raises(ValueError, match="Column 'year' not found in data."):
        pf.panelview(sample_df.drop(columns=["year"]), "unit", "year", "treat")

    # 3. Test for missing 'treat' column
    with pytest.raises(ValueError, match="Column 'treat' not found in data."):
        pf.panelview(sample_df.drop(columns=["treat"]), "unit", "year", "treat")

    # 4. Test for missing outcome column
    with pytest.raises(
        ValueError, match="Outcome column 'nonexistent_col' not found in data."
    ):
        pf.panelview(sample_df, "unit", "year", "treat", outcome="nonexistent_col")

    # 5. Test for 'collapse_to_cohort' and 'subsamp' used together
    with pytest.raises(
        ValueError,
        match="Cannot use 'collapse_to_cohort' together with 'subsamp' or 'units_to_plot'.",
    ):
        pf.panelview(
            sample_df, "unit", "year", "treat", collapse_to_cohort=True, subsamp=10
        )

    # 6. Test for 'collapse_to_cohort' and 'units_to_plot' used together
    with pytest.raises(
        ValueError,
        match="Cannot use 'collapse_to_cohort' together with 'subsamp' or 'units_to_plot'.",
    ):
        pf.panelview(
            sample_df,
            "unit",
            "year",
            "treat",
            collapse_to_cohort=True,
            units_to_plot=[1, 2],
        )


@pytest.mark.parametrize(
    "split, fsplit, expected_exception, error_message",
    [
        # Test TypeError for non-string 'split'
        (123, None, TypeError, "The function argument split needs to be of type str."),
        # Test TypeError for non-string 'fsplit'
        (None, 456, TypeError, "The function argument fsplit needs to be of type str."),
        # Test ValueError for split and fsplit not being identical
        (
            "split_column",
            "different_column",
            ValueError,
            r"Arguments split and fsplit are both specified, but not identical",
        ),
        # Test KeyError for invalid 'split' column
        (
            "invalid_column",
            None,
            KeyError,
            "Column 'invalid_column' not found in data.",
        ),
        # Test KeyError for invalid 'fsplit' column
        (
            None,
            "invalid_column",
            KeyError,
            "Column 'invalid_column' not found in data.",
        ),
    ],
)
def test_split_fsplit_errors(data, split, fsplit, expected_exception, error_message):
    with pytest.raises(expected_exception, match=error_message):
        pf.feols("Y~X1", data=data, split=split, fsplit=fsplit)


def test_separation_check_validations():
    data = pd.DataFrame(
        {
            "Y": [1, 2, 3],
            "X1": [4, 5, 6],
        }
    )

    with pytest.raises(
        ValueError,
        match="The function argument `separation_check` must be a list of strings containing 'fe' and/or 'ir'.",
    ):
        pf.fepois("Y ~ X1", data=data, separation_check=["a"])

    with pytest.raises(
        TypeError,
        match="The function argument `separation_check` must be of type list.",
    ):
        pf.fepois("Y ~ X1", data=data, separation_check="fe")

    with pytest.raises(
        ValueError,
        match="The function argument `separation_check` must be a list of strings containing 'fe' and/or 'ir'.",
    ):
        pf.fepois("Y ~ X1", data=data, separation_check=["fe", "invalid"])


def test_gelbach_errors():
    rng = np.random.default_rng(123)

    data = gelbach_data(nobs=100)
    data["f1"] = rng.choice(range(5), len(data), True)
    data["weights"] = rng.uniform(0.5, 1.5, len(data))

    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

    with pytest.raises(
        ValueError, match=r"The variable 'x32' is not in the mediator names."
    ):
        fit.decompose(param="x1", combine_covariates={"g1": ["x32"]})

    with pytest.raises(
        ValueError, match=r"Variables {'x21'} are in both 'g1' and 'g2' groups."
    ):
        fit.decompose(param="x1", combine_covariates={"g1": ["x21"], "g2": ["x21"]})

    with pytest.raises(TypeError, match=r"combine_covariates_dict must be lists"):
        fit.decompose(param="x1", combine_covariates={"g1": "x21"})

    with pytest.raises(ValueError, match=r"'x99' is not in list"):
        fit.decompose(param="x99")

    with pytest.raises(ValueError, match=r"cannot be included in the x1_vars argument"):
        fit.decompose(decomp_var="x1", x1_vars=["x1"])

    with pytest.raises(
        ValueError, match=r"cannot be in both x1_vars and combine_covariates keys"
    ):
        fit.decompose(
            decomp_var="x1", x1_vars=["x21"], combine_covariates={"g1": ["x21"]}
        )

    med = fit.decompose(param="x1", only_coef=True)
    with pytest.raises(ValueError, match=r"relative_to must be None"):
        med.results.to_dict(relative_to="bogus")

    with pytest.raises(NotImplementedError):
        pf.feols("y ~ 1 | x1 ~ x21", data=data).decompose(
            param="x1", combine_covariates={"g1": ["x21"]}
        )

    with pytest.raises(NotImplementedError):
        pf.feols("y ~ x1", data=data, weights="weights").decompose(
            param="x1", combine_covariates={"g1": ["x21"]}
        )

    with pytest.raises(NotImplementedError):
        dt = pf.get_data(model="Fepois")
        pf.fepois("Y ~ X1", data=dt).decompose(
            param="X1", combine_covariates={"g1": ["x21"]}
        )

    with pytest.raises(
        ValueError, match=r"Either 'param' or 'decomp_var' must be provided\."
    ):
        fit.decompose()

    with pytest.raises(
        ValueError,
        match=r"The 'param' and 'decomp_var' arguments cannot be provided at the same time\.",
    ):
        fit.decompose(param="x1", decomp_var="x1")

    with pytest.warns(
        UserWarning,
        match=r"The 'param' argument is deprecated. Please use 'decomp_var' instead.",
    ):
        fit.decompose(param="x1")

    with pytest.warns(
        UserWarning,
        match=r"You have provided combine_covariates, but agg_first is False. We recommend setting agg_first=True as this might massively decrease the computation time \(in particular when boostrapping CIs\)\.",
    ):
        fit.decompose(
            decomp_var="x1", combine_covariates={"g1": ["x21"]}, agg_first=False
        )


def test_glm_errors():
    "Test that dependent variable must be binary for probit and logit models."
    data = pf.get_data()
    with pytest.raises(
        ValueError, match="The dependent variable must have two unique values."
    ):
        pf.feglm("Y ~ X1", data=data, family="probit")
    with pytest.raises(
        ValueError, match="The dependent variable must have two unique values."
    ):
        pf.feglm("Y ~ X1", data=data, family="logit")

    data["Y"] = np.where(data["Y"] > 0, 2, 0)
    with pytest.raises(
        ValueError, match=r"The dependent variable must be binary \(0 or 1\)."
    ):
        pf.feglm("Y ~ X1", data=data, family="probit")
    with pytest.raises(
        ValueError, match=r"The dependent variable must be binary \(0 or 1\)."
    ):
        pf.feglm("Y ~ X1", data=data, family="logit")

    data["Y"] = np.where(data["Y"] > 0, 1, 0)
    with pytest.raises(
        NotImplementedError, match=r"Fixed effects are not yet supported for GLMs."
    ):
        pf.feglm("Y ~ X1 | f1", data=data, family="probit")


def test_prediction_errors_glm():
    "Test that the prediction errors not supported for GLM models."
    data = pf.get_data()
    data["Y"] = np.where(data["Y"] > 0, 1, 0)

    fit_gaussian = pf.feglm("Y ~ X1", data=data, family="gaussian")
    fit_probit = pf.feglm("Y ~ X1", data=data, family="probit")
    fit_logit = pf.feglm("Y ~ X1", data=data, family="logit")
    fit_pois = pf.fepois("Y ~ X1", data=data)

    for model in [fit_gaussian, fit_probit, fit_logit, fit_pois]:
        with pytest.raises(
            NotImplementedError, match="Prediction with standard errors"
        ):
            model.predict(se_fit=True)


def test_empty_vcov_error():
    data = pf.get_data()
    fit = pf.feols("Y ~ 1 | f1", data=data)

    with pytest.warns(UserWarning):
        fit.tidy()


def test_errors_quantreg(data):
    data = data.dropna()

    # error for CRV3
    with pytest.raises(VcovTypeNotSupportedError):
        pf.quantreg("Y ~ X1", data=data, vcov={"CRV3": "f1"})

    # error for two-way clustering
    with pytest.raises(NotImplementedError):
        pf.quantreg("Y ~ X1", data=data, vcov={"CRV1": "f1+f2"})

    # error for quantile outside [0, 1]
    with pytest.raises(ValueError, match="quantile must be between 0 and 1"):
        pf.quantreg("Y ~ X1", data=data, quantile=1.1)
    with pytest.raises(ValueError, match="quantile must be between 0 and 1"):
        pf.quantreg("Y ~ X1", data=data, quantile=-0.1)

    # error if list provided and quantile is 0, 1, or outside
    with pytest.raises(ValueError, match="quantile must be a list of floats"):
        pf.quantreg("Y ~ X1", data=data, quantile=["0.1", "0.2"])

    # error when fixed effects in formula
    with pytest.raises(NotImplementedError):
        pf.quantreg("Y ~ X1 | f1", data=data)

    # error for invalid method
    with pytest.raises(ValueError, match="`method` must be one of {fn, pfn}"):
        pf.quantreg("Y ~ X1", data=data, method="invalid_method")

    # error for invalid tolerance
    @pytest.mark.parametrize("tol", [0, 1, -0.1, 1.1])
    def test_invalid_tolerance(tol):
        with pytest.raises(ValueError, match=r"tol must be in \(0, 1\)"):
            pf.quantreg("Y ~ X1", data=data, tol=tol)


def test_errors_vcov_kwargs():
    """Test all error conditions for vcov_kwargs in _estimation_input_checks."""
    data = pf.get_data()

    # Error 1: Invalid keys in vcov_kwargs
    with pytest.raises(
        ValueError,
        match="must be a dictionary with keys 'lags', 'time_id', or 'panel_id'",
    ):
        pf.feols("Y ~ X1", data=data, vcov="NW", vcov_kwargs={"invalid_key": 5})

    # Error 2: Multiple invalid keys
    with pytest.raises(
        ValueError,
        match="must be a dictionary with keys 'lags', 'time_id', or 'panel_id'",
    ):
        pf.feols("Y ~ X1", data=data, vcov="NW", vcov_kwargs={"wrong1": 1, "wrong2": 2})

    # Error 3: Mix of valid and invalid keys
    with pytest.raises(
        ValueError,
        match="must be a dictionary with keys 'lags', 'time_id', or 'panel_id'",
    ):
        pf.feols(
            "Y ~ X1",
            data=data,
            vcov="NW",
            vcov_kwargs={"lags": 5, "invalid_key": "test"},
        )

    # Error 4: lags value is not an integer (string)
    with pytest.raises(
        ValueError, match="must be a dictionary with integer values for 'lags'"
    ):
        pf.feols("Y ~ X1", data=data, vcov="NW", vcov_kwargs={"lags": "5"})

    # Error 5: lags value is not an integer (float)
    with pytest.raises(
        ValueError, match="must be a dictionary with integer values for 'lags'"
    ):
        pf.feols("Y ~ X1", data=data, vcov="NW", vcov_kwargs={"lags": 5.5})

    # Error 6: lags value is not an integer (None)
    with pytest.raises(
        ValueError, match="must be a dictionary with integer values for 'lags'"
    ):
        pf.feols("Y ~ X1", data=data, vcov="NW", vcov_kwargs={"lags": None})

    # Error 7: time_id value is not a string (integer)
    with pytest.raises(
        ValueError, match="must be a dictionary with string values for 'time_id'"
    ):
        pf.feols("Y ~ X1", data=data, vcov="NW", vcov_kwargs={"time_id": 123})

    # Error 8: time_id value is not a string (None)
    with pytest.raises(
        ValueError, match="must be a dictionary with string values for 'time_id'"
    ):
        pf.feols("Y ~ X1", data=data, vcov="NW", vcov_kwargs={"time_id": None})

    # Error 9: time_id column does not exist in data
    with pytest.raises(
        ValueError, match="must be a dictionary with string values for 'time_id'"
    ):
        pf.feols(
            "Y ~ X1",
            data=data,
            vcov="NW",
            vcov_kwargs={"time_id": "nonexistent_column"},
        )

    # Error 10: panel_id value is not a string (integer)
    with pytest.raises(
        ValueError, match="must be a dictionary with string values for 'panel_id'"
    ):
        pf.feols("Y ~ X1", data=data, vcov="NW", vcov_kwargs={"panel_id": 456})

    # Error 11: panel_id value is not a string (list)
    with pytest.raises(
        ValueError, match="must be a dictionary with string values for 'panel_id'"
    ):
        pf.feols(
            "Y ~ X1", data=data, vcov="NW", vcov_kwargs={"panel_id": ["col1", "col2"]}
        )

    # Error 12: panel_id column does not exist in data
    with pytest.raises(
        ValueError, match="must be a dictionary with string values for 'panel_id'"
    ):
        pf.feols(
            "Y ~ X1",
            data=data,
            vcov="NW",
            vcov_kwargs={"panel_id": "missing_panel_column"},
        )


def test_errors_hac():
    """Test all error conditions for HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors."""
    data = pf.get_data()
    # Add a time variable for testing
    data["time"] = np.arange(len(data))
    data["panel"] = np.repeat(np.arange(len(data) // 10), 10)[: len(data)]

    # Error 1: Driscoll-Kraay HAC not implemented
    with pytest.raises(
        NotImplementedError,
        match="Driscoll-Kraay HAC standard errors are not yet implemented",
    ):
        pf.feols(
            "Y ~ X1", data=data, vcov="DK", vcov_kwargs={"time_id": "time", "lags": 3}
        )

    # Error 2: Panel-clustered Newey-West HAC not implemented
    with pytest.raises(
        NotImplementedError,
        match="Panel-clustered Newey-West HAC standard errors are not yet implemented",
    ):
        pf.feols(
            "Y ~ X1",
            data=data,
            vcov="NW",
            vcov_kwargs={"time_id": "time", "panel_id": "panel", "lags": 3},
        )
