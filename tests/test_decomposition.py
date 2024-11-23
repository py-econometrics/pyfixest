import pandas as pd
import numpy as np
import pyfixest as pf

def gelbach_test_data(nobs):

    rng = np.random.default_rng(49392)
    df = pd.DataFrame(index=range(nobs))
    df['x1'] = rng.normal(size=nobs)

    df['x21'] = df['x1'] * 1 + rng.normal(loc = 0, scale = 0.1, size=nobs)
    df['x22'] = df['x1'] * 0.25 + df['x21'] * 0.75 + rng.normal(size=nobs)
    df['x23'] = (df['x1'] * 0.4 +
                df['x21'] * 0.6 +
                df['x22'] * 0.4 +
                rng.normal(loc = 0, scale = 0.1, size=nobs)
    )
    df['y'] = (df['x1'] * 1 +
            df['x21'] * 2 +
            df['x22'] * 0.5 +
            df['x23'] * 0.75 +
            rng.normal(loc = 0, scale = 0.1, size=nobs)
    )

    return df

data = gelbach_test_data(nobs = 100_000)
fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)
res = fit.decompose(param = "x1", combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]})

def test_gelbach_example():

    test_ci = False

    data = gelbach_test_data()
    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)
    res = fit.decompose(param = "x1", combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]}).GelbachDecompositionResults

    np.testing.assert_allclose(res.contribution_dict.get("g1"), np.array([2.468092]))
    np.testing.assert_allclose(res.contribution_dict.get("g2"), np.array([1.068156]))

    np.testing.assert_allclose(res.contribution_dict.get("direct_effect"), np.array([4.608666]))
    np.testing.assert_allclose(res.contribution_dict.get("full_effect"), np.array([1.072417]))
    np.testing.assert_allclose(res.contribution_dict.get("explained_effect"), np.array([3.536249]))


    if test_ci:

        np.testing.assert_allclose(res.ci.get("g1"), np.array([2.293714, 2.64247]))
        np.testing.assert_allclose(res.ci.get("g2"), np.array([2.9546626, 1.18165]))

        np.testing.assert_allclose(res.ci.get("direct_effect"), np.array([ 4.401328, 4.816004]))
        np.testing.assert_allclose(res.ci.get("full_effect"), np.array([.9775936, 1.167241]))
        np.testing.assert_allclose(res.ci.get("explained_effect"), np.array([3.3262, 3.746298]))

