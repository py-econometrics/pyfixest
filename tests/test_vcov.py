import numpy as np
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# rpy2 imports
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.utils.utils import ssc

pandas2ri.activate()

fixest = importr("fixest")
stats = importr("stats")


@pytest.fixture()
def data():
    return pf.get_data()


models = ["Y~X1", "Y~X1 | f1"]
adj = [False, True]
cluster_adj = [False, True]


@pytest.mark.parametrize("fml", models)
@pytest.mark.parametrize("adj", adj)
@pytest.mark.parametrize("cluster_adj", cluster_adj)
def test_iid(data, fml, adj, cluster_adj):
    py_mod = pf.feols(
        fml, data=data, vcov="iid", ssc=ssc(adj=adj, cluster_adj=cluster_adj)
    )
    r_mod = fixest.feols(
        ro.Formula(fml),
        data=data,
        vcov="iid",
        ssc=fixest.ssc(adj, "none", cluster_adj, "min", "min", False),
    )

    py_mod_vcov = py_mod._vcov
    r_mod_vcov = stats.vcov(r_mod)

    assert np.all(np.abs(py_mod_vcov) - np.abs(r_mod_vcov) < 1e-16)


@pytest.mark.parametrize("fml", models)
@pytest.mark.parametrize("adj", adj)
@pytest.mark.parametrize("cluster_adj", cluster_adj)
def test_hetero(data, fml, adj, cluster_adj):
    py_mod = pf.feols(
        fml, data=data, vcov="hetero", ssc=ssc(adj=adj, cluster_adj=cluster_adj)
    )
    r_mod = fixest.feols(
        ro.Formula(fml),
        data=data,
        vcov="hetero",
        ssc=fixest.ssc(adj, "none", cluster_adj, "min", "min", False),
    )

    py_mod_vcov = py_mod._vcov
    r_mod_vcov = stats.vcov(r_mod)

    assert np.all(np.abs(py_mod_vcov) - np.abs(r_mod_vcov) < 1e-16)


@pytest.mark.parametrize("fml", models)
@pytest.mark.parametrize("adj", adj)
@pytest.mark.parametrize("cluster_adj", cluster_adj)
def test_crv1(data, fml, adj, cluster_adj):
    py_mod = pf.feols(
        fml,
        data=data,
        vcov={"CRV1": "group_id"},
        ssc=ssc(adj=adj, cluster_adj=cluster_adj),
    )
    r_mod = fixest.feols(
        ro.Formula(fml),
        data=data,
        vcov=ro.Formula("~group_id"),
        ssc=fixest.ssc(adj, "none", cluster_adj, "min", "min", False),
    )

    py_mod_vcov = py_mod._vcov
    r_mod_vcov = stats.vcov(r_mod)

    assert np.all(np.abs(py_mod_vcov) - np.abs(r_mod_vcov) < 1e-16)


@pytest.mark.parametrize("fml", models)
@pytest.mark.parametrize("adj", adj)
@pytest.mark.parametrize("cluster_adj", cluster_adj)
def test_iid_weights(data, fml, adj, cluster_adj):
    py_mod = pf.feols(
        fml,
        data=data,
        vcov="iid",
        weights="weights",
        ssc=ssc(adj=adj, cluster_adj=cluster_adj),
    )
    r_mod = fixest.feols(
        ro.Formula(fml),
        data=data,
        vcov="iid",
        weights=ro.Formula("~weights"),
        ssc=fixest.ssc(adj, "none", cluster_adj, "min", "min", False),
    )

    py_mod_vcov = py_mod._vcov
    r_mod_vcov = stats.vcov(r_mod)

    assert np.all(np.abs(py_mod_vcov) - np.abs(r_mod_vcov) < 1e-16)


@pytest.mark.parametrize("fml", models)
@pytest.mark.parametrize("adj", adj)
@pytest.mark.parametrize("cluster_adj", cluster_adj)
def test_hetero_weights(data, fml, adj, cluster_adj):
    py_mod = pf.feols(
        fml,
        data=data,
        vcov="hetero",
        weights="weights",
        ssc=ssc(adj=adj, cluster_adj=cluster_adj),
    )
    r_mod = fixest.feols(
        ro.Formula(fml),
        data=data,
        vcov="hetero",
        weights=ro.Formula("~weights"),
        ssc=fixest.ssc(adj, "none", cluster_adj, "min", "min", False),
    )

    py_mod_vcov = py_mod._vcov
    r_mod_vcov = stats.vcov(r_mod)

    assert np.all(np.abs(py_mod_vcov) - np.abs(r_mod_vcov) < 1e-16)


@pytest.mark.parametrize("fml", models)
@pytest.mark.parametrize("adj", adj)
@pytest.mark.parametrize("cluster_adj", cluster_adj)
def test_crv1_weights(data, fml, adj, cluster_adj):
    py_mod = pf.feols(
        fml,
        data=data,
        vcov={"CRV1": "group_id"},
        weights="weights",
        ssc=ssc(adj=adj, cluster_adj=cluster_adj),
    )
    r_mod = fixest.feols(
        ro.Formula(fml),
        data=data,
        vcov=ro.Formula("~group_id"),
        weights=ro.Formula("~weights"),
        ssc=fixest.ssc(adj, "none", cluster_adj, "min", "min", False),
    )

    py_mod_vcov = py_mod._vcov
    r_mod_vcov = stats.vcov(r_mod)

    assert np.all(np.abs(py_mod_vcov) - np.abs(r_mod_vcov) < 1e-16)
