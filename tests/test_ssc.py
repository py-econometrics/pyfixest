import pytest

from pyfixest.utils.utils import get_ssc


@pytest.fixture
def params():
    return [100, 1, 10, 1]


def test_iid(params):
    N, k, G, vcov_sign = params

    ssc_dict = {
        "adj": False,
        "fixef_k": "none",
        "cluster_adj": False,
        "cluster_df": "conventional",
    }
    res = get_ssc(ssc_dict, N, k, G, vcov_sign, "iid")
    assert res == 1

    ssc_dict["adj"] = True
    res = get_ssc(ssc_dict, N, k, G, vcov_sign, "iid")
    assert res == (N - 1) / (N - k)


def test_HC(params):
    N, k, G, vcov_sign = params

    ssc_dict = {
        "adj": False,
        "fixef_k": "none",
        "cluster_adj": False,
        "cluster_df": "conventional",
    }
    res = get_ssc(ssc_dict, N, k, G, vcov_sign, "hetero")
    assert res == 1

    ssc_dict["adj"] = True
    res = get_ssc(ssc_dict, N, k, G, vcov_sign, "hetero")
    assert res == (N - 1) / (N - k)

    ssc_dict["cluster_adj"] = True
    res = get_ssc(ssc_dict, N, k, G, vcov_sign, "hetero")
    assert res == (N - 1) / (N - k)


def test_CRV(params):
    N, k, G, vcov_sign = params

    ssc_dict = {
        "adj": False,
        "fixef_k": "none",
        "cluster_adj": False,
        "cluster_df": "conventional",
    }
    res = get_ssc(ssc_dict, N, k, G, vcov_sign, "CRV")
    assert res == 1

    ssc_dict["adj"] = True
    res = get_ssc(ssc_dict, N, k, G, vcov_sign, "CRV")
    assert res == (N - 1) / (N - k)

    ssc_dict["cluster_adj"] = True
    res = get_ssc(ssc_dict, N, k, G, vcov_sign, "CRV")
    assert res == (N - 1) / (N - k) * G / (G - 1)

    ssc_dict["cluster_df"] = "min"
    res = get_ssc(ssc_dict, N, k, G, vcov_sign, "CRV")
    assert res == (N - 1) / (N - k) * G / (G - 1)

    ssc_dict["adj"] = False
    res = get_ssc(ssc_dict, N, k, G, vcov_sign, "CRV")
    assert res == G / (G - 1)
