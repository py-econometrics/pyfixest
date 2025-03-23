import pytest

from pyfixest.utils.utils import get_ssc, _count_fixef_fully_nested


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
    N, k, _, vcov_sign = params

    G = N

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
    assert res == (N - 1) / (N - k) * N / (N - 1)

    ssc_dict["adj"] = False
    res = get_ssc(ssc_dict, N, k, G, vcov_sign, "hetero")
    assert res == N / (N - 1)


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



def test_count_fixef_fully_nested():

    clusters = np.array([1, 1, 2, 2, 2, 1, 1, 2, 2, 2])
    id = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    time = np.array([1 ,1, 1, 1, 1, 2, 2, 2, 2, 2])

    # test 1
    f = np.concatenate([id.reshape(-1, 1), time.reshape(-1, 1)], axis = 1)
    res = _count_fixef_fully_nested(clusters, f)
    assert res == 5, "Did not find 5 fully nested fixed effects."

    # test 2
    res = _count_fixef_fully_nested(clusters, clusters)
    assert res == 2, "Did not find 2 fully nested fixed effects."