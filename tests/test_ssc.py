import pytest
from pyfixest.ssc_utils import get_ssc



@pytest.mark.skip("currently broken due to changes of ssc's")
def test_get_ssc1():

    ssc_dict = dict({'adj' : False, 'fixef_k' : "none", 'cluster_adj' : False, 'cluster_df' : "conventional"})

    N = 100
    k = 1
    G = 10
    vcov_sign = 1
    vcov_type = "iid"

    res = get_ssc(ssc_dict, N, k, G, vcov_sign, vcov_type)

    assert res == 1

@pytest.mark.skip("currently broken due to changes of ssc's")
def test_get_ssc2():

    ssc_dict = dict({'adj' : True, 'fixef_k' : "none", 'cluster_adj' : False, 'cluster_df' : "conventional"})

    N = 100
    k = 1
    G = 10
    vcov_sign = 1
    vcov_type = False

    res = get_ssc(ssc_dict, N, k, G, vcov_sign, vcov_type)

    assert res == (N-1) / (N-k)

@pytest.mark.skip("currently broken due to changes of ssc's")
def test_get_ssc3():

    ssc_dict = dict({'adj' : False, 'fixef_k' : "none", 'cluster_adj' : True, 'cluster_df' : "conventional"})

    N = 100
    k = 1
    G = 10
    vcov_sign = 1
    is_clustered = True

    res = get_ssc(ssc_dict, N, k, G, vcov_sign, is_clustered)

    assert res == G / (G-1)

@pytest.mark.skip("currently broken due to changes of ssc's")
def test_get_ssc4():

    ssc_dict = dict({'adj' : False, 'fixef_k' : "none", 'cluster_adj' : True, 'cluster_df' : "conventional"})

    N = 100
    k = 1
    G = 10
    vcov_sign = 1
    is_clustered = True

    res = get_ssc(ssc_dict, N, k, G, vcov_sign, is_clustered)

    assert res == (N-1) / (N-k) * G / (G-1)