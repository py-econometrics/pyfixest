import pytest
from pyfixest.ssc_utils import get_ssc




def test_get_ssc1():

    ssc_dict = dict({'adj' : False, 'fixef_k' : "none", 'cluster_adj' : False, 'cluster_df' : "conventional"})

    N = 100
    k = 1
    G = 10
    vcov_sign = 1
    is_clustered = False

    res = get_ssc(ssc_dict, N, k, G, vcov_sign, is_clustered)

    assert res == 1


def test_get_ssc2():

    ssc_dict = dict({'adj' : True, 'fixef_k' : "none", 'cluster_adj' : False, 'cluster_df' : "conventional"})

    N = 100
    k = 1
    G = 10
    vcov_sign = 1
    is_clustered = False

    res = get_ssc(ssc_dict, N, k, G, vcov_sign, is_clustered)

    assert res == (N-1) / (N-k)

def test_get_ssc3():

    ssc_dict = dict({'adj' : False, 'fixef_k' : "none", 'cluster_adj' : True, 'cluster_df' : "conventional"})

    N = 100
    k = 1
    G = 10
    vcov_sign = 1
    is_clustered = True

    res = get_ssc(ssc_dict, N, k, G, vcov_sign, is_clustered)

    assert res == G / (G-1)

def test_get_ssc4():

    ssc_dict = dict({'adj' : False, 'fixef_k' : "none", 'cluster_adj' : True, 'cluster_df' : "conventional"})

    N = 100
    k = 1
    G = 10
    vcov_sign = 1
    is_clustered = True

    res = get_ssc(ssc_dict, N, k, G, vcov_sign, is_clustered)

    assert res == (N-1) / (N-k) * G / (G-1)