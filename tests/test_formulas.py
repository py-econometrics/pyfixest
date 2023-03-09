import pytest
from pyfixest.FormulaParser import _unpack_fml, _pack_to_fml, _find_sw, _flatten_list

def test_unpack_fml():

    assert _unpack_fml('a+b+c') == ['a', 'b', 'c']

    assert _unpack_fml('a+sw(b,c)+d') == ['a', 'd', ['b', 'c']]

    assert _unpack_fml('a+sw0(b,c)+d') == ['a', 'd',['0','b', 'c']]

    assert _unpack_fml('a+csw(b,c)+d') == ['a', 'd', ['b', 'b+c']]

    assert _unpack_fml('a+csw0(b,c)+d') == ['a', 'd', ['0','b', 'b+c']]

    #with pytest.raises(ValueError):
    #    _unpack_fml('a + dsw(b, c) + e')



def test_pack_to_fml():

    #assert _pack_to_fml([]) == []

    assert _pack_to_fml(['a', 'b', 'c']) == ['a+b+c']

    #assert _pack_to_fml([['a+b', 'a+c'], 'd']) == ['a+b+d', 'a+c+d']

    #assert _pack_to_fml([['a', 'b', 'c'], 'd', 'e']) == ['a+d+e', 'b+d+e', 'c+d+e']


def test_find_sw_no_match():
    x = "a + b + c"
    assert _find_sw(x) == (x, None)

def test_find_sw_sw():
    x = "sw(a, b, c)"
    expected = (["a", " b", " c"], "sw")
    assert _find_sw(x) == expected

def test_find_sw_csw():
    x = "csw(a, b, c)"
    expected = (["a", " b", " c"], "csw")
    assert _find_sw(x) == expected

def test_find_sw_sw0():
    x = "sw0(a, b, c)"
    expected = (["a", " b", " c"], "sw0")
    assert _find_sw(x) == expected

def test_find_sw_csw0():
    x = "csw0(a, b, c)"
    expected = (["a", " b", " c"], "csw0")
    assert _find_sw(x) == expected

#def test_find_sw_multiple_matches():
#    x = "sw(a, b, c) + csw(d, e) + sw0(f, g)"
#    expected = (["a", " b", " c"], "sw")
#    assert _find_sw(x) == expected


def test_flatten_list():
    assert _flatten_list([1, 2, 3]) == [1, 2, 3]
    assert _flatten_list([[1, 2, 3], 4, 5]) == [1, 2, 3, 4, 5]
    assert _flatten_list([[[1, 2], [3, 4]], 5]) == [1, 2, 3, 4, 5]
    assert _flatten_list([[], [1, [2, [3]]], [4, 5]]) == [1, 2, 3, 4, 5]
    assert _flatten_list([1, [2, [3, [4, [5]]]]]) == [1, 2, 3, 4, 5]
    assert _flatten_list([]) == []

