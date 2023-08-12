import pytest
from pyfixest.FormulaParser import _unpack_fml, _pack_to_fml, _find_sw, _flatten_list


def test_unpack_fml():
    assert _unpack_fml("a+b+c") == {"constant": ["a", "b", "c"]}

    assert _unpack_fml("sw(x,y)") == {"constant": [], "sw": ["x", "y"]}

    assert _unpack_fml("a+sw0(x,y)+d") == {"constant": ["a", "d"], "sw0": ["x", "y"]}

    assert _unpack_fml("csw(x,y)") == {"constant": [], "csw": ["x", "y"]}

    assert _unpack_fml("csw0(x,y,z)") == {"constant": [], "csw0": ["x", "y", "z"]}

    assert _unpack_fml("a+b+csw0(x,y,z)") == {
        "constant": ["a", "b"],
        "csw0": ["x", "y", "z"],
    }


def test_pack_to_fml():
    # assert _pack_to_fml([]) == []

    assert _pack_to_fml({"constant": ["x", "y"], "sw0": ["a", "b"]}) == [
        "x+y",
        "x+y+a",
        "x+y+b",
    ]
    assert _pack_to_fml({"constant": [], "sw0": ["a", "b"]}) == ["0", "a", "b"]
    assert _pack_to_fml({"constant": ["x", "y"], "sw0": []}) == ["x+y"]

    assert _pack_to_fml({"constant": ["x", "y"], "sw": ["a", "b"]}) == [
        "x+y+a",
        "x+y+b",
    ]
    assert _pack_to_fml({"constant": [], "sw": ["a", "b"]}) == ["a", "b"]
    assert _pack_to_fml({"constant": ["x", "y"], "sw": []}) == ["x+y"]

    assert _pack_to_fml({"constant": ["x", "y"], "csw0": ["a", "b"]}) == [
        "x+y",
        "x+y+a",
        "x+y+a+b",
    ]
    assert _pack_to_fml({"constant": [], "csw0": ["a", "b"]}) == ["0", "a", "a+b"]
    assert _pack_to_fml({"constant": ["x", "y"], "csw0": []}) == ["x+y"]

    assert _pack_to_fml({"constant": ["x", "y"], "csw": ["a", "b"]}) == [
        "x+y+a",
        "x+y+a+b",
    ]
    assert _pack_to_fml({"constant": [], "csw": ["a", "b"]}) == ["a", "a+b"]
    assert _pack_to_fml({"constant": ["x", "y"], "csw": []}) == ["x+y"]


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


# def test_find_sw_multiple_matches():
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
