from pyfixest.estimation.deprecated.FormulaParser import (
    _dict_to_list_of_formulas,
    _find_multiple_estimation_syntax,
    _input_formula_to_dict,
)


def test_input_formula_to_dict():
    assert _input_formula_to_dict("a+b+c") == {"constant": ["a", "b", "c"]}

    assert _input_formula_to_dict("sw(x,y)") == {"constant": [], "sw": ["x", "y"]}

    assert _input_formula_to_dict("a+sw0(x,y)+d") == {
        "constant": ["a", "d"],
        "sw0": ["x", "y"],
    }

    assert _input_formula_to_dict("csw(x,y)") == {"constant": [], "csw": ["x", "y"]}

    assert _input_formula_to_dict("csw0(x,y,z)") == {
        "constant": [],
        "csw0": ["x", "y", "z"],
    }

    assert _input_formula_to_dict("a+b+csw0(x,y,z)") == {
        "constant": ["a", "b"],
        "csw0": ["x", "y", "z"],
    }


def test_dict_to_list_of_formulas():
    # assert _dict_to_list_of_formulas([]) == []

    assert _dict_to_list_of_formulas({"constant": ["x", "y"], "sw0": ["a", "b"]}) == [
        "x+y",
        "x+y+a",
        "x+y+b",
    ]
    assert _dict_to_list_of_formulas({"constant": [], "sw0": ["a", "b"]}) == [
        "0",
        "a",
        "b",
    ]
    assert _dict_to_list_of_formulas({"constant": ["x", "y"], "sw0": []}) == ["x+y"]

    assert _dict_to_list_of_formulas({"constant": ["x", "y"], "sw": ["a", "b"]}) == [
        "x+y+a",
        "x+y+b",
    ]
    assert _dict_to_list_of_formulas({"constant": [], "sw": ["a", "b"]}) == ["a", "b"]
    assert _dict_to_list_of_formulas({"constant": ["x", "y"], "sw": []}) == ["x+y"]

    assert _dict_to_list_of_formulas({"constant": ["x", "y"], "csw0": ["a", "b"]}) == [
        "x+y",
        "x+y+a",
        "x+y+a+b",
    ]
    assert _dict_to_list_of_formulas({"constant": [], "csw0": ["a", "b"]}) == [
        "0",
        "a",
        "a+b",
    ]
    assert _dict_to_list_of_formulas({"constant": ["x", "y"], "csw0": []}) == ["x+y"]

    assert _dict_to_list_of_formulas({"constant": ["x", "y"], "csw": ["a", "b"]}) == [
        "x+y+a",
        "x+y+a+b",
    ]
    assert _dict_to_list_of_formulas({"constant": [], "csw": ["a", "b"]}) == [
        "a",
        "a+b",
    ]
    assert _dict_to_list_of_formulas({"constant": ["x", "y"], "csw": []}) == ["x+y"]


def test_find_multiple_estimation_syntax_no_match():
    x = "a + b + c"
    assert _find_multiple_estimation_syntax(x) == (x, None)


def test_find_multiple_estimation_syntax_sw():
    x = "sw(a, b, c)"
    expected = (["a", " b", " c"], "sw")
    assert _find_multiple_estimation_syntax(x) == expected


def test_find_multiple_estimation_syntax_csw():
    x = "csw(a, b, c)"
    expected = (["a", " b", " c"], "csw")
    assert _find_multiple_estimation_syntax(x) == expected


def test_find_multiple_estimation_syntax_sw0():
    x = "sw0(a, b, c)"
    expected = (["a", " b", " c"], "sw0")
    assert _find_multiple_estimation_syntax(x) == expected


def test_find_multiple_estimation_syntax_csw0():
    x = "csw0(a, b, c)"
    expected = (["a", " b", " c"], "csw0")
    assert _find_multiple_estimation_syntax(x) == expected
