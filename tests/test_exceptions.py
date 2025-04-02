from pyfixest.utils._exceptions import find_stack_level


def test_find_stack_level():
    # The function returns 1 when called from test code because we're one level
    # away from the pyfixest package
    level = find_stack_level()
    assert level >= 1, "Stack level should be at least 1 when called from tests"

    def wrapper_function():
        # Should return a higher level as we're nested deeper
        return find_stack_level()

    nested_level = wrapper_function()
    assert nested_level >= level, "Nested call should return same or higher stack level"


def test_find_stack_level_imports():
    # Test that we can properly import and access the package
    import pyfixest as pf

    assert pf is not None
    assert hasattr(pf, "__file__")
