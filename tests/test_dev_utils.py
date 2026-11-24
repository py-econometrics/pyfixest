import pandas as pd
import pytest

from pyfixest.utils.dev_utils import _missing_columns, _require_columns


def test_missing_columns_returns_absent_names_in_order():
    """Absent names are returned in input order; None entries are skipped."""
    data = pd.DataFrame({"a": [1], "b": [2]})
    assert _missing_columns(data, ["a", "nope", None, "b", "nah"]) == ["nope", "nah"]
    assert _missing_columns(data, ["a", "b", None]) == []
    assert _missing_columns(data, []) == []


def test_require_columns_passes_when_present_or_none():
    """Existing columns and None names do not raise."""
    data = pd.DataFrame({"a": [1]})
    _require_columns(
        data,
        [("x", "a"), ("y", None)],
        message=lambda label, column: f"{label}={column!r}",
    )


def test_require_columns_raises_first_missing_with_label():
    """The first absent column raises, with the label and name in the message."""
    data = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match=r"x='nope'"):
        _require_columns(
            data,
            [("x", "a"), ("x", "nope"), ("y", "nah")],
            message=lambda label, column: f"{label}={column!r}",
        )


def test_require_columns_custom_error_type():
    """The error type is configurable, e.g. KeyError for split/fsplit."""
    data = pd.DataFrame({"a": [1]})
    with pytest.raises(KeyError, match=r"Column 'nah' not found"):
        _require_columns(
            data,
            [("split", "nah")],
            error=KeyError,
            message=lambda label, column: f"Column {column!r} not found",
        )
