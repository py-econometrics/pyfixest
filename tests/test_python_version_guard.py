import importlib
import sys

import pytest


def test_import_has_clear_error_on_python39(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 9, 23))
    sys.modules.pop("pyfixest", None)

    with pytest.raises(ImportError, match=r"requires Python >=3\.10"):
        importlib.import_module("pyfixest")
