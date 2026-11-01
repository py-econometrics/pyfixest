from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def _run_in_fresh_interpreter(script: str) -> None:
    """Run `script` in a subprocess that imports pyfixest for the first time."""
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_module_docstring_points_at_docs_and_source() -> None:
    _run_in_fresh_interpreter(
        """
        import pyfixest as pf

        assert 'importlib.resources.files("pyfixest") / "docs"' in pf.__doc__
        assert "https://pyfixest.org/llms.txt" in pf.__doc__
        assert "https://github.com/py-econometrics/pyfixest" in pf.__doc__
        """
    )


def test_module_docstring_preserves_lazy_imports() -> None:
    _run_in_fresh_interpreter(
        """
        import sys

        import pyfixest as pf

        assert "pyfixest.estimation.api.feols" not in sys.modules
        assert pf.feols.__name__ == "feols"
        assert "pyfixest.estimation.api.feols" in sys.modules
        assert set(dir(pf)) == set(pf.__all__)
        """
    )
