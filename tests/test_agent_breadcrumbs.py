from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_module_docstring_preserves_lazy_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import sys

        import pyfixest as pf

        assert "https://pyfixest.org/llms.txt" in pf.__doc__
        assert "https://pyfixest.org/skills.html" in pf.__doc__
        assert "https://github.com/py-econometrics/pyfixest" in pf.__doc__
        assert "pyfixest.estimation.api.feols" not in sys.modules
        assert pf.feols.__name__ == "feols"
        assert "pyfixest.estimation.api.feols" in sys.modules
        assert set(dir(pf)) == set(pf.__all__)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
