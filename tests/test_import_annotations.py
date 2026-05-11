import subprocess
import sys
import textwrap
from pathlib import Path


def test_stringified_dataframe_type_does_not_break_model_import():
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        f"""
        import importlib
        import sys

        sys.path.insert(0, {str(repo_root)!r})

        import pyfixest.utils.dev_utils as dev_utils

        # Reproduce the older narwhals behavior reported in issue #1263, where
        # the imported typing alias behaved like a string at runtime.
        dev_utils.DataFrameType = "IntoDataFrame"

        importlib.import_module("pyfixest.estimation.models.feglm_")
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
