import subprocess
import sys
import textwrap


def test_core_estimation_without_reporting_dependencies():
    code = textwrap.dedent(
        """
        import builtins

        blocked = {
            "great_tables",
            "lets_plot",
            "maketables",
            "matplotlib",
            "seaborn",
            "tabulate",
            "tqdm",
        }
        import_original = builtins.__import__

        def import_without_optional_dependencies(name, *args, **kwargs):
            if name.split(".", maxsplit=1)[0] in blocked:
                raise ImportError(f"blocked optional dependency: {name}")
            return import_original(name, *args, **kwargs)

        builtins.__import__ = import_without_optional_dependencies

        import pyfixest as pf
        from pyfixest.report.utils import rename_categoricals

        assert callable(rename_categoricals)

        fit = pf.feols("Y ~ X1", pf.get_data(N=100))
        fit.summary()

        try:
            fit.etable()
        except ImportError as exc:
            assert "pyfixest[tables]" in str(exc)
        else:
            raise AssertionError("etable() should require the tables extra")

        try:
            fit.coefplot()
        except ImportError as exc:
            assert "pyfixest[plots]" in str(exc)
        else:
            raise AssertionError("coefplot() should require a plotting backend")
        """
    )

    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
