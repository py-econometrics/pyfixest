import subprocess
import sys
import textwrap

import numpy as np
import pytest

import pyfixest as pf


def _fit_and_data():
    data = pf.get_data().dropna()
    fit = pf.feols("Y ~ X1 + X2", data)
    return fit, data


@pytest.mark.parametrize("vcov", [{"CRV1": "f1"}, {"CRV3": "f1"}])
def test_reordered_frame_gives_same_clustered_se(vcov):
    fit, data = _fit_and_data()
    reference = fit.vcov(vcov).se().to_numpy()

    shuffled = data.sample(frac=1, random_state=9)
    se = fit.vcov(vcov, data=shuffled).se().to_numpy()

    np.testing.assert_allclose(se, reference, rtol=1e-10, atol=0)


@pytest.mark.parametrize("vcov", [{"CRV1": "f1"}, {"CRV3": "f1"}, "hetero"])
def test_shorter_frame_raises_value_error(vcov):
    fit, data = _fit_and_data()
    with pytest.raises(ValueError):
        fit.vcov(vcov, data=data.iloc[:100])


def test_frame_with_different_values_raises_value_error():
    fit, data = _fit_and_data()
    other = data.copy()
    other["Y"] = other["Y"] + 1.0
    with pytest.raises(ValueError):
        fit.vcov({"CRV1": "f1"}, data=other)


@pytest.mark.parametrize("vcov", ["{'CRV1': 'f1'}", "{'CRV3': 'f1'}", "'hetero'"])
def test_longer_frame_raises_value_error_without_aborting(vcov):
    # Before the fix, a longer frame made the Rust CRV1 kernel panic and abort
    # the whole process. Run in a subprocess so a regression cannot kill pytest.
    script = textwrap.dedent(
        f"""
        import pandas as pd
        import pyfixest as pf

        d = pf.get_data().dropna()
        fit = pf.feols("Y ~ X1 + X2", d)
        try:
            fit.vcov({vcov}, data=pd.concat([d, d]))
        except ValueError:
            print("ValueError")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr
    assert "ValueError" in result.stdout
