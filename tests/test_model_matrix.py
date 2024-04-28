import pytest

import pyfixest as pf


# Define the fixture to provide data
@pytest.fixture
def data():
    return pf.get_data()


# Parameterize the test function directly with formulas
@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ i(f1)",
        "Y ~ i(f1, ref = 1.0)",
        "Y ~ i(f1, X1)",
        "Y ~ i(f1, X1, ref = 2.0)",
        "Y ~ i(f1) + X2",
        "Y ~ i(f1, ref = 1.0) + X2",
        "Y ~ i(f1, X1) + X2",
        "Y ~ i(f1, X1, ref = 2.0) + X2",
    ],
)
def test_get_icovars(data, fml):
    # Use the data and fml from the fixture and parameterization
    fit = pf.feols(fml, data=data)
    assert len(fit._icovars) > 0, "No icovars found"
    assert "X2" not in fit._icovars, "X2 is found in _icovars"
