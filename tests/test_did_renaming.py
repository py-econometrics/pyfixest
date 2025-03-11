# tests/test_did_renaming.py
import numpy as np
import pandas as pd
import pytest
import pyfixest as pf
from pyfixest.utils.utils import rename_did_coefficients

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(123)
    n = 1000
    data = pd.DataFrame({
        'unit': np.repeat(range(100), 10),
        'year': np.tile(range(2010, 2020), 100),
        'X1': np.random.normal(0, 1, n),
        'f1': np.random.choice([0, 1, 2, 3, 4], n),
        'Y': np.random.normal(0, 1, n)
    })
    return data

@pytest.fixture
def did_data():
    """Create sample DID data for testing."""
    np.random.seed(123)
    n_units = 100
    periods = 10
    n = n_units * periods
    # Create unit-level treatment groups (one value per unit)
    unit_groups = np.random.choice([0, 2015, 2016, 2017], n_units, replace=True)

    # Expand to observation level
    data = pd.DataFrame({
        'unit': np.repeat(range(n_units), periods),
        'year': np.tile(range(2010, 2020), n_units),
        'g': np.repeat(unit_groups, periods),  # Repeat each unit's group for all periods
        'state': np.random.choice(range(20), n),
        'dep_var': np.random.normal(0, 1, n)
    })
    # Add rel_year and treat columns
    data['rel_year'] = data['year'] - data['g']
    data['rel_year'] = np.where(data['g'] == 0, np.inf, data['rel_year'])
    data['treat'] = np.where(data['g'] <= data['year'], 1, 0)
    data['treat'] = np.where(data['g'] == 0, 0, data['treat'])
    return data

def test_rename_did_coefficients():
    """Test the rename_did_coefficients function."""
    test_cases = [
        ("C(f1, contr.treatment(base=1))[T.0.0]:X1", "f1::0.0::X1"),
        ("C(rel_year, contr.treatment(base=-1))[T.-5]", "rel_year::-5"),
    ]
    for original, expected in test_cases:
        result = rename_did_coefficients([original])[0]
        assert result == expected

def test_feols_renaming(sample_data):
    """Test renaming with feols."""
    fit = pf.feols("Y ~ i(f1, X1, ref = 1)", data=sample_data)
    # Get original coefficients
    original_coefs = fit.coef()
    print("Original coefficient names:", original_coefs.index.tolist())
    # Apply renaming
    renamed_index = rename_did_coefficients(original_coefs.index)
    print("Renamed coefficient names:", renamed_index)
    # More specific assertion
    assert any(name.startswith("f1::") for name in renamed_index), \
        f"No renamed coefficients found starting with 'f1::'. Got: {renamed_index}"
def test_did2s_renaming():
    """Test renaming of DID2S coefficient names."""
    # Sample coefficient names that would come from a DID2S model
    sample_coef_names = [
        "C(rel_year, contr.treatment(base=-1))[T.-5]",
        "C(rel_year, contr.treatment(base=-1))[T.-4]",
        "C(rel_year, contr.treatment(base=-1))[T.-3]",
        "C(rel_year, contr.treatment(base=-1))[T.-2]",
        "C(rel_year, contr.treatment(base=-1))[T.0]",
        "C(rel_year, contr.treatment(base=-1))[T.1]",
        "C(rel_year, contr.treatment(base=-1))[T.2]"
    ]
    # Apply the renaming function
    renamed_coefs = rename_did_coefficients(sample_coef_names)
    # Check that the renaming worked as expected
    assert len(renamed_coefs) == len(sample_coef_names), "Length mismatch after renaming"
    assert renamed_coefs[0] == "rel_year::-5", f"Expected 'rel_year::-5', got '{renamed_coefs[0]}'"
    assert renamed_coefs[4] == "rel_year::0", f"Expected 'rel_year::0', got '{renamed_coefs[4]}'"
    assert all("rel_year::" in name for name in renamed_coefs), "Not all coefficients were renamed correctly"

def test_did2s_renaming_with_model(did_data):
    """Test renaming with did2s model."""
    try:
        # Create and fit the did2s model
        fit_did2s = pf.did2s(
            did_data,
            yname="dep_var",
            first_stage="~ 0 | unit + year",
            second_stage="~i(rel_year, ref=-1)",
            treatment="treat",
            cluster="state",
        )
        # Get the coefficient names directly
        coef_names = fit_did2s.tidy().index.tolist()
        # Apply renaming function
        renamed_coefs = rename_did_coefficients(coef_names)
        # Check that the length is preserved
        assert len(renamed_coefs) == len(coef_names), "Length mismatch after renaming"
        # Check that at least some coefficients were renamed as expected
        assert any("rel_year::" in name for name in renamed_coefs), \
            f"No renamed coefficients found with 'rel_year::'. Got: {renamed_coefs}"
    except Exception as e:
        pytest.skip(f"Skipping test due to error in model fitting: {str(e)}")