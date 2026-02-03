import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import pyfixest as pf

# Assuming pf.dtable is imported from the module containing it


@pytest.fixture
def sample_df():
    """Fixture to create a sample dataframe for testing."""
    return pd.DataFrame(
        {
            "var1": [1, 2, 3, 4, 5],
            "var2": [5, 4, 3, 2, 1],
            "group": ["A", "A", "B", "B", "B"],
        }
    )


def standardize_dataframe(df):
    "Standardize dataframe column types before comparison."
    return df.applymap(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else str(x))


def test_minimal_input(sample_df):
    """Test with minimal input and check if it returns correct dataframe."""
    result = pf.dtable(sample_df, vars=["var1"], type="df")

    # Creating the expected DataFrame
    # Note: maketables formats count as "5.00" instead of "5"
    expected = pd.DataFrame(
        {
            "N": ["5.00"],  # maketables formats all values with decimals
            "Mean": ["3.00"],  # Format numbers to strings with 2 decimal places
            "Std. Dev.": ["1.58"],
        },
        index=["var1"],
    )

    # Ensure the expected dataframe's dtypes are object (string-like) for comparison
    expected = expected.astype("object")

    # Standardize the result and expected DataFrames for comparison
    result_standardized = standardize_dataframe(result)
    expected_standardized = standardize_dataframe(expected)

    assert_frame_equal(result_standardized, expected_standardized, check_dtype=False)


def test_multiple_stats(sample_df):
    result = pf.dtable(
        sample_df, vars=["var1"], stats=["count", "mean", "min", "max"], type="df"
    )

    expected = pd.DataFrame(
        {
            "N": ["5.00"],  # maketables formats all values with decimals
            "Mean": ["3.00"],
            "Min": ["1.00"],
            "Max": ["5.00"],
        },
        index=["var1"],
    )

    expected = expected.astype("object")

    result_standardized = standardize_dataframe(result)
    expected_standardized = standardize_dataframe(expected)

    assert_frame_equal(result_standardized, expected_standardized, check_dtype=False)


def test_group_by_column(sample_df):
    result = pf.dtable(
        sample_df, vars=["var1"], bycol=["group"], stats=["mean"], type="df"
    )

    expected = pd.DataFrame(
        {
            ("A", "Mean"): ["1.50"],
            ("B", "Mean"): ["4.00"],
        },
        index=["var1"],
    )

    expected.columns = pd.MultiIndex.from_tuples(
        expected.columns, names=["group", "Statistics"]
    )
    expected = expected.astype("object")

    # Ensure the result has the correct MultiIndex column names
    result.columns.names = ["group", "Statistics"]

    result_standardized = standardize_dataframe(result)
    expected_standardized = standardize_dataframe(expected)

    assert_frame_equal(result_standardized, expected_standardized, check_dtype=False)


def test_group_by_column_multiple_vars(sample_df):
    result = pf.dtable(
        sample_df, vars=["var1", "var2"], bycol=["group"], stats=["mean"], type="df"
    )

    expected = pd.DataFrame(
        {
            ("A", "Mean"): ["1.50", "4.50"],
            ("B", "Mean"): ["4.00", "2.00"],
        },
        index=["var1", "var2"],
    )

    expected.columns = pd.MultiIndex.from_tuples(
        expected.columns, names=["group", "Statistics"]
    )
    expected = expected.astype("object")

    # Ensure the result has the correct MultiIndex column names
    result.columns.names = ["group", "Statistics"]

    result_standardized = standardize_dataframe(result)
    expected_standardized = standardize_dataframe(expected)

    assert_frame_equal(result_standardized, expected_standardized, check_dtype=False)


def test_group_by_row(sample_df):
    result = pf.dtable(
        sample_df, vars=["var1"], byrow="group", stats=["mean", "std"], type="df"
    )

    expected = pd.DataFrame(
        {
            "Mean": ["1.50", "4.00"],
            "Std. Dev.": ["0.71", "1.00"],
        },
        index=pd.MultiIndex.from_tuples(
            [("A", "var1"), ("B", "var1")], names=["group", "var"]
        ),
    )

    expected = expected.astype("object")

    # Ensure the result has the correct MultiIndex level names
    result.index.names = ["group", "var"]

    result_standardized = standardize_dataframe(result)
    expected_standardized = standardize_dataframe(expected)

    assert_frame_equal(result_standardized, expected_standardized, check_dtype=False)


def test_group_by_row_multiple_vars(sample_df):
    result = pf.dtable(
        sample_df,
        vars=["var1", "var2"],
        byrow="group",
        stats=["mean", "std"],
        type="df",
    )

    expected = pd.DataFrame(
        {
            "Mean": ["1.50", "4.00", "4.50", "2.00"],
            "Std. Dev.": ["0.71", "1.00", "0.71", "1.00"],
        },
        index=pd.MultiIndex.from_tuples(
            [("A", "var1"), ("B", "var1"), ("A", "var2"), ("B", "var2")],
            names=["group", "var"],
        ),
    )

    expected = expected.astype("object")

    # Ensure the result has the correct MultiIndex level names
    result.index.names = ["group", "var"]

    # Reorder the result DataFrame to match the expected MultiIndex order
    result = result.reindex(expected.index)

    result_standardized = standardize_dataframe(result)
    expected_standardized = standardize_dataframe(expected)

    assert_frame_equal(result_standardized, expected_standardized, check_dtype=False)


def test_custom_labels(sample_df):
    """Test with custom labels for variables and stats."""
    labels = {"var1": "Variable 1", "var2": "Variable 2"}
    stats_labels = {"mean": "Average", "std": "Standard Deviation"}

    result = pf.dtable(
        sample_df,
        vars=["var1"],
        stats=["mean", "std"],
        labels=labels,
        stats_labels=stats_labels,
        type="df",
    )

    expected = pd.DataFrame(
        {
            "Average": [3.0],
            "Standard Deviation": [1.58],
        },
        index=["Variable 1"],
    ).round(2)

    result_standardized = standardize_dataframe(result)
    expected_standardized = standardize_dataframe(expected)

    assert_frame_equal(result_standardized, expected_standardized, check_dtype=False)


def test_counts_row_below(sample_df):
    result = pf.dtable(
        sample_df,
        vars=["var1"],
        stats=["mean", "count"],
        counts_row_below=True,
        type="df",
    )

    expected = pd.DataFrame(
        {
            "Mean": ["3.00", "5.00"],  # maketables formats count as "5.00"
        },
        index=pd.MultiIndex.from_tuples(
            [("stats", "var1"), ("nobs", "N")], names=["stats", "count"]
        ),
    )

    expected = expected.astype("object")

    # Ensure the result has the correct MultiIndex level names
    result.index.names = ["stats", "count"]

    result_standardized = standardize_dataframe(result)
    expected_standardized = standardize_dataframe(expected)

    assert_frame_equal(result_standardized, expected_standardized, check_dtype=False)


def test_counts_row_below_equal(sample_df):
    result = pf.dtable(
        sample_df,
        vars=["var1", "var2"],
        bycol=["group"],
        stats=["mean", "count"],
        counts_row_below=True,
        type="df",
    )

    expected = pd.DataFrame(
        {
            ("A", "Mean"): ["1.50", "4.50", "2.00"],  # Ensure '2.00' for consistency
            ("B", "Mean"): ["4.00", "2.00", "3.00"],  # Ensure '3.00' for consistency
        },
        index=pd.MultiIndex.from_tuples(
            [("stats", "var1"), ("stats", "var2"), ("nobs", "N")],
            names=["stats", "var"],
        ),
    )

    expected.columns = pd.MultiIndex.from_tuples(
        expected.columns, names=["group", "Statistics"]
    )
    expected = expected.astype("object")

    # Ensure the result has the correct MultiIndex row names
    result.index.names = ["stats", "var"]

    # Ensure the result has the correct MultiIndex column names
    result.columns.names = ["group", "Statistics"]

    # Standardize the result and expected DataFrames for comparison
    result_standardized = standardize_dataframe(result)
    expected_standardized = standardize_dataframe(expected)

    assert_frame_equal(result_standardized, expected_standardized, check_dtype=False)


def test_two_bycol_groups(sample_df):
    # Create an additional column to group by
    sample_df["group2"] = ["X", "X", "Y", "Y", "Y"]

    result = pf.dtable(
        sample_df,
        vars=["var1", "var2"],
        bycol=["group", "group2"],
        stats=["mean", "count"],
        type="df",
    )

    expected = pd.DataFrame(
        {
            ("A", "X", "Mean"): ["1.50", "4.50"],
            ("A", "X", "N"): [2, 2],
            ("B", "Y", "Mean"): ["4.00", "2.00"],
            ("B", "Y", "N"): [3, 3],
        },
        index=["var1", "var2"],
    )

    expected.columns = pd.MultiIndex.from_tuples(
        expected.columns, names=["group", "group2", "Statistics"]
    )
    expected = expected.astype(
        {
            "A": "object",
            "B": "object",
            ("A", "X", "N"): "int64",
            ("B", "Y", "N"): "int64",
        }
    )

    # Ensure the result has the correct MultiIndex column names
    result.columns.names = ["group", "group2", "Statistics"]

    # Standardize the result and expected DataFrames for comparison
    result_standardized = standardize_dataframe(result)
    expected_standardized = standardize_dataframe(expected)

    assert_frame_equal(result_standardized, expected_standardized, check_dtype=False)


def test_invalid_dataframe():
    """Test with an invalid dataframe and ensure it raises an error."""
    with pytest.raises(AssertionError, match=r"df must be a pandas DataFrame\."):
        pf.dtable("not_a_dataframe", vars=["var1"])


def test_non_numeric_column(sample_df):
    """Test with non-numeric column and ensure it raises an error."""
    with pytest.raises(AssertionError, match=r"Variables must be numerical\."):
        pf.dtable(sample_df, vars=["group"])


def test_invalid_byrow(sample_df):
    """Test with an invalid `byrow` column that doesn't exist."""
    with pytest.raises(
        AssertionError, match=r"byrow must be a column in the DataFrame\."
    ):
        pf.dtable(sample_df, vars=["var1"], byrow="non_existent_column")


def test_invalid_bycol(sample_df):
    """Test with an invalid `bycol` column that doesn't exist."""
    with pytest.raises(
        AssertionError, match=r"bycol must be a list of columns in the DataFrame\."
    ):
        pf.dtable(sample_df, vars=["var1"], bycol=["non_existent_column"])


def test_dtable_deprecation_warning(sample_df):
    """Test that dtable() emits a deprecation warning."""
    with pytest.warns(FutureWarning, match=r"pf\.dtable\(\) is deprecated"):
        pf.dtable(sample_df, vars=["var1"], type="df")
