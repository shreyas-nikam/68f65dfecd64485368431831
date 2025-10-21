import pytest
import pandas as pd
import numpy as np
from definition_4b4c62ecf4a94ab382983eb14e1384ac import display_summary_statistics

@pytest.fixture
def sample_df_numeric():
    return pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [10.1, 20.2, 30.3, 40.4, 50.5],
        'col3': [100, 200, 300, 400, 500]
    })

@pytest.fixture
def sample_df_mixed():
    return pd.DataFrame({
        'numeric_col': [1, 2, 3, 4, 5],
        'string_col': ['a', 'b', 'c', 'd', 'e'],
        'bool_col': [True, False, True, False, True]
    })

@pytest.fixture
def sample_df_non_numeric():
    return pd.DataFrame({
        'string_col': ['a', 'b', 'c'],
        'bool_col': [True, False, True]
    })

def test_display_summary_statistics_numeric_df(sample_df_numeric):
    """
    Test with a DataFrame containing only numeric columns.
    Expected: A DataFrame with descriptive statistics for all numeric columns.
    """
    expected_output = sample_df_numeric.describe()
    result = display_summary_statistics(sample_df_numeric)
    pd.testing.assert_frame_equal(result, expected_output)

def test_display_summary_statistics_empty_df():
    """
    Test with an empty DataFrame.
    Expected: An empty DataFrame (as .describe() on an empty df returns an empty df).
    """
    empty_df = pd.DataFrame()
    expected_output = empty_df.describe() # This will be an 8x0 DataFrame by default
    result = display_summary_statistics(empty_df)
    pd.testing.assert_frame_equal(result, expected_output)

def test_display_summary_statistics_mixed_df(sample_df_mixed):
    """
    Test with a DataFrame containing both numeric and non-numeric columns.
    Expected: A DataFrame with descriptive statistics for only the numeric columns.
    """
    expected_output = sample_df_mixed.describe() # describe() automatically selects numeric
    result = display_summary_statistics(sample_df_mixed)
    pd.testing.assert_frame_equal(result, expected_output)

def test_display_summary_statistics_non_numeric_df(sample_df_non_numeric):
    """
    Test with a DataFrame containing only non-numeric columns.
    Expected: An empty DataFrame (as there are no numeric columns to describe).
    """
    expected_output = sample_df_non_numeric.describe() # This will be an 8x0 DataFrame
    result = display_summary_statistics(sample_df_non_numeric)
    pd.testing.assert_frame_equal(result, expected_output)

def test_display_summary_statistics_invalid_input_type():
    """
    Test with an input that is not a Pandas DataFrame.
    Expected: A TypeError.
    """
    with pytest.raises(AttributeError): # A common error if .describe() is called on non-DataFrame
        display_summary_statistics([1, 2, 3])
    with pytest.raises(AttributeError):
        display_summary_statistics("not a dataframe")
    with pytest.raises(AttributeError):
        display_summary_statistics(None)