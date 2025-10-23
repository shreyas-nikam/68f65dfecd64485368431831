import pytest
import pandas as pd
import numpy as np
import ipywidgets as widgets
from unittest.mock import patch

# definition_d6cacdb7759c40cda31b9a93113110f6 block
# DO NOT REPLACE or REMOVE the block
from definition_d6cacdb7759c40cda31b9a93113110f6 import make_interactive_plot_time_series
# END definition_d6cacdb7759c40cda31b9a93113110f6 block

# Mock the plot_time_series function that make_interactive_plot_time_series wraps.
# We don't want to actually generate a plot during tests, just check widget setup.
# The mock needs to be available at the path where make_interactive_plot_time_series imports it.
# Assuming plot_time_series is in the same module as make_interactive_plot_time_series.
@pytest.fixture(autouse=True)
def mock_plot_time_series():
    with patch('definition_d6cacdb7759c40cda31b9a93113110f6.plot_time_series') as mock_func:
        yield mock_func

# Helper to create a basic dataframe
@pytest.fixture
def sample_dataframe():
    data = {
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'numeric_feature_A': [10, 20, 15],
        'numeric_feature_B': [1.1, 2.2, 3.3],
        'categorical_feature': ['A', 'B', 'A']
    }
    return pd.DataFrame(data)

# Test Cases

def test_make_interactive_plot_time_series_basic_functionality(sample_dataframe):
    """
    Test that the function returns an ipywidgets.interactive object and
    configures basic widgets correctly for a valid DataFrame with numeric columns.
    """
    interactive_widget = make_interactive_plot_time_series(sample_dataframe)

    assert isinstance(interactive_widget, widgets.interactive)
    
    # Check if the returned object has the expected controls/widget names
    assert 'value_col' in interactive_widget.children_names
    assert 'title' in interactive_widget.children_names
    assert 'x_label' in interactive_widget.children_names
    assert 'y_label' in interactive_widget.children_names

    # Check the value_col dropdown options
    value_col_widget = interactive_widget.children[interactive_widget.children_names.index('value_col')]
    assert isinstance(value_col_widget, widgets.Dropdown)
    expected_numeric_cols = ['numeric_feature_A', 'numeric_feature_B']
    assert set(value_col_widget.options) == set(expected_numeric_cols)
    assert value_col_widget.value in expected_numeric_cols # Default value should be one of the numeric columns

    # Check text widget types and default values (assuming they are set by the function)
    title_widget = interactive_widget.children[interactive_widget.children_names.index('title')]
    assert isinstance(title_widget, widgets.Text)
    assert title_widget.value == 'Time Series Plot' # Example default

def test_make_interactive_plot_time_series_no_numeric_columns():
    """
    Test with a DataFrame that has a 'timestamp' column but no numeric columns for 'value_col' selection.
    The 'value_col' dropdown should have no options, and its value should be None.
    """
    df_no_numeric = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02']),
        'categorical_feature': ['X', 'Y'],
        'another_string': ['foo', 'bar']
    })
    
    interactive_widget = make_interactive_plot_time_series(df_no_numeric)

    assert isinstance(interactive_widget, widgets.interactive)
    value_col_widget = interactive_widget.children[interactive_widget.children_names.index('value_col')]
    assert isinstance(value_col_widget, widgets.Dropdown)
    assert not value_col_widget.options # Expecting no options
    assert value_col_widget.value is None # Expecting default value to be None if no options

def test_make_interactive_plot_time_series_empty_dataframe():
    """
    Test with an empty DataFrame (but with column definitions and correct dtypes).
    Widgets should still be created, and 'value_col' options should reflect numeric columns
    based on their dtypes, even if there's no data.
    """
    empty_df = pd.DataFrame(columns=[
        'timestamp', 'numeric_feature_A', 'numeric_feature_B', 'categorical_feature'
    ]).astype({
        'timestamp': 'datetime64[ns]',
        'numeric_feature_A': float,
        'numeric_feature_B': float,
        'categorical_feature': str
    })
    
    interactive_widget = make_interactive_plot_time_series(empty_df)

    assert isinstance(interactive_widget, widgets.interactive)
    value_col_widget = interactive_widget.children[interactive_widget.children_names.index('value_col')]
    assert isinstance(value_col_widget, widgets.Dropdown)
    # Even if empty, it should still identify numeric columns from dtype
    expected_numeric_cols = ['numeric_feature_A', 'numeric_feature_B']
    assert set(value_col_widget.options) == set(expected_numeric_cols)
    assert value_col_widget.value in expected_numeric_cols # Default value should be from expected options


def test_make_interactive_plot_time_series_missing_timestamp_column():
    """
    Test with a DataFrame that is missing the expected 'timestamp' column.
    This should raise a KeyError, as the underlying plot_time_series function (or the wrapper itself)
    likely assumes its presence for time series plotting.
    """
    df_no_timestamp = pd.DataFrame({
        'numeric_feature_A': [1, 2, 3],
        'numeric_feature_B': [4, 5, 6],
        'categorical_feature': ['X', 'Y', 'Z']
    })
    
    # Expecting a KeyError if the function tries to access df['timestamp']
    with pytest.raises(KeyError, match="timestamp"): 
        make_interactive_plot_time_series(df_no_timestamp)

def test_make_interactive_plot_time_series_invalid_input_type():
    """
    Test with various invalid input types (not a pandas DataFrame).
    Should raise a TypeError or AttributeError when DataFrame methods are called.
    """
    invalid_inputs = ["not a dataframe", None, [1, 2, 3], 123]
    for invalid_input in invalid_inputs:
        with pytest.raises((TypeError, AttributeError), match="DataFrame"):
            make_interactive_plot_time_series(invalid_input)
