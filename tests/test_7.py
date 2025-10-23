import pytest
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from definition_4a6ff9fcbd7d4691b89b66d8970328b8 import plot_time_series

# Fixture for a temporary file path for plot saving
@pytest.fixture
def temp_plot_path(tmp_path):
    path = tmp_path / "test_plot.png"
    yield path
    # Ensure cleanup after the test
    if os.path.exists(path):
        os.remove(path)

# Fixture for a basic DataFrame to use in tests
@pytest.fixture
def sample_dataframe():
    data = {
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
        'value_a': [10, 12, 15, 13],
        'value_b': [100, 110, 105, 120],
        'category': ['A', 'B', 'A', 'B']
    }
    return pd.DataFrame(data)

# Test 1: Successful Plot Generation and Saving to a file
def test_plot_time_series_success_and_save(sample_dataframe, temp_plot_path, monkeypatch):
    # Mock plt.show to prevent the plot from appearing during test execution
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    # Create a dummy figure and axes for matplotlib/seaborn to operate on
    # and direct current figure/axes to them using monkeypatch.
    # This allows plt.savefig to function on a real, albeit mocked, plot.
    fig, ax = plt.subplots()
    monkeypatch.setattr(plt, 'gca', lambda: ax) 
    monkeypatch.setattr(plt, 'gcf', lambda: fig)

    # Call the function with valid arguments and a save_path
    plot_time_series(
        dataframe=sample_dataframe,
        time_col='timestamp',
        value_col='value_a',
        title='Test Plot Title',
        x_label='Time Axis',
        y_label='Value Axis',
        save_path=temp_plot_path
    )
    
    # Assert that the plot file was created
    assert os.path.exists(temp_plot_path)
    plt.close(fig) # Close the dummy figure to prevent resource leaks

# Test 2: Handling an Empty DataFrame
def test_plot_time_series_empty_dataframe(monkeypatch):
    # Create an empty DataFrame with the expected column dtypes
    empty_df = pd.DataFrame({'timestamp': pd.Series(dtype='datetime64[ns]'), 'value_a': pd.Series(dtype='float')})
    
    # Mock all plotting functions to prevent actual display or file operations
    monkeypatch.setattr(plt, 'show', lambda: None)
    monkeypatch.setattr(plt, 'savefig', lambda *args, **kwargs: None)
    monkeypatch.setattr(sns, 'lineplot', lambda *args, **kwargs: None) # Mock seaborn's core function
    
    # The function should execute without raising an error, gracefully handling the empty data
    plot_time_series(
        dataframe=empty_df,
        time_col='timestamp',
        value_col='value_a',
        title='Empty Plot',
        x_label='Time',
        y_label='Value A'
    )
    # No assert is strictly needed here, as the test passes if no exception is raised

# Test 3: `time_col` is not datetime, but contains convertible string dates
def test_plot_time_series_string_time_col(sample_dataframe, monkeypatch):
    df_str_time = sample_dataframe.copy()
    # Convert the 'timestamp' column to string format to simulate non-datetime input
    df_str_time['timestamp'] = df_str_time['timestamp'].dt.strftime('%Y-%m-%d')
    
    # Mock plotting functions
    monkeypatch.setattr(plt, 'show', lambda: None)
    monkeypatch.setattr(plt, 'savefig', lambda *args, **kwargs: None)
    
    # Ensure a figure and axes are available for seaborn/matplotlib
    fig, ax = plt.subplots()
    monkeypatch.setattr(plt, 'gca', lambda: ax)
    monkeypatch.setattr(plt, 'gcf', lambda: fig)

    # The function's docstring implies it "ensures the time column is of datetime type",
    # so it should handle conversion of valid date strings gracefully without error.
    plot_time_series(
        dataframe=df_str_time,
        time_col='timestamp',
        value_col='value_a',
        title='String Time Col Plot',
        x_label='Date String',
        y_label='Value A'
    )
    # No assert for success, just that no exception was raised
    plt.close(fig)

# Test 4: Missing `time_col` or `value_col` in the DataFrame
def test_plot_time_series_missing_columns(sample_dataframe, monkeypatch):
    # Mock plotting functions as the error is expected before plotting occurs
    monkeypatch.setattr(plt, 'show', lambda: None)
    monkeypatch.setattr(plt, 'savefig', lambda *args, **kwargs: None)
    
    # Test case where the specified `time_col` does not exist in the DataFrame
    with pytest.raises(KeyError, match="'non_existent_time'"):
        plot_time_series(
            dataframe=sample_dataframe,
            time_col='non_existent_time', # This column does not exist
            value_col='value_a',
            title='Missing Time Col',
            x_label='Time',
            y_label='Value A'
        )

    # Test case where the specified `value_col` does not exist in the DataFrame
    with pytest.raises(KeyError, match="'non_existent_value'"):
        plot_time_series(
            dataframe=sample_dataframe,
            time_col='timestamp',
            value_col='non_existent_value', # This column does not exist
            title='Missing Value Col',
            x_label='Time',
            y_label='Value A'
        )

# Test 5: `value_col` contains non-numeric data
def test_plot_time_series_non_numeric_value_col(sample_dataframe, monkeypatch):
    df_non_numeric_value = sample_dataframe.copy()
    # Replace the numeric 'value_a' column with categorical (string) data
    df_non_numeric_value['value_a'] = df_non_numeric_value['category'] 
    
    # Mock plotting functions, but allow seaborn.lineplot to be called
    # so it can raise the expected type error.
    monkeypatch.setattr(plt, 'show', lambda: None)
    monkeypatch.setattr(plt, 'savefig', lambda *args, **kwargs: None)

    # Ensure a figure and axes are available for seaborn/matplotlib
    fig, ax = plt.subplots()
    monkeypatch.setattr(plt, 'gca', lambda: ax)
    monkeypatch.setattr(plt, 'gcf', lambda: fig)

    # seaborn.lineplot expects numeric data for the y-axis.
    # Passing a column with string values should result in a TypeError or ValueError
    # from pandas/seaborn when attempting to plot.
    with pytest.raises((TypeError, ValueError)): 
        plot_time_series(
            dataframe=df_non_numeric_value,
            time_col='timestamp',
            value_col='value_a', # This column now contains strings like 'A', 'B'
            title='Non-numeric Value Col',
            x_label='Time',
            y_label='Category as Value'
        )
    plt.close(fig)