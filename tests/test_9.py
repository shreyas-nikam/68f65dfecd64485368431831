import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import os
import matplotlib.pyplot as plt

# The placeholder for your module import
# DO NOT REPLACE or REMOVE this block
# definition_4f0925b2fde848468a5931d773b89072
from your_module import plot_categorical_comparison 
# </your_module>

@pytest.fixture
def sample_dataframe():
    """Returns a sample pandas DataFrame for testing."""
    data = {
        'category_col': ['Group1', 'Group2', 'Group1', 'Group3', 'Group2', 'Group3', 'Group1', 'Group2'],
        'value_col': [10, 20, 15, 25, 30, 35, 12, 22],
        'non_numeric_col': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B'] # For testing non-numeric value_col if needed
    }
    return pd.DataFrame(data)

# Test Case 1: Basic functionality with default aggregation ('mean') and display (no save_path).
@patch('seaborn.barplot')
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.figure') # Mock figure creation to prevent actual figure opening
def test_plot_categorical_comparison_basic_mean_display(mock_figure, mock_savefig, mock_show, mock_barplot, sample_dataframe):
    df = sample_dataframe
    plot_categorical_comparison(
        dataframe=df,
        category_col='category_col',
        value_col='value_col',
        aggregation_func='mean',
        title='Test Title Mean',
        x_label='Test X Mean',
        y_label='Test Y Mean',
        color_palette='viridis',
        save_path=None
    )
    mock_barplot.assert_called_once()
    # Check that seaborn.barplot was called with appropriate data and arguments
    args, kwargs = mock_barplot.call_args
    assert kwargs.get('x') == 'category_col'
    # Assuming the aggregated value column is named 'aggregated_value' internally
    assert kwargs.get('y') == 'aggregated_value' 
    assert 'data' in kwargs
    assert kwargs.get('palette') == 'viridis'
    
    mock_show.assert_called_once() # Should display the plot
    mock_savefig.assert_not_called() # Should not save the plot

# Test Case 2: Custom aggregation ('sum') and saving the plot to a file.
@patch('seaborn.barplot')
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.figure')
def test_plot_categorical_comparison_custom_sum_and_save(mock_figure, mock_savefig, mock_show, mock_barplot, sample_dataframe):
    df = sample_dataframe
    save_path = 'test_comparison_sum_plot.png'
    
    plot_categorical_comparison(
        dataframe=df,
        category_col='category_col',
        value_col='value_col',
        aggregation_func='sum',
        title='Test Title Sum',
        x_label='Test X Sum',
        y_label='Test Y Sum',
        color_palette='magma',
        save_path=save_path
    )
    
    mock_barplot.assert_called_once()
    # Check that savefig was called with the correct path
    mock_savefig.assert_called_once_with(save_path, bbox_inches='tight') 
    mock_show.assert_not_called() # Should not display if saved
    
    # Ensure the file is not actually created by the mock
    assert not os.path.exists(save_path)

# Test Case 3: Invalid column names (`category_col` or `value_col` not found).
@pytest.mark.parametrize("category_col, value_col, expected_exception", [
    ('non_existent_category', 'value_col', KeyError),
    ('category_col', 'non_existent_value', KeyError),
])
@patch('seaborn.barplot')
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.figure')
def test_plot_categorical_comparison_invalid_columns(mock_figure, mock_savefig, mock_show, mock_barplot, sample_dataframe, category_col, value_col, expected_exception):
    with pytest.raises(expected_exception):
        plot_categorical_comparison(
            dataframe=sample_dataframe,
            category_col=category_col,
            value_col=value_col,
            aggregation_func='mean',
            title='Test Title',
            x_label='Test X',
            y_label='Test Y',
            color_palette='viridis',
            save_path=None
        )
    mock_barplot.assert_not_called() # Plotting should not occur if columns are invalid

# Test Case 4: Invalid aggregation function string.
@patch('seaborn.barplot')
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.figure')
def test_plot_categorical_comparison_invalid_aggregation_func(mock_figure, mock_savefig, mock_show, mock_barplot, sample_dataframe):
    # Pandas .agg() raises ValueError for unrecognized strings
    with pytest.raises(ValueError, match="Invalid aggregation function"): 
        plot_categorical_comparison(
            dataframe=sample_dataframe,
            category_col='category_col',
            value_col='value_col',
            aggregation_func='unknown_func', # An unrecognized aggregation function
            title='Test Title',
            x_label='Test X',
            y_label='Test Y',
            color_palette='viridis',
            save_path=None
        )
    mock_barplot.assert_not_called() # Plotting should not occur

# Test Case 5: Empty DataFrame.
@patch('seaborn.barplot')
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.figure')
def test_plot_categorical_comparison_empty_dataframe(mock_figure, mock_savefig, mock_show, mock_barplot):
    empty_df = pd.DataFrame(columns=['category_col', 'value_col'], dtype=float) # Ensure columns exist to avoid KeyError initially
    
    # An empty DataFrame should be handled gracefully by the function and seaborn
    plot_categorical_comparison(
        dataframe=empty_df,
        category_col='category_col',
        value_col='value_col',
        aggregation_func='mean',
        title='Empty Data Test',
        x_label='Categories',
        y_label='Values',
        color_palette='viridis',
        save_path=None
    )
    
    mock_barplot.assert_called_once()
    args, kwargs = mock_barplot.call_args
    # Verify that the 'data' passed to barplot is empty after aggregation
    # Groupby on empty DF with mean() will result in an empty DataFrame/Series.
    assert kwargs.get('data').empty
    mock_show.assert_called_once()
    mock_savefig.assert_not_called()