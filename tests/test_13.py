import pytest
import pandas as pd
import ipywidgets as widgets
from unittest.mock import MagicMock, patch # Required for mocking

# Placeholder for the module import
from definition_6160e044a57f4206a8021370b4684cab import make_interactive_plot_relationship

# Mock ipywidgets components and the plot_relationship function.
# These mock classes capture arguments passed to the widgets for later assertion.
class MockWidget:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    def __eq__(self, other): # Needed for comparing if mock_interactive returns a MockWidget
        return isinstance(other, self.__class__) and self.args == other.args and self.kwargs == other.kwargs

class MockDropdown(MockWidget):
    pass

class MockText(MockWidget):
    pass

# For cases where 'plot_relationship' needs to be mocked.
# It is assumed to be in the same module as make_interactive_plot_relationship
# and `ipywidgets` is a standard import for the notebook environment.
mock_plot_relationship_path = 'definition_6160e044a57f4206a8021370b4684cab.plot_relationship'
mock_ipywidgets_interactive_path = 'ipywidgets.interactive'
mock_ipywidgets_dropdown_path = 'ipywidgets.Dropdown'
mock_ipywidgets_text_path = 'ipywidgets.Text'


@pytest.mark.parametrize(
    "dataframe_input, expected_exception, expected_x_options, expected_y_options, expected_hue_options",
    [
        # Test Case 1: Valid DataFrame with mixed numeric/categorical columns
        (
            pd.DataFrame({
                'num_A': [1, 2, 3],
                'num_B': [4.0, 5.0, 6.0],
                'cat_C': ['X', 'Y', 'Z']
            }),
            None, # No exception expected
            ['num_A', 'num_B'], # Expected options for x_col (numeric columns)
            ['num_A', 'num_B'], # Expected options for y_col (numeric columns)
            ['num_A', 'num_B', 'cat_C', None], # Expected options for hue_col (all columns + None)
        ),
        # Test Case 2: Empty DataFrame
        (
            pd.DataFrame(),
            None,
            [], # No numeric columns
            [], # No numeric columns
            [None], # No columns at all, only None
        ),
        # Test Case 3: DataFrame with only non-numeric columns
        (
            pd.DataFrame({
                'str_D': ['a', 'b', 'c'],
                'bool_E': [True, False, True]
            }),
            None,
            [], # No numeric columns
            [], # No numeric columns
            ['str_D', 'bool_E', None], # All columns + None
        ),
        # Test Case 4: Invalid input (not a DataFrame) - expected AttributeError
        (
            None, # Input is None
            AttributeError,
            None, None, None # Not applicable for exceptions
        ),
        # Test Case 5: DataFrame with a single numeric column and one categorical
        (
            pd.DataFrame({
                'single_num': [100, 200, 300],
                'single_cat': ['P', 'Q', 'R']
            }),
            None,
            ['single_num'],
            ['single_num'],
            ['single_num', 'single_cat', None],
        ),
    ]
)
# Use patch decorators to mock dependencies for each test case
@patch(mock_ipywidgets_interactive_path)
@patch(mock_ipywidgets_dropdown_path, side_effect=MockDropdown)
@patch(mock_ipywidgets_text_path, side_effect=MockText)
@patch(mock_plot_relationship_path)
def test_make_interactive_plot_relationship(
    mock_plot_relationship, # Patched functions are passed as arguments (in reverse order of patch)
    mock_text,
    mock_dropdown,
    mock_interactive,
    dataframe_input, 
    expected_exception,
    expected_x_options, 
    expected_y_options, 
    expected_hue_options
):
    if expected_exception:
        with pytest.raises(expected_exception):
            make_interactive_plot_relationship(dataframe_input)
    else:
        result = make_interactive_plot_relationship(dataframe_input)
        
        # Assert that ipywidgets.interactive was called once
        mock_interactive.assert_called_once()
        
        # The first argument to interactive should be plot_relationship
        assert mock_interactive.call_args[0][0] == mock_plot_relationship

        # Check the keyword arguments (widgets) passed to interactive
        kwargs = mock_interactive.call_args[1]

        # Assert widgets are of the expected type (MockDropdown/MockText)
        assert isinstance(kwargs['x_col'], MockDropdown)
        assert isinstance(kwargs['y_col'], MockDropdown)
        assert isinstance(kwargs['hue_col'], MockDropdown)
        assert isinstance(kwargs['title'], MockText)
        assert isinstance(kwargs['x_label'], MockText)
        assert isinstance(kwargs['y_label'], MockText)

        # Assert dropdown options are correctly populated.
        # We sort them to ensure order independence and handle None values for sorting.
        key_func = lambda x: str(x) if x is not None else ''
        assert sorted(kwargs['x_col'].kwargs['options'], key=key_func) == sorted(expected_x_options, key=key_func)
        assert sorted(kwargs['y_col'].kwargs['options'], key=key_func) == sorted(expected_y_options, key=key_func)
        assert sorted(kwargs['hue_col'].kwargs['options'], key=key_func) == sorted(expected_hue_options, key=key_func)

        # Assert that the returned result is the return value of mock_interactive
        assert result == mock_interactive.return_value