import pytest
import pandas as pd
from definition_34919d322a1e42efa931601cf141b8a1 import simulate_tool_execution

# Helper function to create a pandas DataFrame for test inputs
def create_test_dataframe(data_list, columns):
    return pd.DataFrame(data_list, columns=columns)

# Define common columns for the synthetic data
TEST_COLUMNS = ['gene_expression_level', 'tissue_type', 'gene_name']

@pytest.mark.parametrize(
    "data_input, tool_parameters_input, expected_output",
    [
        # Test Case 1: Successful filtering with all relevant parameters
        (
            create_test_dataframe(
                [[0.1, 'liver', 'SORT1'], [0.02, 'brain', 'BRCA1'], [0.06, 'liver', 'SORT1'], [0.15, 'kidney', 'SORT1']],
                TEST_COLUMNS
            ),
            {
                'gene_name': {'extracted_value': 'SORT1', 'mapped_successfully': True},
                'tissue_type': {'extracted_value': 'liver', 'mapped_successfully': True},
                'min_expression_level': {'extracted_value': 0.05, 'mapped_successfully': True}
            },
            {
                'simulated_results': {
                    'filtered_dataframe_head': [{'gene_expression_level': 0.06, 'tissue_type': 'liver', 'gene_name': 'SORT1'}],
                    'number_of_filtered_rows': 1
                }
            }
        ),
        # Test Case 2: No relevant parameters in tool_parameters (should return original data)
        (
            create_test_dataframe(
                [[0.1, 'liver', 'SORT1'], [0.02, 'brain', 'BRCA1'], [0.06, 'liver', 'SORT1']],
                TEST_COLUMNS
            ),
            {
                'some_other_param': {'extracted_value': 'value', 'mapped_successfully': True}
            },
            {
                'simulated_results': {
                    'filtered_dataframe_head': [
                        {'gene_expression_level': 0.1, 'tissue_type': 'liver', 'gene_name': 'SORT1'},
                        {'gene_expression_level': 0.02, 'tissue_type': 'brain', 'gene_name': 'BRCA1'},
                        {'gene_expression_level': 0.06, 'tissue_type': 'liver', 'gene_name': 'SORT1'}
                    ],
                    'number_of_filtered_rows': 3
                }
            }
        ),
        # Test Case 3: Filtering results in an empty DataFrame (no matching rows)
        (
            create_test_dataframe(
                [[0.1, 'liver', 'SORT1'], [0.02, 'brain', 'BRCA1']],
                TEST_COLUMNS
            ),
            {
                'gene_name': {'extracted_value': 'NONEXISTENT', 'mapped_successfully': True}
            },
            {
                'simulated_results': {
                    'filtered_dataframe_head': [],
                    'number_of_filtered_rows': 0
                }
            }
        ),
        # Test Case 4: Invalid 'data' input type (not a pandas.DataFrame)
        (
            [1, 2, 3],  # list instead of DataFrame
            {
                'gene_name': {'extracted_value': 'SORT1', 'mapped_successfully': True}
            },
            TypeError  # Expected exception
        ),
        # Test Case 5: Invalid 'tool_parameters' input type (not a dictionary)
        (
            create_test_dataframe(
                [[0.1, 'liver', 'SORT1']],
                TEST_COLUMNS
            ),
            None,  # None instead of dict
            TypeError  # Expected exception
        ),
    ]
)
def test_simulate_tool_execution(data_input, tool_parameters_input, expected_output):
    try:
        result = simulate_tool_execution(data_input, tool_parameters_input)
        # If no exception was expected, assert the result
        assert not isinstance(expected_output, type) # Check if expected_output is not an exception class
        assert result == expected_output
    except Exception as e:
        # If an exception was expected, assert its type
        assert isinstance(e, expected_output)