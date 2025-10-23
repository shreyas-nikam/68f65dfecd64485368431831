import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO
import sys

# BLOCK START: your_module
from definition_a99f7175b1e64e3b93d019b37bcdef93 import validate_and_summarize_data
# BLOCK END: your_module


# Helper function to create a basic valid DataFrame
def create_valid_df():
    data = {
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'entity_id': [1, 2, 3],
        'numeric_feature_A': [10.1, 20.2, 30.3],
        'numeric_feature_B': [1.1, 2.2, 3.3],
        'categorical_feature': ['cat_A', 'cat_B', 'cat_A'],
        'gene_expression_level': [0.5, 0.7, 0.6],
        'tissue_type': ['liver', 'muscle', 'brain']
    }
    return pd.DataFrame(data)

# Test cases for validate_and_summarize_data
@pytest.mark.parametrize("dataframe, expected_exception, expected_output_substring", [
    # Test Case 1: Valid DataFrame - Expected functionality, check summary output
    (
        create_valid_df(),
        None, # No exception expected
        "mean" # Expect 'mean' in the describe() output for numeric summary
    ),
    # Test Case 2: Missing required column ('entity_id') - Validation failure
    (
        pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'numeric_feature_A': [10.1, 20.2],
            'numeric_feature_B': [1.1, 2.2],
            'categorical_feature': ['cat_A', 'cat_B'],
            'gene_expression_level': [0.5, 0.7],
            'tissue_type': ['liver', 'muscle']
        }),
        ValueError, # Expected to raise ValueError for missing column
        "missing required columns: entity_id" # Expected error message substring
    ),
    # Test Case 3: Duplicate 'entity_id' - Primary key uniqueness validation failure
    (
        pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'entity_id': [1, 2, 1], # Duplicate entity_id
            'numeric_feature_A': [10.1, 20.2, 30.3],
            'numeric_feature_B': [1.1, 2.2, 3.3],
            'categorical_feature': ['cat_A', 'cat_B', 'cat_A'],
            'gene_expression_level': [0.5, 0.7, 0.6],
            'tissue_type': ['liver', 'muscle', 'brain']
        }),
        ValueError, # Expected to raise ValueError for non-unique entity_id
        "'entity_id' contains duplicate values" # Expected error message substring
    ),
    # Test Case 4: Missing values in a critical column ('numeric_feature_A') - Data quality validation failure
    (
        pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'entity_id': [1, 2, 3],
            'numeric_feature_A': [10.1, np.nan, 30.3], # NaN in critical column
            'numeric_feature_B': [1.1, 2.2, 3.3],
            'categorical_feature': ['cat_A', 'cat_B', 'cat_A'],
            'gene_expression_level': [0.5, 0.7, 0.6],
            'tissue_type': ['liver', 'muscle', 'brain']
        }),
        ValueError, # Expected to raise ValueError for missing critical values
        "'numeric_feature_A' contains missing values (NaN)" # Expected error message substring
    ),
    # Test Case 5: Invalid input type (not a DataFrame) - Edge case for input type
    (
        "this is not a dataframe", # Not a pandas DataFrame
        TypeError, # Expected to raise TypeError
        "Input must be a pandas DataFrame" # Expected error message substring
    ),
])
def test_validate_and_summarize_data(dataframe, expected_exception, expected_output_substring, capsys):
    if expected_exception:
        # If an exception is expected, assert that it is raised with the correct message
        with pytest.raises(expected_exception) as excinfo:
            validate_and_summarize_data(dataframe)
        assert expected_output_substring in str(excinfo.value)
    else:
        # If no exception is expected, call the function and capture its stdout
        validate_and_summarize_data(dataframe)
        captured = capsys.readouterr()
        # Assert that the expected substring (e.g., 'mean' from describe()) is in stdout
        assert expected_output_substring in captured.out
        # Assert that there's no error output on stderr for successful cases
        assert captured.err == ""