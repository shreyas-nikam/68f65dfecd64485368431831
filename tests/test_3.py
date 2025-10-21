import pytest
import pandas as pd
import numpy as np
from definition_e01c188bbc164421bab1d6a55b1f1514 import validate_data

@pytest.mark.parametrize(
    "df, expected_cols, expected_dtypes, critical_na_cols, expected_output",
    [
        # Test Case 1: All valid - DataFrame conforms to all expectations.
        (
            pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie'], 'value': [10.1, 20.2, 30.3]}),
            ['id', 'name', 'value'],
            {'id': 'int64', 'name': 'object', 'value': 'float64'},
            ['id', 'value'],
            True
        ),
        # Test Case 2: Missing an expected column.
        (
            pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']}),
            ['id', 'name', 'value'], # 'value' column is missing
            {'id': 'int64', 'name': 'object', 'value': 'float64'},
            ['id', 'value'],
            False
        ),
        # Test Case 3: Incorrect data type for an expected column.
        (
            pd.DataFrame({'id': [1.0, 2.0, 3.0], 'name': ['Alice', 'Bob', 'Charlie'], 'value': [10.1, 20.2, 30.3]}),
            ['id', 'name', 'value'],
            {'id': 'int64', 'name': 'object', 'value': 'float64'}, # 'id' is float64 but int64 expected
            ['id', 'value'],
            False
        ),
        # Test Case 4: Missing values in a critical column.
        (
            pd.DataFrame({'id': [1, 2, np.nan], 'name': ['Alice', 'Bob', 'Charlie'], 'value': [10.1, 20.2, 30.3]}),
            ['id', 'name', 'value'],
            {'id': 'float64', 'name': 'object', 'value': 'float64'}, # id becomes float due to NaN
            ['id', 'value'], # 'id' has a NaN, so it fails critical NA check
            False
        ),
        # Test Case 5: Empty DataFrame that matches structure and dtypes, no NAs in critical columns.
        (
            pd.DataFrame({
                'id': pd.Series(dtype='int64'),
                'name': pd.Series(dtype='object'),
                'value': pd.Series(dtype='float64')
            }),
            ['id', 'name', 'value'],
            {'id': 'int64', 'name': 'object', 'value': 'float64'},
            ['id', 'value'], # Empty columns do not contain NaN values, thus pass the NA check
            True
        ),
    ]
)
def test_validate_data(df, expected_cols, expected_dtypes, critical_na_cols, expected_output):
    """
    Test the validate_data function for various scenarios including valid data,
    missing columns, incorrect dtypes, and missing values in critical columns.
    """
    result = validate_data(df, expected_cols, expected_dtypes, critical_na_cols)
    assert result == expected_output