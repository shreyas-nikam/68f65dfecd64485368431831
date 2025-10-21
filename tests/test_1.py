import pytest
import pandas as pd
from definition_a3f17c52c8734deba9b48e8697304a22 import simulate_variant_effect_scoring

# Define test cases using parametrize
# Each tuple represents: (variants_df, modalities_list, tissue_cell_types_list, expected_outcome)
# Expected_outcome can be:
#   - A dictionary for successful DataFrame outputs, detailing expected rows, columns, data types, and score ranges.
#   - An Exception type for cases where an error is expected.
test_cases = [
    (  # Test Case 1: Standard functionality - Multiple variants, modalities, and tissues
        pd.DataFrame({'variant_id': ['v1', 'v2', 'v3'], 'chromosome': ['chr1']*3}), # Added 'chromosome' to be more realistic for variants_df
        ['RNA-seq', 'ATAC-seq'],
        ['Liver', 'Brain'],
        {'rows': 12, # 3 variants * 2 modalities * 2 tissues
         'columns': ['variant_id', 'modality', 'tissue_cell_type', 'quantile_score', 'log2fc_expression'],
         'data_types': {'variant_id': object, 'modality': object, 'tissue_cell_type': object,
                        'quantile_score': float, 'log2fc_expression': float},
         'score_ranges': {'quantile_score': (0.0, 1.0), 'log2fc_expression': (-2.0, 2.0)}}
    ),
    (  # Test Case 2: Edge case - Empty variants_df
        pd.DataFrame({'variant_id': []}), # Empty DataFrame with 'variant_id' column defined
        ['RNA-seq'],
        ['Liver'],
        {'rows': 0,
         'columns': ['variant_id', 'modality', 'tissue_cell_type', 'quantile_score', 'log2fc_expression'],
         'data_types': {'variant_id': object, 'modality': object, 'tissue_cell_type': object,
                        'quantile_score': float, 'log2fc_expression': float},
         'score_ranges': {'quantile_score': (0.0, 1.0), 'log2fc_expression': (-2.0, 2.0)}}
    ),
    (  # Test Case 3: Edge case - Empty modalities_list
        pd.DataFrame({'variant_id': ['v1']}),
        [], # Empty modalities list
        ['Liver'],
        {'rows': 0,
         'columns': ['variant_id', 'modality', 'tissue_cell_type', 'quantile_score', 'log2fc_expression'],
         'data_types': {'variant_id': object, 'modality': object, 'tissue_cell_type': object,
                        'quantile_score': float, 'log2fc_expression': float},
         'score_ranges': {'quantile_score': (0.0, 1.0), 'log2fc_expression': (-2.0, 2.0)}}
    ),
    (  # Test Case 4: Invalid input - variants_df missing 'variant_id' column
        pd.DataFrame({'some_other_id': ['alt1']}), # Missing 'variant_id'
        ['RNA-seq'],
        ['Liver'],
        KeyError # Expecting KeyError as 'variant_id' is required for output
    ),
    (  # Test Case 5: Invalid input type - modalities_list is not a list
        pd.DataFrame({'variant_id': ['v1']}),
        "RNA-seq", # Invalid type: string instead of list
        ['Liver'],
        TypeError # Expecting TypeError for incorrect argument type
    ),
]

@pytest.mark.parametrize("variants_df, modalities_list, tissue_cell_types_list, expected_outcome", test_cases)
def test_simulate_variant_effect_scoring(variants_df, modalities_list, tissue_cell_types_list, expected_outcome):
    """
    Tests the simulate_variant_effect_scoring function for expected functionality,
    edge cases, and error handling.
    """
    if isinstance(expected_outcome, type) and issubclass(expected_outcome, Exception):
        # If an exception is expected, assert that it is raised
        with pytest.raises(expected_outcome):
            simulate_variant_effect_scoring(variants_df, modalities_list, tissue_cell_types_list)
    else:
        # If a DataFrame is expected, perform detailed assertions
        result_df = simulate_variant_effect_scoring(variants_df, modalities_list, tissue_cell_types_list)

        # 1. Assert the return type is a pandas DataFrame
        assert isinstance(result_df, pd.DataFrame), "Function should return a pandas DataFrame."

        # 2. Assert the DataFrame has the expected columns
        assert list(result_df.columns) == expected_outcome['columns'], "DataFrame columns do not match expected."

        # 3. Assert the number of rows matches the expected count
        assert len(result_df) == expected_outcome['rows'], "DataFrame row count does not match expected."

        if expected_outcome['rows'] > 0:
            # 4. Assert data types for non-empty DataFrames
            for col, expected_dtype in expected_outcome['data_types'].items():
                if expected_dtype == float:
                    # Check if the column's dtype is a float type (e.g., float64)
                    assert pd.api.types.is_float_dtype(result_df[col]), f"Column '{col}' has incorrect data type. Expected float."
                else:
                    # For object (string) types, check for object dtype
                    assert pd.api.types.is_object_dtype(result_df[col]), f"Column '{col}' has incorrect data type. Expected object."

            # 5. Assert quantile_score is within the range [0, 1]
            q_min, q_max = expected_outcome['score_ranges']['quantile_score']
            assert (result_df['quantile_score'] >= q_min).all() and \
                   (result_df['quantile_score'] <= q_max).all(), "quantile_score values are not within [0, 1]."

            # 6. Assert log2fc_expression is within the range [-2, 2]
            l_min, l_max = expected_outcome['score_ranges']['log2fc_expression']
            assert (result_df['log2fc_expression'] >= l_min).all() and \
                   (result_df['log2fc_expression'] <= l_max).all(), "log2fc_expression values are not within [-2, 2]."

            # 7. Ensure all input variants, modalities, and tissues are represented in the output
            assert set(result_df['variant_id'].unique()) == set(variants_df['variant_id']), \
                "Not all input variant_ids are represented in the output."
            assert set(result_df['modality'].unique()) == set(modalities_list), \
                "Not all input modalities are represented in the output."
            assert set(result_df['tissue_cell_type'].unique()) == set(tissue_cell_types_list), \
                "Not all input tissue_cell_types are represented in the output."