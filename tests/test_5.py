import pytest
import pandas as pd
from definition_5e468c073ad14684aa395b57aed03dff import display_variant_scores_summary

# Mock DataFrame that `display_variant_scores_summary` is expected to operate on.
# In a real testing setup, this DataFrame (`synthetic_scores_df`) would typically
# be provided by a fixture or patched into the module where `display_variant_scores_summary` is defined.
# Here, it's defined globally so that `expected_result` DataFrames can be pre-calculated
# for the @pytest.mark.parametrize decorator.
_mock_synthetic_scores_df = pd.DataFrame({
    'variant_id': ['VAR001', 'VAR001', 'VAR001', 'VAR002', 'VAR002', 'VAR003', 'VAR003'],
    'modality': ['RNA-seq', 'RNA-seq', 'ATAC-seq', 'RNA-seq', 'ATAC-seq', 'ChIP-seq_histone', 'RNA-seq'],
    'tissue_cell_type': ['Liver', 'Brain', 'Liver', 'Lung', 'Kidney', 'Brain', 'Liver'],
    'quantile_score': [0.95, 0.10, 0.70, 0.88, 0.25, 0.55, 0.60],
    'log2fc_expression': [1.2, -0.5, 0.1, 0.9, -0.1, 0.0, 0.3]
})

# Helper function to generate expected DataFrames for comparison.
# This simulates the intended filtering logic of `display_variant_scores_summary`.
def _get_expected_filtered_df(variant_id: str, modality: str) -> pd.DataFrame:
    # This helper explicitly takes string types as expected by the function under test.
    filtered_df = _mock_synthetic_scores_df[
        (_mock_synthetic_scores_df['variant_id'] == variant_id) &
        (_mock_synthetic_scores_df['modality'] == modality)
    ]
    return filtered_df.reset_index(drop=True)

# Pre-calculate expected DataFrames based on the mock data and intended logic.
expected_df_var001_rna_seq = _get_expected_filtered_df('VAR001', 'RNA-seq')
expected_df_var001_atac_seq = _get_expected_filtered_df('VAR001', 'ATAC-seq')

# Create an empty DataFrame with the same columns and dtypes as the mock data.
# This is crucial for `pd.testing.assert_frame_equal` when no matches are found,
# as it expects column types to match.
_empty_df_template = pd.DataFrame(columns=_mock_synthetic_scores_df.columns)
expected_df_empty = _empty_df_template.astype(_mock_synthetic_scores_df.dtypes)


@pytest.mark.parametrize(
    "selected_variant_id, selected_modality, expected_result",
    [
        # Test 1: Expected functionality - valid variant and modality, multiple matches.
        # This covers a common usage scenario where a variant has scores across multiple tissue types for a modality.
        ('VAR001', 'RNA-seq', expected_df_var001_rna_seq),

        # Test 2: Expected functionality - valid variant and modality, single match.
        # Checks filtering for a specific, less numerous outcome to ensure precision.
        ('VAR001', 'ATAC-seq', expected_df_var001_atac_seq),

        # Test 3: Edge case - variant ID not found.
        # Ensures the function gracefully handles non-existent variants by returning an empty DataFrame.
        ('NONEXISTENT_VAR', 'RNA-seq', expected_df_empty),

        # Test 4: Edge case - modality not found.
        # Ensures the function handles non-existent modalities by returning an empty DataFrame.
        ('VAR002', 'NONEXISTENT_MODALITY', expected_df_empty),

        # Test 5: Edge case - invalid input type for selected_variant_id.
        # Verifies robust error handling for unexpected input types, expecting a TypeError.
        (123, 'RNA-seq', TypeError),

        # Note: If more test cases were allowed (max 5 in this request), an invalid type for
        # selected_modality like ('VAR001', None) could also be added for comprehensive type checking.
    ]
)
def test_display_variant_scores_summary(selected_variant_id, selected_modality, expected_result):
    try:
        # Assuming `display_variant_scores_summary` from definition_5e468c073ad14684aa395b57aed03dff
        # will internally access a DataFrame like `_mock_synthetic_scores_df`.
        # In a full test setup, this dependency would typically be managed
        # (e.g., via monkeypatching a module-level `synthetic_scores_df` to `_mock_synthetic_scores_df`).
        result = display_variant_scores_summary(selected_variant_id, selected_modality)

        # If an exception type is expected, the execution should jump to the `except` block.
        # If it reaches here, no exception was raised, so if an exception was expected, it's a failure.
        if isinstance(expected_result, type) and issubclass(expected_result, Exception):
            pytest.fail(f"Expected {expected_result.__name__} but no exception was raised.")
        else:
            # For DataFrame comparisons, use pandas testing utility for robust checks.
            pd.testing.assert_frame_equal(result, expected_result)
    except Exception as e:
        # If an exception was caught, check if it matches the expected exception type.
        if isinstance(expected_result, type) and issubclass(expected_result, Exception):
            assert isinstance(e, expected_result)
        else:
            # If an unexpected exception occurred, fail the test.
            pytest.fail(f"An unexpected exception {type(e).__name__} was raised: {e}")
