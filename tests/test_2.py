import pytest
import pandas as pd
from definition_e1af15771de2428abb6a2f2712421445 import simulate_visualization_data

# Test Case 1: Standard valid inputs - ensuring correct DataFrame structure and numeric types
@pytest.mark.parametrize(
    "variant_id, modality, tissue, width, shift",
    [
        ("VAR_001", "RNA-seq", "Liver", 1000, 0),
        ("VAR_002", "ATAC-seq", "Brain", 5000, -500),
        ("VAR_003", "ChIP-seq_histone", "Lung", 100, 10),
    ]
)
def test_simulate_visualization_data_valid_inputs(variant_id, modality, tissue, width, shift):
    df = simulate_visualization_data(variant_id, modality, tissue, width, shift)
    assert isinstance(df, pd.DataFrame)
    assert "genomic_position_relative" in df.columns
    assert "predicted_effect_value" in df.columns
    assert pd.api.types.is_numeric_dtype(df["genomic_position_relative"])
    assert pd.api.types.is_numeric_dtype(df["predicted_effect_value"])
    assert len(df) > 0, "DataFrame should not be empty for a positive interval width"

# Test Case 2: Edge case - Zero plot_interval_width
# A region with zero width logically contains no points for a 'trend'. Expecting an empty DataFrame.
def test_simulate_visualization_data_zero_width():
    df = simulate_visualization_data("VAR_004", "RNA-seq", "Muscle", 0, 0)
    assert isinstance(df, pd.DataFrame)
    assert "genomic_position_relative" in df.columns # Columns should still exist
    assert "predicted_effect_value" in df.columns
    assert len(df) == 0, "DataFrame should be empty for zero plot_interval_width"

# Test Case 3: Edge case - Negative plot_interval_width
# Width must be non-negative. Expecting a ValueError.
def test_simulate_visualization_data_negative_width():
    with pytest.raises(ValueError, match="plot_interval_width must be non-negative"):
        simulate_visualization_data("VAR_005", "RNA-seq", "Lung", -100, 0)

# Test Case 4: Invalid types for plot_interval_width and plot_interval_shift
# Expecting TypeError for non-numeric or incorrect numeric types (e.g., string instead of int).
@pytest.mark.parametrize(
    "variant_id, modality, tissue, width, shift, expected_error_type",
    [
        ("VAR_006", "RNA-seq", "Kidney", "1000", 0, TypeError),  # String width
        ("VAR_007", "RNA-seq", "Kidney", 1000, "0", TypeError),    # String shift
        ("VAR_008", "RNA-seq", "Kidney", [1000], 0, TypeError),   # List width
        ("VAR_009", "RNA-seq", "Kidney", 1000, None, TypeError), # None shift
    ]
)
def test_simulate_visualization_data_invalid_numeric_param_types(variant_id, modality, tissue, width, shift, expected_error_type):
    with pytest.raises(expected_error_type):
        simulate_visualization_data(variant_id, modality, tissue, width, shift)

# Test Case 5: Invalid types for selected_variant_id, selected_modality, and selected_tissue
# Expecting TypeError for non-string types where string is expected.
@pytest.mark.parametrize(
    "variant_id, modality, tissue, width, shift, expected_error_type",
    [
        (123, "RNA-seq", "Liver", 1000, 0, TypeError),        # Int variant_id
        ("VAR_010", 123, "Liver", 1000, 0, TypeError),         # Int modality
        ("VAR_011", "RNA-seq", ["Liver"], 1000, 0, TypeError), # List tissue
        (None, "RNA-seq", "Liver", 1000, 0, TypeError),       # None variant_id
    ]
)
def test_simulate_visualization_data_invalid_string_param_types(variant_id, modality, tissue, width, shift, expected_error_type):
    with pytest.raises(expected_error_type):
        simulate_visualization_data(variant_id, modality, tissue, width, shift)
