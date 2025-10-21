import pytest
import pandas as pd
import random # The function will likely use this module internally.

# ---
# Placeholder for the module import. DO NOT REPLACE or REMOVE this block.
from definition_a68f2378b2804ebc970e6b9cec6b7828 import generate_synthetic_genetic_variants
# ---

@pytest.mark.parametrize(
    "input_args, expected",
    [
        # Test Case 1: Standard generation - multiple variants, diverse inputs, check DataFrame structure and ranges.
        (
            {
                "num_variants": 3,
                "chromosomes_list": ["chr1", "chr2", "chrX"],
                "min_pos": 1_000_000,
                "max_pos": 10_000_000,
                "alleles_list": ["A", "T", "G", "C"],
                "organisms_list": ["human", "mouse"],
            },
            {
                "result_type": "dataframe",
                "shape_rows": 3,
                "columns": ["variant_id", "chromosome", "position", "reference_allele", "alternate_allele", "organism"],
                "position_range": (1_000_000, 10_000_000),
            },
        ),
        # Test Case 2: Zero variants - should return an empty DataFrame with correct columns.
        (
            {
                "num_variants": 0,
                "chromosomes_list": ["chr1"],
                "min_pos": 1,
                "max_pos": 10,
                "alleles_list": ["A"],
                "organisms_list": ["human"],
            },
            {
                "result_type": "dataframe",
                "shape_rows": 0,
                "columns": ["variant_id", "chromosome", "position", "reference_allele", "alternate_allele", "organism"],
            },
        ),
        # Test Case 3: Single option for all lists - verify generated content matches the single option.
        (
            {
                "num_variants": 2,
                "chromosomes_list": ["chrY"],
                "min_pos": 500,
                "max_pos": 1000,
                "alleles_list": ["G"],
                "organisms_list": ["dog"],
            },
            {
                "result_type": "dataframe",
                "shape_rows": 2,
                "columns": ["variant_id", "chromosome", "position", "reference_allele", "alternate_allele", "organism"],
                "expected_values": {
                    "chromosome": "chrY",
                    "reference_allele": "G",
                    "alternate_allele": "G", 
                    "organism": "dog",
                },
            },
        ),
        # Test Case 4: Invalid position range (min_pos > max_pos) - should raise a ValueError.
        (
            {
                "num_variants": 1,
                "chromosomes_list": ["chr1"],
                "min_pos": 100,
                "max_pos": 99, # Invalid range
                "alleles_list": ["A"],
                "organisms_list": ["human"],
            },
            ValueError, # random.randint(min, max) raises ValueError if min > max
        ),
        # Test Case 5: Empty list for a critical input (chromosomes_list) - should raise an IndexError.
        (
            {
                "num_variants": 1,
                "chromosomes_list": [], # Empty list
                "min_pos": 1,
                "max_pos": 10,
                "alleles_list": ["A"],
                "organisms_list": ["human"],
            },
            IndexError, # random.choice([]) raises IndexError
        ),
    ],
)
def test_generate_synthetic_genetic_variants(input_args, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        # If an exception is expected, use pytest.raises
        with pytest.raises(expected):
            generate_synthetic_genetic_variants(**input_args)
    else:
        # If a DataFrame is expected
        df = generate_synthetic_genetic_variants(**input_args)

        # 1. Check if the output is a pandas DataFrame
        assert isinstance(df, pd.DataFrame), "Output must be a Pandas DataFrame"

        # 2. Check the number of rows
        assert df.shape[0] == expected["shape_rows"], \
            f"Expected {expected['shape_rows']} rows, but got {df.shape[0]}"
        
        # 3. Check column names
        expected_cols = set(expected["columns"])
        actual_cols = set(df.columns)
        assert expected_cols == actual_cols, \
            f"Column mismatch. Expected {sorted(list(expected_cols))}, got {sorted(list(actual_cols))}"

        # Additional checks for non-empty DataFrames
        if expected["shape_rows"] > 0:
            # Check variant_id format and uniqueness
            assert all(df['variant_id'].apply(lambda x: isinstance(x, str) and x.startswith("VAR_"))), \
                "All variant_id's should be strings starting with 'VAR_'"
            assert df['variant_id'].nunique() == expected["shape_rows"], \
                "All variant_id's should be unique"

            # Check position data type and range
            assert pd.api.types.is_integer_dtype(df['position']), "Position column should be of integer type"
            if "position_range" in expected:
                min_pos, max_pos = expected["position_range"]
                assert df['position'].min() >= min_pos, f"Minimum position {df['position'].min()} is below expected {min_pos}"
                assert df['position'].max() <= max_pos, f"Maximum position {df['position'].max()} is above expected {max_pos}"

            # Check allele data types and length (single character)
            assert all(df['reference_allele'].apply(lambda x: isinstance(x, str) and len(x) == 1)), \
                "Reference allele must be a single-character string"
            assert all(df['alternate_allele'].apply(lambda x: isinstance(x, str) and len(x) == 1)), \
                "Alternate allele must be a single-character string"

            # Check specific column values if expected (e.g., for single-option lists)
            if "expected_values" in expected:
                for col, val in expected["expected_values"].items():
                    assert (df[col] == val).all(), \
                        f"All values in column '{col}' should be '{val}', but found other values."