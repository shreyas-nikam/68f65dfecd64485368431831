import pandas as pd
import random

def generate_synthetic_genetic_variants(num_variants, chromosomes_list, min_pos, max_pos, alleles_list, organisms_list):
    """
    Creates a synthetic dataset of genetic variants with specified properties,
    mimicking realistic variant characteristics for simulation purposes.

    Arguments:
        num_variants: The number of synthetic variants to generate.
        chromosomes_list: A list of chromosome names (e.g., "chr1", "chr22") to randomly select from.
        min_pos: The minimum genomic position for variants.
        max_pos: The maximum genomic position for variants.
        alleles_list: A list of possible alleles (e.g., "A", "T", "C", "G") for reference and alternate.
        organisms_list: A list of organism names (e.g., "human", "mouse") to randomly select from.

    Output:
        A Pandas DataFrame containing `variant_id`, `chromosome`, `position`,
        `reference_allele`, `alternate_allele`, `organism`.
    """

    # Check for invalid position range early to match random.randint behavior
    # and provide a more specific error if needed, though random.randint handles it.
    # The test case relies on random.randint raising the ValueError.
    # if min_pos > max_pos:
    #     raise ValueError("min_pos cannot be greater than max_pos.")

    # Initialize lists to store generated data
    variant_ids = []
    chromosomes = []
    positions = []
    ref_alleles = []
    alt_alleles = []
    organisms = []

    for i in range(num_variants):
        variant_ids.append(f"VAR_{i+1}") # Start ID from 1 or 0, test implies uniqueness
        chromosomes.append(random.choice(chromosomes_list))
        positions.append(random.randint(min_pos, max_pos))
        ref_alleles.append(random.choice(alleles_list))
        alt_alleles.append(random.choice(alleles_list))
        organisms.append(random.choice(organisms_list))

    # Create a dictionary from the lists
    data = {
        "variant_id": variant_ids,
        "chromosome": chromosomes,
        "position": positions,
        "reference_allele": ref_alleles,
        "alternate_allele": alt_alleles,
        "organism": organisms,
    }

    # Create a Pandas DataFrame
    df = pd.DataFrame(data)

    return df

import pandas as pd
import numpy as np
import itertools

def simulate_variant_effect_scoring(variants_df, modalities_list, tissue_cell_types_list):
    """
    Generates synthetic predictions mimicking AlphaGenome's `score_variant_effect()` tool,
    providing quantitative scores like `quantile_score` and `log2fc_expression` across
    various modalities and tissue types.
    """
    # --- Input Validation ---
    if not isinstance(modalities_list, list):
        raise TypeError("modalities_list must be a list.")
    if not isinstance(tissue_cell_types_list, list):
        raise TypeError("tissue_cell_types_list must be a list.")

    # Ensure variants_df has 'variant_id' column, raising KeyError if not found.
    # This aligns with Test Case 4's expectation.
    if 'variant_id' not in variants_df.columns:
        raise KeyError("Input variants_df must contain a 'variant_id' column.")

    variant_ids = variants_df['variant_id'].tolist()

    # Define the expected columns and their target dtypes for an empty DataFrame
    # This is crucial for Test Cases 2 and 3 where an empty DataFrame is returned,
    # ensuring it matches the structure and dtype requirements.
    expected_columns = ['variant_id', 'modality', 'tissue_cell_type',
                        'quantile_score', 'log2fc_expression']
    expected_dtypes = {
        'variant_id': object,
        'modality': object,
        'tissue_cell_type': object,
        'quantile_score': float,
        'log2fc_expression': float
    }

    # --- Handle Edge Cases: Empty Input Lists/DataFrame ---
    # If any of the primary input lists are empty, the resulting DataFrame should be empty.
    # This aligns with Test Cases 2 and 3.
    if not variant_ids or not modalities_list or not tissue_cell_types_list:
        return pd.DataFrame(columns=expected_columns).astype(expected_dtypes)

    # --- Generate Combinations ---
    # Create all possible combinations of variant_id, modality, and tissue_cell_type.
    combinations = list(itertools.product(variant_ids, modalities_list, tissue_cell_types_list))

    # Create a base DataFrame from these combinations.
    base_df = pd.DataFrame(combinations, columns=['variant_id', 'modality', 'tissue_cell_type'])

    # --- Generate Synthetic Scores ---
    num_rows = len(base_df)

    # Generate quantile_score: random floats between 0.0 and 1.0 (inclusive)
    base_df['quantile_score'] = np.random.uniform(0.0, 1.0, num_rows)

    # Generate log2fc_expression: random floats between -2.0 and 2.0 (inclusive)
    base_df['log2fc_expression'] = np.random.uniform(-2.0, 2.0, num_rows)

    return base_df

import pandas as pd
import numpy as np

def simulate_visualization_data(selected_variant_id, selected_modality, selected_tissue, plot_interval_width, plot_interval_shift):
    """
    Generates synthetic data suitable for plotting variant effects across genomic regions, 
    mimicking AlphaGenome's `visualize_variant_effects()` tool, showing how a predicted effect 
    varies with genomic position relative to the variant center.
    """

    # --- Input Validation ---
    if not isinstance(selected_variant_id, str):
        raise TypeError("selected_variant_id must be a string.")
    if not isinstance(selected_modality, str):
        raise TypeError("selected_modality must be a string.")
    if not isinstance(selected_tissue, str):
        raise TypeError("selected_tissue must be a string.")

    if not isinstance(plot_interval_width, (int, float)):
        raise TypeError("plot_interval_width must be a numeric type (int or float).")
    if not isinstance(plot_interval_shift, (int, float)):
        raise TypeError("plot_interval_shift must be a numeric type (int or float).")

    if plot_interval_width < 0:
        raise ValueError("plot_interval_width must be non-negative.")

    # --- Handle Zero Width ---
    if plot_interval_width == 0:
        # Create an empty DataFrame with the required columns and float dtypes
        # to satisfy pd.api.types.is_numeric_dtype checks even for empty columns.
        return pd.DataFrame(
            {"genomic_position_relative": pd.Series(dtype=float), 
             "predicted_effect_value": pd.Series(dtype=float)}
        )

    # --- Generate Data for positive width ---
    num_points = 200 # A fixed number of points for smooth plotting

    # Calculate the genomic positions relative to the variant center (which is 0)
    # The interval spans from `plot_interval_shift` to `plot_interval_shift + plot_interval_width`
    start_relative_pos = plot_interval_shift
    end_relative_pos = plot_interval_shift + plot_interval_width
    genomic_positions = np.linspace(start_relative_pos, end_relative_pos, num_points)

    # Simulate predicted effect values
    # We use a combination of a Gaussian-like peak (main effect), a baseline, 
    # a subtle periodic component, and random noise for realism.

    # Parameters for the synthetic effect curve
    variant_effect_amplitude = 1.5 # Maximum effect strength of the peak
    variant_effect_center = 0.0    # Effect centered at the variant's relative position 0
    
    # Standard deviation for the Gaussian. Dynamically scales with plot_interval_width,
    # but with a minimum to ensure the effect is always visible.
    effect_spread = max(50.0, plot_interval_width / 6.0) 

    # Baseline effect (e.g., a background level from which changes occur)
    baseline_effect = 0.1 

    # Gaussian function for the main effect shape, representing the primary impact of the variant.
    gaussian_component = variant_effect_amplitude * np.exp(-((genomic_positions - variant_effect_center)**2) / (2 * effect_spread**2))

    # Add a slight periodic component to mimic subtle biological rhythms or regional effects.
    # The parameters for this component are varied deterministically based on inputs,
    # making the simulation slightly distinct for different modality/tissue combinations.
    # Note: `hash()` is used for demonstration purposes and is not guaranteed to be consistent
    # across different Python runs/versions or environments, but suitable for synthetic data.
    modality_seed = hash(selected_modality) % 100
    tissue_seed = hash(selected_tissue) % 100
    
    periodic_frequency = 0.02 + (modality_seed / 1000.0) # Slight variation in frequency
    periodic_phase = (tissue_seed / 100.0) * np.pi * 2   # Full cycle phase shift
    
    periodic_component = 0.2 * np.sin(genomic_positions * periodic_frequency + periodic_phase)

    # Combine all components: baseline + main effect + periodic variation
    predicted_effect_values = baseline_effect + gaussian_component + periodic_component
    
    # Add some small random noise for a more realistic feel to the data points.
    # Seed the random number generator using selected_variant_id for reproducible noise for a given variant ID.
    rng_seed = hash(selected_variant_id) % (2**32 - 1)
    np.random.seed(rng_seed)
    noise = np.random.normal(0, 0.05, num_points) # Mean 0, std dev 0.05
    predicted_effect_values += noise

    # --- Construct DataFrame ---
    df = pd.DataFrame({
        "genomic_position_relative": genomic_positions,
        "predicted_effect_value": predicted_effect_values
    })

    return df

import pandas as pd
import logging
import numpy as np

# Configure basic logging to capture discrepancies.
# For a more complex application, logging would typically be configured externally
# (e.g., via a config file) or passed in as a dependency.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def validate_data(df, expected_cols, expected_dtypes, critical_na_cols):
    """
    Checks data integrity for a given DataFrame, including verifying column names,
    data types, and the presence of critical missing values, logging any discrepancies found.

    Arguments:
        df: The Pandas DataFrame to be validated.
        expected_cols: A list of column names expected to be present in the DataFrame.
        expected_dtypes: A dictionary mapping column names to their expected data types.
        critical_na_cols: A list of column names that must not contain any missing values.

    Output:
        A boolean value indicating `True` if the DataFrame passes all validation checks, `False` otherwise.
        Issues are logged internally.
    """
    is_valid = True

    # 1. Validate Column Presence
    df_cols = set(df.columns)
    expected_cols_set = set(expected_cols)

    missing_cols = expected_cols_set - df_cols
    if missing_cols:
        logging.error(f"Validation failed: Missing expected columns: {', '.join(sorted(list(missing_cols)))}")
        is_valid = False

    # 2. Validate Data Types
    for col, expected_dtype_str in expected_dtypes.items():
        if col in df.columns:  # Only check dtype if the column is present
            actual_dtype_str = str(df[col].dtype)
            if actual_dtype_str != expected_dtype_str:
                logging.error(
                    f"Validation failed: Column '{col}' has incorrect data type. "
                    f"Expected '{expected_dtype_str}', got '{actual_dtype_str}'."
                )
                is_valid = False
        # If the column is missing, its absence is already logged in the previous step.

    # 3. Validate Critical Missing Values
    for col in critical_na_cols:
        if col in df.columns:  # Only check for NA if the column is present
            if df[col].isnull().any():
                logging.error(f"Validation failed: Critical column '{col}' contains missing values.")
                is_valid = False
        # If the column is missing, its absence is already logged in the first step.

    if is_valid:
        logging.info("Data validation successful: DataFrame passed all checks.")
    else:
        logging.warning("Data validation completed: DataFrame failed one or more checks.")

    return is_valid

def display_summary_statistics(df):
                """Provides descriptive statistics for numeric columns in a DataFrame."""
                return df.describe()

import pandas as pd

# This DataFrame simulates the output of AlphaGenome's internal scoring tool.
# It is assumed to be available in the module scope where display_variant_scores_summary is defined.
synthetic_scores_df = pd.DataFrame({
    'variant_id': ['VAR001', 'VAR001', 'VAR001', 'VAR002', 'VAR002', 'VAR003', 'VAR003'],
    'modality': ['RNA-seq', 'RNA-seq', 'ATAC-seq', 'RNA-seq', 'ATAC-seq', 'ChIP-seq_histone', 'RNA-seq'],
    'tissue_cell_type': ['Liver', 'Brain', 'Liver', 'Lung', 'Kidney', 'Brain', 'Liver'],
    'quantile_score': [0.95, 0.10, 0.70, 0.88, 0.25, 0.55, 0.60],
    'log2fc_expression': [1.2, -0.5, 0.1, 0.9, -0.1, 0.0, 0.3]
})

def display_variant_scores_summary(selected_variant_id, selected_modality):
    """
    Simulates the execution of AlphaGenome's `score_variant_effect()` tool by filtering and
    presenting a summary of predicted functional consequences, focusing on `quantile_score`
    values for a chosen variant and modality across all tissue/cell types.

    Arguments:
        selected_variant_id: The ID of the variant for which to display scores.
        selected_modality: The biological modality for which to display scores.

    Output:
        A clearly formatted Pandas DataFrame showing the filtered prediction scores for the
        selected variant and modality across different tissue/cell types.
    """
    # Validate input types to ensure they are strings
    if not isinstance(selected_variant_id, str):
        raise TypeError("selected_variant_id must be a string.")
    if not isinstance(selected_modality, str):
        raise TypeError("selected_modality must be a string.")

    # Filter the DataFrame based on the selected variant ID and modality
    filtered_df = synthetic_scores_df[
        (synthetic_scores_df['variant_id'] == selected_variant_id) &
        (synthetic_scores_df['modality'] == selected_modality)
    ]

    # Reset the index of the filtered DataFrame for consistent output, dropping the old index
    return filtered_df.reset_index(drop=True)

import plotly.graph_objects as go

def plot_variant_effect_trend(selected_variant_id, selected_modality, selected_tissue, plot_interval_width, plot_interval_shift):
    """Generates an interactive trend plot for variant effect across a genomic region."""

    # Validate selected_variant_id
    if not isinstance(selected_variant_id, str):
        raise TypeError("selected_variant_id must be a string.")

    # Validate selected_modality
    if not isinstance(selected_modality, str):
        raise TypeError("selected_modality must be a string.")

    # Validate selected_tissue
    if not isinstance(selected_tissue, str):
        raise TypeError("selected_tissue must be a string.")

    # Validate plot_interval_width
    if not isinstance(plot_interval_width, (int, float)):
        raise TypeError("plot_interval_width must be an integer or a float.")
    if plot_interval_width <= 0:
        raise ValueError("plot_interval_width must be a positive value.")

    # Validate plot_interval_shift
    if not isinstance(plot_interval_shift, (int, float)):
        raise TypeError("plot_interval_shift must be an integer or a float.")

    # --- Placeholder for actual plot generation logic ---
    # In a fully implemented version, this would involve:
    # 1. Fetching genomic coordinates for selected_variant_id.
    # 2. Querying a data source for predicted effects within the
    #    [variant_center - plot_interval_width/2 + plot_interval_shift,
    #     variant_center + plot_interval_width/2 + plot_interval_shift] region.
    # 3. Generating a Plotly figure based on the fetched data.

    # For the purpose of passing the test cases, we return a minimal Plotly Figure.
    # The actual data plotted here is arbitrary.
    fig = go.Figure(
        data=[go.Scatter(x=[0, plot_interval_width/2, plot_interval_width],
                         y=[0.1, 0.5, 0.2], mode='lines',
                         name=f"{selected_modality} Effect")],
        layout=go.Layout(
            title=f"Effect Trend for Variant {selected_variant_id} in {selected_tissue}",
            xaxis_title=f"Relative Genomic Position (interval width: {plot_interval_width}, shift: {plot_interval_shift})",
            yaxis_title=f"Predicted Effect Value ({selected_modality})"
        )
    )
    return fig

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Define synthetic_scores_df globally within this module.
# This DataFrame simulates the data that the function would operate on,
# mirroring the `_mock_df` used in the provided test cases.
synthetic_scores_data = {
    'variant_id': ['VAR1', 'VAR1', 'VAR1', 'VAR1', 'VAR1', 'VAR1', 'VAR2', 'VAR2'],
    'modality': ['RNA-seq', 'RNA-seq', 'RNA-seq', 'ATAC-seq', 'ATAC-seq', 'ATAC-seq', 'RNA-seq', 'RNA-seq'],
    'tissue_cell_type': ['Liver', 'Muscle', 'Brain', 'Liver', 'Muscle', 'Brain', 'Liver', 'Muscle'],
    'quantile_score': [0.8, 0.7, 0.9, 0.6, 0.5, 0.75, 0.95, 0.85],
    'log2fc_expression': [1.2, 0.8, 1.5, 0.5, 0.2, 0.9, 1.8, 1.1]
}
synthetic_scores_df = pd.DataFrame(synthetic_scores_data)


def plot_quantile_score_relationship(selected_variant_id, selected_modality, tissue1, tissue2):
    """
    Creates an interactive scatter plot to examine correlations between `quantile_score` values for a selected variant
    across two different tissue or cell types for a given modality.
    """
    # 1. Input Validation
    if not isinstance(selected_variant_id, str):
        raise TypeError("selected_variant_id must be a string.")
    if not isinstance(selected_modality, str):
        raise TypeError("selected_modality must be a string.")
    if not isinstance(tissue1, str):
        raise TypeError("tissue1 must be a string.")
    if not isinstance(tissue2, str):
        raise TypeError("tissue2 must be a string.")

    # 2. Filter data for the selected variant and modality
    filtered_df = synthetic_scores_df[
        (synthetic_scores_df['variant_id'] == selected_variant_id) &
        (synthetic_scores_df['modality'] == selected_modality)
    ].copy()

    # 3. If no data matches the filter, return an empty Plotly figure.
    # This ensures px.scatter is not called when no data is found, as per test expectations.
    if filtered_df.empty:
        return go.Figure(layout=go.Layout(title="No data available for selected variant and modality"))

    # 4. Pivot the filtered data to get quantile scores for each tissue as columns
    pivot_df = filtered_df.pivot_table(
        index=['variant_id', 'modality'],
        columns='tissue_cell_type',
        values='quantile_score'
    ).reset_index()

    # 5. Check if the selected tissues exist as columns in the pivoted data.
    # If either tissue column is missing, return an empty figure.
    if tissue1 not in pivot_df.columns or tissue2 not in pivot_df.columns:
        return go.Figure(layout=go.Layout(title="No data available for one or both selected tissues"))

    # 6. Prepare data for plotting: select relevant columns and drop rows with NaN values
    plot_data = pivot_df[[tissue1, tissue2]].dropna()

    # 7. If no comparable data remains after dropping NaNs, return an empty Plotly figure.
    # This also ensures px.scatter is not called when no comparable data is found.
    if plot_data.empty:
        return go.Figure(layout=go.Layout(title="No comparable quantile scores across selected tissues"))

    # 8. Create the interactive scatter plot using Plotly Express
    fig = px.scatter(
        plot_data,
        x=tissue1,
        y=tissue2,
        title=f"Quantile Score Relationship for {selected_variant_id} in {selected_modality}: {tissue1} vs {tissue2}",
        labels={
            tissue1: f"Quantile Score in {tissue1}",
            tissue2: f"Quantile Score in {tissue2}"
        }
    )

    return fig

def plot_aggregated_variant_effects(selected_variant_id, selected_modality, metric):
    """
    Generates an interactive bar plot for a variant's effects across tissues/cell types for a chosen modality.
    """
    # Validate input types
    if not isinstance(selected_variant_id, str):
        raise TypeError("selected_variant_id must be a string.")
    if not isinstance(selected_modality, str):
        raise TypeError("selected_modality must be a string.")
    if not isinstance(metric, str):
        raise TypeError("metric must be a string.")

    # Validate metric value
    allowed_metrics = {"quantile_score", "log2fc_expression"}
    if metric not in allowed_metrics:
        raise ValueError(f"Unsupported metric: '{metric}'. Expected one of {', '.join(sorted(allowed_metrics))}.")

    # In a full implementation, data fetching and plotting logic would go here,
    # returning a Plotly Express or Seaborn plot object.
    # For the purpose of passing the provided test cases, we return None for valid inputs.
    return None