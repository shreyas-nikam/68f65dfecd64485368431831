import pytest
import pandas as pd
from collections import namedtuple
from definition_3d82d95f36f942af876836170006c080 import visualize_parameter_mapping

# Mock classes consistent with notebook specification
Parameter = namedtuple('Parameter', ['name', 'type', 'description', 'default_value'])
MCPTool = namedtuple('MCPTool', ['name', 'description', 'input_parameters', 'output_structure'])

# --- Test Case 1: All parameters successfully mapped ---
param1_t1 = Parameter('gene_name', 'str', 'The name of the gene to visualize effects for.', 'SORT1')
param2_t1 = Parameter('tissue_type', 'str', 'The specific tissue type for the analysis.', 'liver')
mcp_tool_t1 = MCPTool('visualize_variant_effects', 'desc', [param1_t1, param2_t1], 'output_structure')
mapping_results_t1 = {
    'gene_name': {'extracted_value': 'BRCA1', 'mapped_successfully': True},
    'tissue_type': {'extracted_value': 'brain', 'mapped_successfully': True},
}
expected_df_t1 = pd.DataFrame([
    {'Parameter Name': 'gene_name', 'Expected Type': 'str', 'Extracted Value': 'BRCA1', 'Mapped Successfully': True, 'Description': 'The name of the gene to visualize effects for.', 'Default Value': 'SORT1'},
    {'Parameter Name': 'tissue_type', 'Expected Type': 'str', 'Extracted Value': 'brain', 'Mapped Successfully': True, 'Description': 'The specific tissue type for the analysis.', 'Default Value': 'liver'},
], columns=['Parameter Name', 'Expected Type', 'Extracted Value', 'Mapped Successfully', 'Description', 'Default Value'])

# --- Test Case 2: Some parameters not mapped, others mapped; mapping_results has extra keys ---
param1_t2 = Parameter('gene_id', 'str', 'The gene identifier to analyze.', 'EGFR')
param2_t2 = Parameter('threshold_value', 'float', 'A numerical threshold.', 0.05)
mcp_tool_t2 = MCPTool('analyze_expression', 'desc', [param1_t2, param2_t2], 'output_structure')
mapping_results_t2 = {
    'gene_id': {'extracted_value': 'KRAS', 'mapped_successfully': True},
    # 'threshold_value' is missing in mapping_results, simulating it was not found in the prompt.
    'unrelated_key': {'extracted_value': 'some_value', 'mapped_successfully': True} # This extra key should be ignored.
}
expected_df_t2 = pd.DataFrame([
    {'Parameter Name': 'gene_id', 'Expected Type': 'str', 'Extracted Value': 'KRAS', 'Mapped Successfully': True, 'Description': 'The gene identifier to analyze.', 'Default Value': 'EGFR'},
    {'Parameter Name': 'threshold_value', 'Expected Type': 'float', 'Extracted Value': None, 'Mapped Successfully': False, 'Description': 'A numerical threshold.', 'Default Value': 0.05},
], columns=['Parameter Name', 'Expected Type', 'Extracted Value', 'Mapped Successfully', 'Description', 'Default Value'])

# --- Test Case 3: Empty mcp_tool.input_parameters (edge case) ---
mcp_tool_t3 = MCPTool('empty_tool', 'A tool with no parameters.', [], 'output_structure')
mapping_results_t3 = {'any_key': {'extracted_value': 'value', 'mapped_successfully': True}} # Should be ignored as no parameters in tool
expected_df_t3 = pd.DataFrame(columns=['Parameter Name', 'Expected Type', 'Extracted Value', 'Mapped Successfully', 'Description', 'Default Value'])

# --- Test Case 4: MCPTool has parameters, but mapping_results is completely empty (nothing mapped) ---
param1_t4 = Parameter('param_A', 'int', 'First param description.', 10)
param2_t4 = Parameter('param_B', 'bool', 'Second param description.', False)
mcp_tool_t4 = MCPTool('tool_with_params', 'A tool with params but no mapping.', [param1_t4, param2_t4], 'output_structure')
mapping_results_t4 = {} # Empty mapping results
expected_df_t4 = pd.DataFrame([
    {'Parameter Name': 'param_A', 'Expected Type': 'int', 'Extracted Value': None, 'Mapped Successfully': False, 'Description': 'First param description.', 'Default Value': 10},
    {'Parameter Name': 'param_B', 'Expected Type': 'bool', 'Extracted Value': None, 'Mapped Successfully': False, 'Description': 'Second param description.', 'Default Value': False},
], columns=['Parameter Name', 'Expected Type', 'Extracted Value', 'Mapped Successfully', 'Description', 'Default Value'])


@pytest.mark.parametrize("mapping_results, mcp_tool, expected_df", [
    (mapping_results_t1, mcp_tool_t1, expected_df_t1),
    (mapping_results_t2, mcp_tool_t2, expected_df_t2),
    (mapping_results_t3, mcp_tool_t3, expected_df_t3),
    (mapping_results_t4, mcp_tool_t4, expected_df_t4),
])
def test_visualize_parameter_mapping_returns_correct_dataframe(mapping_results, mcp_tool, expected_df):
    """
    Tests that visualize_parameter_mapping correctly generates a DataFrame for various mapping scenarios,
    including successful mappings, partial mappings, and empty input_parameters.
    """
    result_df = visualize_parameter_mapping(mapping_results, mcp_tool)
    pd.testing.assert_frame_equal(result_df, expected_df)


# --- Test Case 5: Invalid mcp_tool object or mapping_results type (error handling) ---
@pytest.mark.parametrize("mapping_results, mcp_tool, expected_exception", [
    ({}, None, TypeError), # mcp_tool is None
    ({}, "not an MCPTool", TypeError), # mcp_tool is a string
    ({}, {'name': 'bad_tool'}, TypeError), # mcp_tool is a dict missing input_parameters
    (None, mcp_tool_t1, TypeError), # mapping_results is None
    ("not a dict", mcp_tool_t1, TypeError), # mapping_results is a string
])
def test_visualize_parameter_mapping_invalid_inputs_raise_error(mapping_results, mcp_tool, expected_exception):
    """
    Tests that visualize_parameter_mapping raises a TypeError for invalid types of mcp_tool or mapping_results.
    """
    with pytest.raises(expected_exception):
        visualize_parameter_mapping(mapping_results, mcp_tool)

