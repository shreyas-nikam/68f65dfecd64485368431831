import pytest
import collections

# Mock classes for testing purposes, matching the spec from the notebook
Parameter = collections.namedtuple('Parameter', ['name', 'type', 'description', 'default_value'])
MCPTool = collections.namedtuple('MCPTool', ['name', 'description', 'input_parameters', 'output_structure'])

# Define sample MCPTool instances based on the notebook specification
visualize_variant_effects_tool = MCPTool(
    name="visualize_variant_effects",
    description="Generates modality-specific visualizations that simplify the interpretation of regulatory impact for a given gene and tissue.",
    input_parameters=[
        Parameter('gene_name', 'str', 'The name of the gene to visualize effects for.', 'SORT1'),
        Parameter('tissue_type', 'str', 'The specific tissue type for the analysis (e.g., "liver", "muscle").', 'liver'),
        Parameter('min_expression_level', 'float', 'Minimum gene expression level to consider.', 0.01)
    ],
    output_structure="A dictionary containing visualization plots (PNG bytes) and summary statistics (DataFrame)."
)

process_data_tool = MCPTool(
    name="process_data",
    description="Processes data based on various criteria.",
    input_parameters=[
        Parameter('entity_id', 'int', 'Unique identifier for the entity.', None),
        Parameter('temperature', 'float', 'Temperature reading.', None),
        Parameter('verbose', 'bool', 'Enable verbose logging.', False),
        Parameter('message', 'str', 'A custom message.', '')
    ],
    output_structure="A processed data summary."
)

# This block should be kept as is. DO NOT REPLACE or REMOVE.
from definition_b95aca517ab3444abafcfe3faff6d392 import simulate_prompt_parsing
# End of your_module block


@pytest.mark.parametrize("mcp_tool, prompt_text, expected_output", [
    # Test Case 1: Happy Path - All parameters found and correctly typed (for visualize_variant_effects_tool)
    (
        visualize_variant_effects_tool,
        "Analyze gene expression for gene 'SORT1' in 'liver' tissue with minimum expression 0.05.",
        {
            'gene_name': {'extracted_value': 'SORT1', 'mapped_successfully': True},
            'tissue_type': {'extracted_value': 'liver', 'mapped_successfully': True},
            'min_expression_level': {'extracted_value': 0.05, 'mapped_successfully': True}
        }
    ),
    # Test Case 2: Edge Case - No parameters found (for visualize_variant_effects_tool)
    (
        visualize_variant_effects_tool,
        "This prompt has nothing to do with genes or tissues at all.",
        {
            'gene_name': {'extracted_value': None, 'mapped_successfully': False},
            'tissue_type': {'extracted_value': None, 'mapped_successfully': False},
            'min_expression_level': {'extracted_value': None, 'mapped_successfully': False}
        }
    ),
    # Test Case 3: Edge Case - Some parameters found, some missing (for visualize_variant_effects_tool)
    (
        visualize_variant_effects_tool,
        "I want to visualize gene 'BRCA1' with minimum expression 0.1.",
        {
            'gene_name': {'extracted_value': 'BRCA1', 'mapped_successfully': True},
            'tissue_type': {'extracted_value': None, 'mapped_successfully': False},
            'min_expression_level': {'extracted_value': 0.1, 'mapped_successfully': True}
        }
    ),
    # Test Case 4: Edge Case - Empty prompt (for visualize_variant_effects_tool)
    (
        visualize_variant_effects_tool,
        "",
        {
            'gene_name': {'extracted_value': None, 'mapped_successfully': False},
            'tissue_type': {'extracted_value': None, 'mapped_successfully': False},
            'min_expression_level': {'extracted_value': None, 'mapped_successfully': False}
        }
    ),
    # Test Case 5: Happy Path - Diverse types (int, float, bool, str) for process_data_tool
    (
        process_data_tool,
        "Process data for entity 123, temperature 98.7, verbose true, and message 'hello world'.",
        {
            'entity_id': {'extracted_value': 123, 'mapped_successfully': True},
            'temperature': {'extracted_value': 98.7, 'mapped_successfully': True},
            'verbose': {'extracted_value': True, 'mapped_successfully': True},
            'message': {'extracted_value': 'hello world', 'mapped_successfully': True}
        }
    ),
])
def test_simulate_prompt_parsing(mcp_tool, prompt_text, expected_output):
    """
    Tests the simulate_prompt_parsing function for various scenarios
    including full and partial parameter extraction, empty prompts, and different data types.
    """
    actual_output = simulate_prompt_parsing(mcp_tool, prompt_text)

    assert actual_output == expected_output
