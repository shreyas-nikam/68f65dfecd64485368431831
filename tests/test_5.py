import pytest
import collections
from definition_0b6c01c637254387b1edd62c1044415f import generate_mcp_tool_snippet

# Helper definitions for MCPTool and Parameter, assuming these are defined
# in your_module or are fundamental data structures for the function.
# For robust testing, we define them here to ensure consistency.
Parameter = collections.namedtuple('Parameter', ['name', 'type', 'description', 'default_value'])
MCPTool = collections.namedtuple('MCPTool', ['name', 'description', 'input_parameters', 'output_structure'])

def test_generate_mcp_tool_snippet_standard_tool():
    """
    Test case 1: Verifies correct snippet generation for a standard tool
    with a mix of required and optional parameters of various types.
    """
    tool_params = [
        Parameter('tissue_type', 'str', 'Specific tissue type.', None), # Required parameter
        Parameter('gene_name', 'str', 'The name of the gene.', 'SORT1'), # Optional parameter with default
        Parameter('min_expression', 'float', 'Minimum expression level.', 0.05),
        Parameter('max_results', 'int', 'Max results to return.', 100)
    ]
    mcp_tool_standard = MCPTool(
        name='analyze_gene_expression',
        description='Analyzes gene expression levels and returns insights.',
        input_parameters=tool_params,
        output_structure='dict'
    )
    expected_snippet = """@mcp.tools
def analyze_gene_expression(tissue_type: str, gene_name: str = 'SORT1', min_expression: float = 0.05, max_results: int = 100):
    \"\"\"
        Analyzes gene expression levels and returns insights.
    \"\"\"
    pass"""
    assert generate_mcp_tool_snippet(mcp_tool_standard) == expected_snippet

def test_generate_mcp_tool_snippet_no_parameters():
    """
    Test case 2: Checks snippet generation for a tool with no input parameters.
    The function signature should be empty `()`.
    """
    mcp_tool_no_params = MCPTool(
        name='get_system_status',
        description='Retrieves the current system status.',
        input_parameters=[],
        output_structure='str'
    )
    expected_snippet = """@mcp.tools
def get_system_status():
    \"\"\"
        Retrieves the current system status.
    \"\"\"
    pass"""
    assert generate_mcp_tool_snippet(mcp_tool_no_params) == expected_snippet

def test_generate_mcp_tool_snippet_all_required_parameters():
    """
    Test case 3: Ensures correct signature generation when all parameters are required
    (i.e., have no default values).
    """
    tool_params_no_defaults = [
        Parameter('file_path', 'str', 'Path to the file to process.', None),
        Parameter('threshold', 'float', 'Processing threshold value.', None),
    ]
    mcp_tool_all_required = MCPTool(
        name='process_data',
        description='Processes data from a given file path with a specified threshold.',
        input_parameters=tool_params_no_defaults,
        output_structure='DataFrame'
    )
    expected_snippet = """@mcp.tools
def process_data(file_path: str, threshold: float):
    \"\"\"
        Processes data from a given file path with a specified threshold.
    \"\"\"
    pass"""
    assert generate_mcp_tool_snippet(mcp_tool_all_required) == expected_snippet

def test_generate_mcp_tool_snippet_empty_description():
    """
    Test case 4: Verifies the behavior when the MCPTool has an empty description.
    The docstring block should still be present but empty.
    """
    tool_params_empty_desc = [
        Parameter('input_id', 'int', 'ID for the input.', 1)
    ]
    mcp_tool_empty_desc = MCPTool(
        name='simple_action',
        description='',  # Empty description
        input_parameters=tool_params_empty_desc,
        output_structure='bool'
    )
    expected_snippet = """@mcp.tools
def simple_action(input_id: int = 1):
    \"\"\"
    \"\"\"
    pass"""
    assert generate_mcp_tool_snippet(mcp_tool_empty_desc) == expected_snippet

@pytest.mark.parametrize("invalid_input", [
    None,
    123,
    "not_an_mcp_tool",
    [],
    {},
    object(),
])
def test_generate_mcp_tool_snippet_invalid_mcp_tool_type(invalid_input):
    """
    Test case 5: Checks for appropriate error handling when the input `mcp_tool`
    is not an instance of MCPTool or is malformed. Expects an AttributeError
    as the function would attempt to access attributes like .name, .description, etc.
    """
    with pytest.raises(AttributeError):
        generate_mcp_tool_snippet(invalid_input)