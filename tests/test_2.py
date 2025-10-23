import pytest
from collections import namedtuple

# definition_7abcc847d8354ed8886f98d58a3e9cec
from definition_7abcc847d8354ed8886f98d58a3e9cec import validate_mcp_tool_definition, MCPTool, Parameter

@pytest.mark.parametrize("mcp_tool_input, expected_output", [
    # Test Case 1: Fully Valid MCPTool definition
    (MCPTool("analyze_data", "Analyzes various data properties.",
             [Parameter("dataset_id", "str", "Identifier for the dataset.", "D001"),
              Parameter("threshold", "float", "A numerical threshold.", 0.5)],
             "Summary report and filtered data."),
     []),
    
    # Test Case 2: MCPTool with an empty top-level field (name)
    (MCPTool("", "Analyzes various data properties.",
             [Parameter("dataset_id", "str", "Identifier for the dataset.", "D001")],
             "Summary report."),
     ['Tool name cannot be empty.']), # Assuming specific error message for empty name
    
    # Test Case 3: MCPTool with an invalid type string for an input parameter
    (MCPTool("analyze_data", "Analyzes various data properties.",
             [Parameter("dataset_id", "str", "Identifier for the dataset.", "D001"),
              Parameter("invalid_param", "non_existent_type", "This parameter has an invalid type.", None)],
             "Summary report."),
     ['Input parameter type "non_existent_type" is not a valid Python type.']), # Assuming specific error message
    
    # Test Case 4: MCPTool with an empty list of input parameters (valid edge case)
    (MCPTool("simple_tool", "A tool with no parameters.", [], "A simple output."),
     []),
    
    # Test Case 5: Passing an object that is not an MCPTool instance (expected TypeError)
    ("this is a string, not an MCPTool object",
     TypeError),
])
def test_validate_mcp_tool_definition(mcp_tool_input, expected_output):
    try:
        actual_output = validate_mcp_tool_definition(mcp_tool_input)
        # If the expected_output is a list (i.e., validation messages)
        assert actual_output == expected_output
    except Exception as e:
        # If the expected_output is an Exception type
        assert isinstance(e, expected_output)

