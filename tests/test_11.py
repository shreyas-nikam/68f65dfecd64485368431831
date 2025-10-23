import pytest
# Parameter and MCPTool namedtuples are assumed to be defined in definition_88e21105527b4c0a93802d4057b4011c
# as per the notebook specification and are required for constructing the expected output.
from definition_88e21105527b4c0a93802d4057b4011c import collect_widget_values_into_mcp_tool_func, Parameter, MCPTool

@pytest.mark.parametrize(
    "tool_name, tool_desc, output_struct, param1_name, param1_type, param1_desc, param1_default, expected",
    [
        # Test Case 1: Happy Path - All valid inputs
        # Verifies that a well-formed MCPTool object is created with all provided details.
        ("visualize_gene_data", "Visualizes gene expression data based on provided gene ID.", "dict",
         "gene_id", "str", "The unique identifier of the gene to visualize.", "BRCA1",
         MCPTool(name="visualize_gene_data", description="Visualizes gene expression data based on provided gene ID.",
                 output_structure="dict",
                 input_parameters=[Parameter(name="gene_id", type="str", description="The unique identifier of the gene to visualize.", default_value="BRCA1")])),

        # Test Case 2: Edge Case - Empty strings for optional description and default value fields
        # Ensures the function correctly handles empty strings for fields where they are permissible.
        ("analyze_sample_data", "", "json_output",
         "sample_name", "str", "", "",
         MCPTool(name="analyze_sample_data", description="", output_structure="json_output",
                 input_parameters=[Parameter(name="sample_name", type="str", description="", default_value="")])),

        # Test Case 3: Edge Case - Empty strings for 'mandatory' names (tool_name, param1_name)
        # The function's role is construction, not validation of name content. It should construct the object
        # with empty strings, leaving subsequent validation logic to flag these as errors.
        ("", "A tool with no explicit name but functional description.", "text",
         "", "int", "A parameter with no specific name.", "0",
         MCPTool(name="", description="A tool with no explicit name but functional description.",
                 output_structure="text",
                 input_parameters=[Parameter(name="", type="int", description="A parameter with no specific name.", default_value="0")])),

        # Test Case 4: Error Case - Invalid type for 'tool_name' (expected str, got int)
        # Tests robust error handling when an argument does not match the expected string type.
        (123, "Tool description.", "xml",
         "data_id", "str", "Identifier for data.", "id_123",
         TypeError),

        # Test Case 5: Error Case - Invalid type for 'param1_default' (expected str, got bool)
        # Verifies that the function raises a TypeError if a parameter's default value is not a string,
        # adhering to the function signature's type hints.
        ("log_event", "Logs a system event.", "log_entry",
         "is_critical", "bool", "Indicates if the event is critical.", True, # This value is a boolean, not a string
         TypeError),
    ]
)
def test_collect_widget_values_into_mcp_tool_func(tool_name, tool_desc, output_struct, param1_name, param1_type, param1_desc, param1_default, expected):
    try:
        # Call the function from the module under test
        result = collect_widget_values_into_mcp_tool_func(tool_name, tool_desc, output_struct, param1_name, param1_type, param1_desc, param1_default)
        
        # Assert the result is an MCPTool and matches the expected MCPTool object
        assert result == expected
        assert isinstance(result, MCPTool)
        # Further check the nested parameter type for completeness, assuming there's at least one.
        assert isinstance(result.input_parameters[0], Parameter)

    except Exception as e:
        # If an exception is expected, assert its type
        assert isinstance(e, expected)
