
# Technical Specification for Jupyter Notebook: MCP Tool Designer

## 1. Notebook Overview

This Jupyter Notebook provides an interactive environment to design and simulate Model Context Protocol (MCP) tools, the fundamental building blocks for AI agents as described in the Paper2Agent framework. Users will gain practical experience in defining tool structures, simulating agent interactions, and visualizing parameter mappings.

### Learning Goals

*   Understand the core components of an MCP tool, including its name, description, input parameters, and expected outputs, based on the Paper2Agent framework.
*   Learn how to define clear and flexible input parameters with specified data types, descriptions, and default values, as exemplified by tools within AlphaGenome and Scanpy.
*   Explore the conceptual process of how natural language prompts are translated into structured tool invocations by a simulated AI agent.
*   Grasp the concept of MCP as a standardized protocol for exposing APIs and tools to large language models (LLMs) and agent frameworks.
*   Develop skills in generating and validating structured tool definitions.
*   Understand data validation techniques and generate insightful visualizations for various data types.

## 2. Code Requirements

### List of Expected Libraries

1.  `pandas`: For data manipulation and tabular data handling.
2.  `numpy`: For numerical operations and synthetic data generation.
3.  `matplotlib.pyplot`: For static plotting and fallback visualizations.
4.  `seaborn`: For enhanced statistical data visualization, utilizing color-blind friendly palettes.
5.  `ipywidgets`: For interactive user inputs (sliders, dropdowns, text inputs) and interactive plot controls.
6.  `IPython.display`: For displaying rich output like Markdown and images.
7.  `datetime`: For handling time-series data generation.
8.  `collections.namedtuple`: For a simple, structured definition of MCP tools and parameters.

### List of Algorithms or Functions to be Implemented

1.  **`generate_synthetic_data(num_rows, start_date, num_categories)`**: Generates a `pandas.DataFrame` with specified rows, starting date for time-series, and number of categories. It will include numeric, categorical, and time-series columns, mimicking data suitable for genomic or biological analysis (e.g., `timestamp`, `entity_id`, `numeric_feature_A`, `numeric_feature_B`, `categorical_feature`, `gene_expression_level`, `tissue_type`).
2.  **`validate_and_summarize_data(dataframe)`**: Confirms expected column names and data types, asserts primary-key uniqueness (`entity_id`), checks for missing values in critical fields, and logs summary statistics for numeric columns (mean, std, min, max, quartiles).
3.  **`MCPTool` Class Definition**: A Python class or `namedtuple` to represent an MCP tool, containing attributes like `name`, `description`, `input_parameters` (a list of `Parameter` objects), and `output_structure`.
4.  **`Parameter` Class Definition**: A Python class or `namedtuple` to represent an input parameter, containing attributes like `name`, `type` (as a string, e.g., "str", "int", "float", "list"), `description`, and `default_value`.
5.  **`validate_mcp_tool_definition(mcp_tool)`**: Validates an `MCPTool` instance. Checks if `name`, `description`, and `output_structure` are non-empty strings. For each `input_parameter`, it checks if `name`, `type`, and `description` are non-empty, and if the specified `type` string can be resolved to a valid Python type (e.g., `eval("int")`).
6.  **`simulate_prompt_parsing(mcp_tool, prompt_text)`**: Simulates an AI agent parsing a natural language `prompt_text` to extract values for `mcp_tool`'s input parameters. This function will use simple string matching (e.g., regex-like patterns) or keyword extraction to find and map values, returning a dictionary of `{'parameter_name': 'extracted_value', 'mapped_successfully': True/False}`. It will use predefined keywords associated with parameter names for this simulation.
7.  **`visualize_parameter_mapping(mapping_results)`**: Generates a `pandas.DataFrame` and displays it, showing `Parameter Name`, `Expected Type`, `Extracted Value`, and `Mapped Successfully` status for clear feedback on prompt parsing.
8.  **`generate_mcp_tool_snippet(mcp_tool)`**: Generates a Python-like code string for the defined `mcp_tool`, illustrating how it would be encapsulated with an `@mcp.tools` decorator, including its function signature based on `input_parameters` and a docstring from `tool_description`.
9.  **`simulate_tool_execution(data, tool_parameters)`**: A dummy function that simulates the execution of the defined MCP tool. It takes the synthetic `data` and the `tool_parameters` (extracted from prompt parsing) and performs a placeholder operation (e.g., filtering the dataset based on `gene_expression_level` and `tissue_type` if they are in `tool_parameters`). It returns a dictionary of 'simulated_results'.
10. **`plot_time_series(dataframe, time_col, value_col, title, x_label, y_label, color_palette='viridis', save_path=None)`**: Generates a line plot using `seaborn.lineplot` for time-based metrics.
11. **`plot_relationship(dataframe, x_col, y_col, hue_col=None, title, x_label, y_label, color_palette='viridis', save_path=None)`**: Generates a scatter plot using `seaborn.scatterplot` to examine correlations.
12. **`plot_categorical_comparison(dataframe, category_col, value_col, aggregation_func='mean', title, x_label, y_label, color_palette='viridis', save_path=None)`**: Generates a bar plot or heatmap using `seaborn.barplot` (after aggregation) for categorical insights.
13. **`make_interactive_plot(plot_func, data, **widget_params)`**: A wrapper function using `ipywidgets.interactive` to enable user interaction with the plotting functions by providing dropdowns for column selection, text inputs for titles/labels, etc.

### Visualization like charts, tables, plots that should be generated

1.  **Tables for Data Overview:** `pandas.DataFrame` displays of synthetic data head, `info()`, `describe()`, and validation results.
2.  **Parameter Mapping Table:** A `pandas.DataFrame` visualizing `Parameter Name`, `Expected Type`, `Extracted Value`, and `Mapped Successfully` from prompt parsing.
3.  **Core Visual 1: Trend Plot (Line Plot)**: A line plot showing trends over time, e.g., `gene_expression_level` over `timestamp`.
    *   **Style:** `seaborn.lineplot`, color-blind friendly palette (`viridis`, `cividis`, or `mako`), font size $\ge 12 \text{pt}$.
    *   **Elements:** Clear title, labeled X and Y axes, legend if multiple lines.
    *   **Interactivity:** Allow selection of `value_col` and aggregation (mean, sum) via `ipywidgets` dropdowns/sliders.
    *   **Fallback:** Static PNG saved if interactive environment not available.
4.  **Core Visual 2: Relationship Plot (Scatter Plot)**: A scatter plot examining correlations between two numeric features, potentially with a categorical hue. E.g., `numeric_feature_A` vs. `numeric_feature_B`, colored by `categorical_feature` or `tissue_type`.
    *   **Style:** `seaborn.scatterplot`, color-blind friendly palette, font size $\ge 12 \text{pt}$.
    *   **Elements:** Clear title, labeled X and Y axes, legend for hue.
    *   **Interactivity:** Allow selection of `x_col`, `y_col`, and optional `hue_col` via `ipywidgets` dropdowns.
    *   **Fallback:** Static PNG saved.
5.  **Core Visual 3: Aggregated Comparison (Bar Plot)**: A bar plot or heatmap comparing a numeric metric across categories. E.g., average `gene_expression_level` per `tissue_type` or `categorical_feature`.
    *   **Style:** `seaborn.barplot`, color-blind friendly palette, font size $\ge 12 \text{pt}$.
    *   **Elements:** Clear title, labeled X and Y axes, legend if applicable.
    *   **Interactivity:** Allow selection of `category_col`, `value_col`, and aggregation function (mean, median, sum) via `ipywidgets` dropdowns.
    *   **Fallback:** Static PNG saved.

## 3. Notebook Sections (in detail)

### Section 1: Introduction to Paper2Agent and MCP Tools

*   **Markdown Cell:**
    Explains the context of Paper2Agent as an automated framework converting research papers into interactive AI agents. Introduces Model Context Protocol (MCP) as a standardized way to expose APIs and tools to LLMs and agent frameworks. Emphasizes how MCP tools encapsulate methodological contributions from papers, enabling natural language interaction and autonomous execution. Mentions the core components of MCP tools: name, description, input parameters, and output structure, referencing Figure 2B of the Paper2Agent paper.

### Section 2: Environment Setup and Library Imports

*   **Markdown Cell:**
    Explains the necessary Python libraries for data handling, numerical operations, interactive widgets, and advanced visualizations. It highlights the use of `ipywidgets` for user interaction and `seaborn` for creating aesthetically pleasing and accessible plots with color-blind friendly palettes.
*   **Code Cell (Imports):**
    This cell imports all required libraries:
    `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `ipywidgets`, `IPython.display`, `datetime`, `collections.namedtuple`.
*   **Code Cell (Execution):**
    No direct execution, just imports.
*   **Markdown Cell:**
    Confirms that all libraries have been imported successfully, making them available for use throughout the notebook.

### Section 3: Synthetic Dataset Generation

*   **Markdown Cell:**
    Describes the generation of a synthetic dataset to simulate real-world biological or genomic data. The dataset will include time-series, numerical, and categorical features to allow for diverse analytical scenarios.
    $$ N_{\text{rows}} $$
    where $N_{\text{rows}}$ is the number of rows (e.g., samples or observations).
    The dataset will contain the following columns:
    *   `timestamp`: A `datetime` column, simulating temporal progression.
    *   `entity_id`: An `int` column, serving as a unique identifier for each observation.
    *   `numeric_feature_A`: A `float` column, representing a continuous measurement.
    *   `numeric_feature_B`: A `float` column, representing another continuous measurement, possibly correlated with `numeric_feature_A`.
    *   `categorical_feature`: A `string` column with a few distinct categories.
    *   `gene_expression_level`: A `float` column, simulating gene expression.
    *   `tissue_type`: A `string` column, representing different tissue types (e.g., 'liver', 'muscle', 'brain').
*   **Code Cell (Function Definition):**
    Define `generate_synthetic_data(num_rows, start_date, num_categories)`:
    This function uses `numpy.random.rand` for numeric data, `pandas.date_range` for timestamps, and random choices for categorical data. It creates and returns a `pandas.DataFrame`.
*   **Code Cell (Execution):**
    Execute `generate_synthetic_data` with `num_rows=1000`, `start_date='2023-01-01'`, `num_categories=3` to create `synthetic_df`. Display the head of `synthetic_df` and its `.info()`.
*   **Markdown Cell:**
    Explains the structure of the generated synthetic dataset, showing the first few rows and detailing the data types and non-null counts for each column.

### Section 4: Dataset Validation and Summary Statistics

*   **Markdown Cell:**
    Emphasizes the importance of data validation to ensure data quality and integrity before analysis. This step confirms expected column names, data types, primary key uniqueness, and absence of critical missing values, as well as providing key descriptive statistics.
*   **Code Cell (Function Definition):**
    Define `validate_and_summarize_data(dataframe)`:
    This function performs the validation:
    1.  Checks for expected columns (`timestamp`, `entity_id`, `numeric_feature_A`, `numeric_feature_B`, `categorical_feature`, `gene_expression_level`, `tissue_type`) and their `dtype`s.
    2.  Asserts `entity_id` column has unique values.
    3.  Checks for any missing values in `numeric_feature_A`, `numeric_feature_B`, `gene_expression_level`, and `tissue_type`.
    4.  Prints `dataframe.describe()` for all numeric columns.
    5.  Prints `dataframe.value_counts()` for categorical columns.
*   **Code Cell (Execution):**
    Execute `validate_and_summarize_data(synthetic_df)`.
*   **Markdown Cell:**
    Interprets the output of the validation and summary statistics, confirming data readiness and highlighting key characteristics of the synthetic dataset.

### Section 5: Defining a Base MCP Tool Structure

*   **Markdown Cell:**
    Introduces the fundamental structure for defining an MCP tool, which mimics the design shown in Paper2Agent's Figure 2B. This structure allows agents to understand a tool's capabilities, its required inputs, and expected outputs. It uses simple Python constructs for clarity.
    An MCP tool consists of:
    *   `name`: A unique identifier for the tool.
    *   `description`: A natural language explanation of what the tool does.
    *   `input_parameters`: A list of `Parameter` objects, each detailing an input.
    *   `output_structure`: A description of the data or artifacts produced by the tool.
*   **Code Cell (Class/NamedTuple Definition):**
    Define `Parameter` (e.g., `namedtuple('Parameter', ['name', 'type', 'description', 'default_value'])`) and `MCPTool` (e.g., `namedtuple('MCPTool', ['name', 'description', 'input_parameters', 'output_structure'])`).
*   **Code Cell (Execution):**
    Create a sample `MCPTool` instance, e.g., `visualize_variant_effects_tool`:
    *   `name`: "visualize_variant_effects"
    *   `description`: "Generates modality-specific visualizations that simplify the interpretation of regulatory impact for a given gene and tissue."
    *   `input_parameters`: A list containing:
        *   `Parameter('gene_name', 'str', 'The name of the gene to visualize effects for.', 'SORT1')`
        *   `Parameter('tissue_type', 'str', 'The specific tissue type for the analysis (e.g., "liver", "muscle").', 'liver')`
        *   `Parameter('min_expression_level', 'float', 'Minimum gene expression level to consider.', 0.01)`
    *   `output_structure`: "A dictionary containing visualization plots (PNG bytes) and summary statistics (DataFrame)."
    Display the created `visualize_variant_effects_tool`.
*   **Markdown Cell:**
    Explains the structure of the `visualize_variant_effects_tool` object, detailing how each component (`name`, `description`, `input_parameters`, `output_structure`) contributes to a clear and functional tool definition.

### Section 6: User Input for Tool Definition (Simulated via Widgets)

*   **Markdown Cell:**
    Simulates the user interface for defining an MCP tool. Users can specify the tool's core attributes and its input parameters using interactive widgets, mirroring the "Tool Definition Form" feature.
*   **Code Cell (Function Definition & Widget Setup):**
    Define an interactive function `define_tool_widgets(tool_name, tool_desc, output_struct, param1_name, param1_type, param1_desc, param1_default)` that uses `ipywidgets.Text` for tool-level details and `ipywidgets.Text`, `ipywidgets.Dropdown` for parameter details. This function captures these inputs.
    Use `ipywidgets.interactive` to display the widgets.
    *   `tool_name_widget = ipywidgets.Text(value='analyze_gene_expression', description='Tool Name:')`
    *   `tool_desc_widget = ipywidgets.Textarea(value='Analyzes gene expression for a given gene and tissue, providing summary statistics.', description='Description:')`
    *   `output_struct_widget = ipywidgets.Text(value='{"summary_df": "pandas.DataFrame", "plot_path": "str"}', description='Output Structure:')`
    *   `param1_name_widget = ipywidgets.Text(value='target_gene', description='Param Name 1:')`
    *   `param1_type_widget = ipywidgets.Dropdown(options=['str', 'int', 'float', 'bool'], value='str', description='Param Type 1:')`
    *   `param1_desc_widget = ipywidgets.Text(value='The gene identifier to analyze.', description='Param Desc 1:')`
    *   `param1_default_widget = ipywidgets.Text(value='EGFR', description='Param Default 1:')`
    (Repeat for a second parameter `tissue_type` or `threshold_value`)
    A function to collect these widget values into an `MCPTool` object once the user is satisfied.
*   **Code Cell (Execution):**
    Display the widgets and capture the current values into a new `user_defined_mcp_tool` object upon interaction.
    `IPython.display.display(interactive_widget_for_tool_definition)`
    `user_defined_mcp_tool = collect_widget_values_into_mcp_tool_func(...)`
    Display `user_defined_mcp_tool`.
*   **Markdown Cell:**
    Illustrates how user inputs through the interactive widgets are captured and transformed into a structured `MCPTool` object, ready for further validation or simulation.

### Section 7: Parameter Validation and Feedback

*   **Markdown Cell:**
    Explains the validation process for the user-defined MCP tool. This crucial step ensures that the tool definition is complete and consistent, highlighting any missing descriptions or invalid data types. This provides instant feedback for refining the tool definition.
*   **Code Cell (Function Definition):**
    Define `validate_mcp_tool_definition(mcp_tool)`:
    This function takes an `MCPTool` object. It checks:
    1.  If `mcp_tool.name`, `mcp_tool.description`, `mcp_tool.output_structure` are non-empty.
    2.  For each `param` in `mcp_tool.input_parameters`:
        *   Checks if `param.name`, `param.type`, `param.description` are non-empty.
        *   Attempts to `eval(param.type)` to ensure it's a valid Python type string (e.g., 'str', 'int').
    It returns a list of validation messages (errors/warnings).
*   **Code Cell (Execution):**
    Execute `validation_messages = validate_mcp_tool_definition(user_defined_mcp_tool)`.
    Print `validation_messages`.
*   **Markdown Cell:**
    Interprets the validation feedback, explaining any identified issues and how they would guide the user in correcting their tool definition.

### Section 8: Natural Language Prompt to Tool Invocation Simulation

*   **Markdown Cell:**
    Describes how an AI agent translates a natural language prompt into a structured tool invocation. This simulation demonstrates the core agent capability of understanding user intent and mapping it to predefined MCP tool parameters.
*   **Code Cell (Function Definition):**
    Define `simulate_prompt_parsing(mcp_tool, prompt_text)`:
    This function takes an `MCPTool` and a `prompt_text`. It iterates through `mcp_tool.input_parameters`. For each parameter, it uses simple `if/elif` and `in` checks (or simple `re.search` if regex library was allowed without explicit import) to find keywords in `prompt_text` and extract values.
    Example: if `param.name` is 'gene_name', it might look for "gene X" or "gene: X".
    It returns a dictionary where keys are parameter names, and values are dictionaries containing `{'extracted_value': value, 'mapped_successfully': bool}`. If a parameter is not found, `extracted_value` is `None` and `mapped_successfully` is `False`.
*   **Code Cell (Execution):**
    Define a `sample_prompt = "Analyze gene expression for gene 'SORT1' in 'liver' tissue with minimum expression 0.05."`.
    Execute `parsed_params = simulate_prompt_parsing(visualize_variant_effects_tool, sample_prompt)`.
    Print `parsed_params`.
*   **Markdown Cell:**
    Explains the simulated parsing results, showing which parameters were successfully extracted from the natural language prompt and which were not.

### Section 9: Parameter Mapping Visualization

*   **Markdown Cell:**
    Visualizes the outcome of the simulated prompt parsing in a clear tabular format. This "Parameter Mapping Visualization" helps users understand how the AI agent interpreted their natural language input and assigned values to the tool's parameters.
*   **Code Cell (Function Definition):**
    Define `visualize_parameter_mapping(mapping_results, mcp_tool)`:
    This function takes `mapping_results` (from `simulate_prompt_parsing`) and the original `mcp_tool`. It creates a `pandas.DataFrame` with columns: `Parameter Name`, `Expected Type`, `Extracted Value`, `Mapped Successfully`, `Description`, `Default Value`. It populates this DataFrame and displays it.
*   **Code Cell (Execution):**
    Execute `visualize_parameter_mapping(parsed_params, visualize_variant_effects_tool)`.
*   **Markdown Cell:**
    Interprets the displayed table, explaining how the `Extracted Value` for each parameter aligns (or doesn't align) with the `Expected Type` and `Default Value`.

### Section 10: Generating a Python Code Snippet for the Tool

*   **Markdown Cell:**
    Demonstrates how the defined MCP tool structure translates into a Python code snippet, illustrating its encapsulation for use within an agent framework. This "Code Snippet Generation" feature helps developers understand the implementation side of MCP tools.
*   **Code Cell (Function Definition):**
    Define `generate_mcp_tool_snippet(mcp_tool)`:
    This function constructs a multi-line string representing a Python function.
    It will start with `@mcp.tools` (as a placeholder decorator), define a function signature using `mcp_tool.name` and parameters from `mcp_tool.input_parameters` (including type hints and default values). It will also include a docstring generated from `mcp_tool.description`.
*   **Code Cell (Execution):**
    Execute `code_snippet = generate_mcp_tool_snippet(visualize_variant_effects_tool)`.
    Display `code_snippet` using `IPython.display.Markdown(f"```python\n{code_snippet}\n```")`.
*   **Markdown Cell:**
    Explains the generated Python code, detailing how the tool's definition forms the function signature and docstring, making it agent-interpretable.

### Section 11: Implementing a Simulated MCP Tool Function (Execution)

*   **Markdown Cell:**
    Explains the simulation of the actual tool execution. In a real scenario, this would involve complex analysis. Here, a simplified function demonstrates how the tool would use the parameters extracted from the natural language prompt to process the synthetic data.
*   **Code Cell (Function Definition):**
    Define `simulate_tool_execution(data, tool_parameters)`:
    This function takes the `synthetic_df` and the `tool_parameters` (after parsing). It performs a dummy operation: filters `data` based on `tool_parameters['gene_name']['extracted_value']`, `tool_parameters['tissue_type']['extracted_value']`, and `tool_parameters['min_expression_level']['extracted_value']` (if mapped successfully). It returns a dictionary containing a 'filtered_dataframe_head' and 'number_of_filtered_rows'.
*   **Code Cell (Execution):**
    Execute `simulated_results = simulate_tool_execution(synthetic_df, parsed_params)`.
    Print `simulated_results['number_of_filtered_rows']` and display `simulated_results['filtered_dataframe_head']`.
*   **Markdown Cell:**
    Interprets the simulated tool execution, showing how the input parameters influenced the dummy analysis and what hypothetical results were produced.

### Section 12: Core Visualization 1: Trend Plot (Time-based metrics)

*   **Markdown Cell:**
    Introduces the first core visualization: a trend plot for time-based metrics. This plot helps identify patterns and changes in data over time, crucial for understanding dynamic processes.
*   **Code Cell (Function Definition):**
    Define `plot_time_series(dataframe, time_col, value_col, title, x_label, y_label, color_palette='viridis', save_path=None)`:
    This function uses `seaborn.lineplot` to create a line plot. It ensures `time_col` is a datetime type. It saves a static PNG to `save_path` if provided. It uses a color-blind friendly palette.
*   **Code Cell (Execution):**
    Execute `plot_time_series(synthetic_df, 'timestamp', 'gene_expression_level', 'Gene Expression Over Time', 'Date', 'Expression Level')`.
    Include a fallback mechanism to save as PNG: `plot_time_series(synthetic_df, 'timestamp', 'gene_expression_level', 'Gene Expression Over Time', 'Date', 'Expression Level', save_path='trend_plot.png')`.
*   **Markdown Cell:**
    Explains the observed trend in the gene expression level over time, discussing any visible patterns or fluctuations.

### Section 13: Core Visualization 2: Relationship Plot (Scatter)

*   **Markdown Cell:**
    Presents a relationship plot, specifically a scatter plot, designed to explore correlations and distributions between two continuous variables, with an optional categorical variable for grouping.
*   **Code Cell (Function Definition):**
    Define `plot_relationship(dataframe, x_col, y_col, hue_col=None, title, x_label, y_label, color_palette='viridis', save_path=None)`:
    This function uses `seaborn.scatterplot`. It allows an optional `hue_col` for coloring points by category. It saves a static PNG to `save_path` if provided. It uses a color-blind friendly palette.
*   **Code Cell (Execution):**
    Execute `plot_relationship(synthetic_df, 'numeric_feature_A', 'numeric_feature_B', 'tissue_type', 'Relationship between Feature A and B by Tissue Type', 'Feature A', 'Feature B')`.
    Include a fallback mechanism: `plot_relationship(synthetic_df, 'numeric_feature_A', 'numeric_feature_B', 'tissue_type', 'Relationship between Feature A and B by Tissue Type', 'Feature A', 'Feature B', save_path='relationship_plot.png')`.
*   **Markdown Cell:**
    Analyzes the scatter plot, describing any observed relationships between the numeric features and how the categorical `tissue_type` influences this relationship.

### Section 14: Core Visualization 3: Aggregated Comparison (Bar Plot)

*   **Markdown Cell:**
    Introduces an aggregated comparison plot, typically a bar chart, to visualize and compare a numeric metric across different categories. This is useful for gaining insights into group differences.
*   **Code Cell (Function Definition):**
    Define `plot_categorical_comparison(dataframe, category_col, value_col, aggregation_func='mean', title, x_label, y_label, color_palette='viridis', save_path=None)`:
    This function first aggregates the `dataframe` by `category_col` using `aggregation_func` on `value_col`. Then, it uses `seaborn.barplot` to plot the results. It saves a static PNG to `save_path` if provided. It uses a color-blind friendly palette.
*   **Code Cell (Execution):**
    Execute `plot_categorical_comparison(synthetic_df, 'tissue_type', 'gene_expression_level', 'mean', 'Average Gene Expression by Tissue Type', 'Tissue Type', 'Average Expression')`.
    Include a fallback mechanism: `plot_categorical_comparison(synthetic_df, 'tissue_type', 'gene_expression_level', 'mean', 'Average Gene Expression by Tissue Type', 'Tissue Type', 'Average Expression', save_path='comparison_plot.png')`.
*   **Markdown Cell:**
    Interprets the bar plot, highlighting significant differences in average gene expression levels across various tissue types.

### Section 15: Enabling User Interaction for Visualizations

*   **Markdown Cell:**
    Explains how interactive widgets can be integrated with the visualization functions to allow users to dynamically adjust plot parameters (e.g., selecting columns, aggregation methods, titles). This enhances usability and enables deeper exploration.
*   **Code Cell (Function Definition & Widget Setup):**
    Define `make_interactive_plot_time_series(dataframe)`:
    This function wraps `plot_time_series` with `ipywidgets.interactive`. It creates `ipywidgets.Dropdown` for `value_col` selection (from `dataframe.columns` of numeric type), and `ipywidgets.Text` for `title`, `x_label`, `y_label`. It includes inline help text for each control.
    Similarly, define `make_interactive_plot_relationship(dataframe)` and `make_interactive_plot_comparison(dataframe)`.
*   **Code Cell (Execution):**
    Execute `interactive_time_series_plot = make_interactive_plot_time_series(synthetic_df)` and `IPython.display.display(interactive_time_series_plot)`.
    Repeat for `interactive_relationship_plot` and `interactive_comparison_plot`.
*   **Markdown Cell:**
    Demonstrates the interactive controls, showing how changing widget values instantly updates the corresponding plot, facilitating exploratory data analysis.

### Section 16: Conclusion

*   **Markdown Cell:**
    Summarizes the key concepts learned throughout the notebook, reiterating the importance of well-defined MCP tools for robust AI agent interaction. It emphasizes how the Paper2Agent framework, by converting research papers into interactive agents, streamlines scientific workflows, enhances reproducibility, and lowers barriers to adopting new methodologies. This lab provided a practical understanding of how `MCP_tools` and `MCP_prompts` work together to enable natural language querying and autonomous execution through an AI agent.

### Section 17: References

*   **Markdown Cell:**
    Lists the key references that underpin the concepts explored in this notebook.
    *   Paper2Agent: Xinyi Hou, Yanjie Zhao, Shenao Wang, and Haoyu Wang. Model context protocol (mcp): Landscape, security threats, and future research directions. arXiv preprint arXiv:2503.23278, 2025.
    *   `pandas`: The pandas development team. (2020). pandas-dev/pandas: Pandas. Zenodo. DOI: 10.5281/zenodo.3509134
    *   `numpy`: Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2
    *   `matplotlib`: Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90-95.
    *   `seaborn`: Waskom, M. L. (2021). seaborn: statistical data visualization. Journal of Open Source Software, 6(60), 3021.
    *   `ipywidgets`: Bussonnier, Matthias, et al. (2018). Jupyter Widgets: Interactive Controls for Jupyter Notebooks. The Journal of Open Source Software, 3(22), 614.

