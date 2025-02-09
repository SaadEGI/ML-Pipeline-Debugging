import json
import random
import pandas as pd


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def add_line_numbers(code):
    """
    Add line numbers to the provided code.
    """
    lines = code.split('\n')
    num_width = len(str(len(lines)))
    numbered_lines = [f"{i + 1:>{num_width}}| {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)


def build_prompt(code, issue_name, issue_description_full, additional_context=None):
    """
    Build a prompt to instruct the LLM to analyze the given ML pipeline code.
    The prompt includes the code with line numbers, the issue name/description, and
    any additional context (e.g., column names, DAG structure, aggregated statistics).
    """
    numbered_code = add_line_numbers(code)
    base_prompt = (
        f"Analyze the following Python ML pipeline code (with line numbers) and detect if the issue of "
        f"{issue_name.lower()} is present. Return the affected line numbers in JSON format.\n\n"
        f"Issue description: {issue_description_full}\n\n"
        f"Code:\n```python\n{numbered_code}\n```"
    )
    if additional_context:
        base_prompt += f"\n\nAdditional Context:\n{additional_context}"
    return base_prompt


def simulate_llm_response(prompt):
    """
    Simulate an LLM response for demonstration purposes.

    For this example, we use a simple heuristic: if the prompt contains the keyword
    'aggregation error' and the code includes a special marker (e.g., "# AGG_ERROR")
    then we return that the issue is detected; otherwise, not.

    In practice, this function would call an API like OpenAI's chat completions.
    """
    # For simulation, check if "aggregation error" is mentioned and a marker exists.
    issue_detected = False
    affected_lines = []

    # Simple heuristic: if prompt contains marker for aggregation error, flag it.
    if "aggregation error" in prompt.lower():
        if "# AGG_ERROR" in prompt:
            issue_detected = True
            # Simulate detecting the marker on a specific line (e.g., line 3 and 4)
            affected_lines = [3, 4]
        else:
            # Random chance if not clearly marked.
            issue_detected = random.choice([True, False])
            affected_lines = [random.randint(1, 7)] if issue_detected else []
    else:
        # For other issues, simulate a 50/50 chance.
        issue_detected = random.choice([True, False])
        affected_lines = [random.randint(1, 7)] if issue_detected else []

    # Build a JSON-like response (as a string) similar to what an LLM might return.
    response = {
        "issue_detected": issue_detected,
        "affected_lines": affected_lines
    }
    # For our simulation, return the JSON string.
    return json.dumps(response)


# -----------------------------------------------------------------------------
# Experiment Setup: Pipeline Examples and Prompt Configurations
# -----------------------------------------------------------------------------

# Define a list of example pipelines.
# For simplicity, each pipeline is represented as a dictionary containing:
# - id: a unique identifier
# - code: a string of Python code
# - is_faulty: a boolean flag (faulty pipelines include a marker for error)
# - ground_truth: expected result for issue detection (for later comparison)
pipelines = [
    {
        "id": "pipeline_1_correct",
        "code": (
            "import pandas as pd\n"
            "df = pd.read_csv('data.csv')\n"
            "X = df[['feature1', 'feature2']]\n"
            "y = df['target']\n"
            "model = DecisionTreeClassifier()\n"
            "model.fit(X, y)\n"
            "predictions = model.predict(X)"
        ),
        "is_faulty": False,
        "ground_truth": {"issue_detected": False, "affected_lines": []}
    },
    {
        "id": "pipeline_2_faulty",
        "code": (
            "import pandas as pd\n"
            "df = pd.read_csv('data.csv')\n"
            "# AGG_ERROR: Incorrect aggregation operation below\n"
            "agg_result = df.groupby('category').sum()\n"
            "X = agg_result[['feature1']]\n"
            "y = agg_result['target']\n"
            "model = DecisionTreeClassifier()\n"
            "model.fit(X, y)\n"
            "predictions = model.predict(X)"
        ),
        "is_faulty": True,
        "ground_truth": {"issue_detected": True, "affected_lines": [3, 4]}
    },
    {
        "id": "pipeline_3_correct",
        "code": (
            "import pandas as pd\n"
            "df = pd.read_csv('data.csv')\n"
            "df['new_feature'] = df['feature1'] * 2\n"
            "X = df[['new_feature', 'feature2']]\n"
            "y = df['target']\n"
            "model = DecisionTreeClassifier()\n"
            "model.fit(X, y)\n"
            "predictions = model.predict(X)"
        ),
        "is_faulty": False,
        "ground_truth": {"issue_detected": False, "affected_lines": []}
    },
    {
        "id": "pipeline_4_faulty",
        "code": (
            "import pandas as pd\n"
            "df = pd.read_csv('data.csv')\n"
            "# AGG_ERROR: Problem in aggregation below\n"
            "agg = df[['feature1', 'feature2']].mean()\n"
            "X = df[['feature1']]\n"
            "y = df['target']\n"
            "model = DecisionTreeClassifier()\n"
            "model.fit(X, y)\n"
            "predictions = model.predict(X)"
        ),
        "is_faulty": True,
        "ground_truth": {"issue_detected": True, "affected_lines": [3, 4]}
    }
]

# Define prompt configurations.
# For each configuration, we provide additional context if desired.
prompt_configurations = {
    "Code Only": None,
    "Code + Column Names": "Column names: ['feature1', 'feature2', 'target']",
    "Code + DAG": "DAG structure: ReadCSV -> ProcessData -> ModelTraining -> Prediction",
    "Code + Aggregated Statistics": "Aggregated stats: mean(feature1)=0.5, std(feature1)=0.1, mean(feature2)=0.7, std(feature2)=0.15"
}

# Define the issue details we are looking for.
issue_name = "Aggregation Error"
issue_description_full = (
    "Aggregation errors occur when data is incorrectly summarized, for example by applying a group-by "
    "operation improperly or missing critical steps in the aggregation pipeline."
)

# -----------------------------------------------------------------------------
# Run Experiment: Build Prompts, Simulate LLM Calls, and Record Results
# -----------------------------------------------------------------------------

results = []  # To store each experiment result

for pipeline in pipelines:
    code = pipeline["code"]
    pipeline_id = pipeline["id"]
    ground_truth = pipeline["ground_truth"]

    for config_name, additional_context in prompt_configurations.items():
        # Build the prompt using our helper function
        prompt = build_prompt(code, issue_name, issue_description_full, additional_context)

        # Simulate calling the LLM with this prompt.
        llm_response_str = simulate_llm_response(prompt)
        try:
            llm_response = json.loads(llm_response_str)
        except json.JSONDecodeError:
            llm_response = {"issue_detected": None, "affected_lines": []}

        # Record the result along with metadata.
        result_entry = {
            "pipeline_id": pipeline_id,
            "configuration": config_name,
            "ground_truth_issue": ground_truth["issue_detected"],
            "ground_truth_lines": ground_truth["affected_lines"],
            "llm_issue_detected": llm_response.get("issue_detected"),
            "llm_affected_lines": llm_response.get("affected_lines"),
            "prompt_snippet": prompt[:100]  # store the first 100 chars of the prompt for reference
        }
        results.append(result_entry)

# Create a DataFrame (experiment matrix) from the results.
df_results = pd.DataFrame(results)

# Reorder and display columns for clarity.
df_results = df_results[[
    "pipeline_id", "configuration",
    "ground_truth_issue", "ground_truth_lines",
    "llm_issue_detected", "llm_affected_lines",
    "prompt_snippet"
]]

print("Experiment Matrix (each row represents one prompt configuration for a pipeline):")
print(df_results.to_string(index=False))

# Optionally, save the matrix to a CSV file.
df_results.to_csv("experiment_matrix.csv", index=False)
