import os

import dspy

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")


lm = dspy.LM('openai/deepseek-chat', api_key=deepseek_api_key, api_base="https://api.deepseek.com")
dspy.configure(lm=lm)

from typing import List


class AnalyzeCodeIssue(dspy.Signature):
    """
    Analyze Python code to check if a specific issue is present.

    Inputs:
      - code: The Python code (with line numbers) to analyze.
      - dag: Optional JSON representing the DAG (can be empty if not available).
      - issue_name: The name of the issue to check.
      - issue_description: A description of the issue (optional).

    Outputs:
      - issue_detected: A boolean indicating if the issue was found.
      - affected_lines: A list of line numbers affected by the issue.
    """
    code: str = dspy.InputField(desc="Machine Pipeline Python Code")
    #dag: str = dspy.InputField(desc=" JSON of the DAG of the pipeline code", default="")
    issue_name: str = dspy.InputField(desc="Name of the issue to check")
    #issue_description: str = dspy.InputField(desc="Detailed issue description", default="")

    issue_detected: bool = dspy.OutputField()
    affected_code_blocks: List[int] = dspy.OutputField(desc="affected code blocks")
    reason: str = dspy.OutputField(desc="reason for the issue")


# Example input data
with open("example_pipelines/aggregation_errors/example-0.py", "r") as file:
    code_1 = file.read()

with open("corrected_pipelines/example-0-fixed.py", "r") as file:
    code = file.read()

dag_json = '{"nodes": ["foo", "bar"], "edges": [["bar", "foo"]]}'
issue_name = "aggregation error"
issue_description = "The code might contain unreachable or unused sections."

analyze_module = dspy.ChainOfThought(AnalyzeCodeIssue)

# Invoke the module.
result = analyze_module(
    code=code,
    #dag=dag_json,
    issue_name=issue_name,
    #issue_description=issue_description
)

print("Issue detected:", result.issue_detected)
print("Affected code blocks:", result.affected_code_blocks)
print("Reason:", result.reason)
