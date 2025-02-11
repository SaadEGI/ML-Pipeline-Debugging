import csv
import dspy
import os
from typing import Literal
import pandas as pd
from dotenv import load_dotenv
from dspy import Evaluate
from matplotlib import pyplot as plt

load_dotenv(dotenv_path="../.env")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_r1_api_key = os.getenv("OPENROUTER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

lm = dspy.LM('openai/deepseek-chat', api_key=deepseek_api_key, api_base="https://api.deepseek.com")

deepseek_r1 = dspy.LM('openai/deepseek/deepseek-r1', api_key=deepseek_r1_api_key,
                      api_base="https://openrouter.ai/api/v1")

issue_names = []
issue_descriptions = []
issues_impact = []
pipeline_phases = []

with open("../example_pipelines2/issues_names.txt", "r") as file:
    for line in file:
        issue_names.append(line.strip())

with open("../example_pipelines2/issues_descriptions.txt", "r") as file:
    for line in file:
        issue_descriptions.append(line.strip())

with open("../example_pipelines2/issues_impact.txt", "r") as file:
    for line in file:
        issues_impact.append(line.strip())

with open("../example_pipelines2/pipeline_phase.txt", "r") as file:
    for line in file:
        pipeline_phases.append(line.strip())

pipelines = []
for root, dirs, files in os.walk("../example_pipelines2"):
    for file in files:
        if file == "example-0.py":
            print("True")
            pipelines.append(os.path.join(root, file))

dags = []
for root, dirs, files in os.walk("../example_pipelines2"):
    for file in files:
        if file == "dag.json":
            dags.append(os.path.join(root, file))


def read_code_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            code = file.read()
        return code
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return ""
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


data = []
for pipeline in pipelines:
    code = read_code_from_file(pipeline)
    data.append({"pipeline_code": code})

for i, issue in enumerate(issue_names):
    data[i]["issue_name"] = issue
    data[i]["dag"] = dags[i]

for i, issue in enumerate(issues_impact):
    data[i]["issue_impact"] = issue

for i, issue in enumerate(pipeline_phases):
    data[i]["pipeline_phase"] = issue

df = pd.DataFrame(data)


class CodeIssue(dspy.Signature):
    """
   Analyze Python code to check if a specific issue is present.

   Inputs:
     - code: The Python code to analyze.
     - dag: The DAG of the pipeline.
     - issue_name: The name of the issue to check.
     - potential_pipeline_phase: The potential pipeline phase.
     - issue_impact: The impact of the issue.

   Outputs:
     - issue_detected: A boolean indicating if the issue was found.
     - confidence: A float indicating the confidence in the result.
   """
    code: str = dspy.InputField()
    dag: str = dspy.InputField()
    issue_name: str = dspy.InputField(desc="Name of the issue to check")
    potential_pipeline_phase: str = dspy.InputField(desc="The pipeline phase")
    issue_impact: str = dspy.InputField(desc="The impact of the issue")

    issue_detected: bool = dspy.OutputField()
    confidence: float = dspy.OutputField()


classify_code_issue = dspy.Predict(CodeIssue)


# Here is how we call this module
def classify_with_single_model(df, model_name_index):
    if model_name_index == 0:
        lm = dspy.LM('openai/deepseek-chat', api_key=deepseek_api_key, api_base="https://api.deepseek.com")
        model_name = "deepseek_v3"
    elif model_name_index == 99:
        lm = dspy.LM('openai/deepseek/deepseek-r1', api_key=deepseek_r1_api_key,
                     api_base="https://openrouter.ai/api/v1")
        model_name = "deepseek_r1"
    elif model_name_index == 1:
        lm = dspy.LM('openai/deepseek-reasoner', api_key=deepseek_api_key, api_base="https://api.deepseek.com")
        model_name = "deepseek_r1_reasoner"
    elif model_name_index == 2:
        lm = dspy.LM('openai/gpt-4o-mini', api_key=openai_api_key)
        model_name = "gpt-4o-mini"
    elif model_name_index == 3:
        lm = dspy.LM('gemini/gemini-2.0-flash-001', api_key=gemini_api_key)
        model_name = "gemini-2.0-flash-001"
    elif model_name_index == 4:
        lm = dspy.LM('openai/gpt-4o', api_key=openai_api_key)
        model_name = "gpt-4o"

    dspy.configure(lm=lm)

    results = []
    for index, row in df.iterrows():
        code = row['pipeline_code']
        dag = row['dag']
        issue_to_look_for = row['issue_name']
        pipeline_phase = row['pipeline_phase']
        issue_impact = row['issue_impact']
        classification = classify_code_issue(code=code, issue_name=issue_to_look_for,
                                             potential_pipeline_phase=pipeline_phase, issue_impact=issue_impact,
                                             dag=dag)
        print(f"Model: {model_name}, Result: {classification}")
        results.append({
            "pipeline_code": pipelines[index],
            "model": model_name,
            "detected": classification.issue_detected,
            "confidence": classification.confidence
        })
    # save the results to a file
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{model_name}_results.csv", index=False)
    return pd.DataFrame(results)


#
#
# classify_with_single_model(df, 0)
# classify_with_single_model(df, 1)
classify_with_single_model(df, 2)
classify_with_single_model(df, 3)
classify_with_single_model(df, 4)

# Visualisation

visualization_data = []
models = ["deepseek_v3", "deepseek_r1_reasoner", "gpt-4o-mini", "gemini-2.0-flash-001", "gpt-4o"]
for model in models:
    # if file exists
    if os.path.isfile(f"{model}_results.csv"):
        df = pd.read_csv(f"{model}_results.csv")
        for index, row in df.iterrows():
            visualization_data.append({
                "model": model,
                "pipeline_index": index,
                "issue_detected": row['detected'],
                "confidence": row['confidence']
            })

print(visualization_data)
df_visualization = pd.DataFrame(visualization_data)
print(df_visualization)
df_visualization['index'] = df_visualization.index
df_visualization['color'] = df_visualization['issue_detected'].map({True: 'green', False: 'red'})

# Create subplots: one for each model.
models = df_visualization['model'].unique()
fig, axes = plt.subplots(1, len(models), figsize=(14, 6), sharey=True)

for ax, model in zip(axes, models):
    subset = df_visualization[df_visualization['model'] == model]
    ax.scatter(subset['pipeline_index'], subset['confidence'], color=subset['color'], s=100, edgecolor='k')
    ax.set_title(model)
    ax.set_xlabel("Pipeline Index")
    ax.set_ylim(0, 1.1)
    ax.grid(True)

axes[0].set_ylabel("Confidence")
plt.suptitle("Prediction Confidence per Model")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
