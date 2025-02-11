import os

import dspy
import pandas as pd
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from dspy_approach.utils import read_code_from_file

load_dotenv(dotenv_path="../../.env")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_r1_api_key = os.getenv("OPENROUTER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")

lm = dspy.LM('openai/deepseek-chat', api_key=deepseek_api_key, api_base="https://api.deepseek.com")

deepseek_r1 = dspy.LM('openai/deepseek/deepseek-r1', api_key=deepseek_r1_api_key,
                      api_base="https://openrouter.ai/api/v1")

issue_names = []
issue_descriptions = []

with open("../../example_pipelines2/issues_names.txt", "r") as file:
    for line in file:
        issue_names.append(line.strip())


pipelines = []
for root, dirs, files in os.walk("../../corrected_pipelines"):
    for file in files:

        if file == "example-0-fixed.py":
            pipelines.append(os.path.join(root, file))

data = []
for pipeline in pipelines:
    code = read_code_from_file(pipeline)
    data.append({"pipeline_code": code})

for i, issue in enumerate(issue_names):
    data[i]["issue_name"] = issue

df = pd.DataFrame(data)


class CodeIssue(dspy.Signature):
    """
   Analyze Python code to check if a specific issue is present.

   Inputs:
     - code: The Python code to analyze.
     - issue_name: The name of the issue to check.

   Outputs:
     - issue_detected: A boolean indicating if the issue was found.
     - confidence: A float indicating the confidence in the result.
   """
    code: str = dspy.InputField()
    issue_name: str = dspy.InputField(desc="Name of the issue to check")
    hint: str = dspy.InputField()

    issue_detected: bool = dspy.OutputField()
    confidence: float = dspy.OutputField()


classify_code_issue = dspy.Predict(CodeIssue)


# Here is how we call this module
def classify_with_single_model(df, model_name_index):
    if model_name_index == 0:
        lm = dspy.LM('openai/deepseek-ai/DeepSeek-V3', api_key=together_api_key,
                     api_base="https://api.together.xyz/v1", cache=False)
        model_name = "deepseek_v3"
    elif model_name_index == 99:
        lm = dspy.LM('openai/deepseek/deepseek-r1', api_key=deepseek_r1_api_key,
                     api_base="https://openrouter.ai/api/v1", cache=False)
        model_name = "deepseek_r1"
    elif model_name_index == 1:
        lm = dspy.LM('openai/deepseek-reasoner', api_key=deepseek_api_key, api_base="https://api.deepseek.com", cache=False)
        model_name = "deepseek_r1_reasoner"
    elif model_name_index == 2:
        lm = dspy.LM('openai/gpt-4o-mini', api_key=openai_api_key, cache=False)
        model_name = "gpt-4o-mini"
    elif model_name_index == 3:
        lm = dspy.LM('gemini/gemini-2.0-flash-001', api_key=gemini_api_key, cache=False)
        model_name = "gemini-2.0-flash-001"
    elif model_name_index == 4:
        lm = dspy.LM('openai/gpt-4o', api_key=openai_api_key, cache=False)
        model_name = "gpt-4o"

    dspy.configure(lm=lm)

    results = []
    for index, row in df.iterrows():
        code = row['pipeline_code']
        issue_to_look_for = row['issue_name']
        print(f"Looking for issue: {issue_to_look_for}")
        print(pipelines[index])
        classification = classify_code_issue(code=code, issue_name=issue_to_look_for, hint="it is possible that the code pipeline does not contain the issue.")
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
classify_with_single_model(df, 0)
#classify_with_single_model(df, 1)
#classify_with_single_model(df, 2)
#classify_with_single_model(df, 3)
#classify_with_single_model(df, 4)

# Visualisation
print(dspy.inspect_history())

visualization_data = []
models = ["deepseek_v3", "deepseek_r1_reasoner", "gpt-4o-mini", "gemini-2.0-flash-001", "gpt-4o"]
for model in models:
    # if file exists
    if os.path.isfile(f"{model}_results.csv"):
        df = pd.read_csv(f"{model}_results.csv")
        for index, row in df.iterrows():
            visualization_data.append({
                "model": model,
                "pipeline_index" : index,
                "issue_detected": row['detected'],
                "confidence": row['confidence']
            })

print(visualization_data)
df_visualization = pd.DataFrame(visualization_data)
print(df_visualization)
df_visualization['index'] = df_visualization.index
df_visualization['color'] = df_visualization['issue_detected'].map({True: 'red', False: 'green'})

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
plt.suptitle("Prediction per Model")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

