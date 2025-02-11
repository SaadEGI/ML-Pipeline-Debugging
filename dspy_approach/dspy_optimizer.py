# Import the optimizer
import csv
import os
from typing import Literal

import dspy
import pandas as pd
from dotenv import load_dotenv
from dspy import Example

from dspy_class_list_approach import read_code_from_file

load_dotenv(dotenv_path="../.env")
gemini_api_key = os.getenv("GEMINI_API_KEY")
# Configure LM
# lm = dspy.LM('openai/deepseek-chat', api_key=deepseek_api_key, api_base="https://api.deepseek.com")
# lm = dspy.LM('gemini/gemini-2.0-flash-001', api_key=gemini_api_key)
# dspy.configure(lm=lm)

pipelines = []
for root, dirs, files in os.walk("../example_pipelines2"):
    for file in files:
        if file == "example-0.py":
            pipelines.append(os.path.join(root, file))

corrected_pipelines = []
for root, dirs, files in os.walk("../example_pipelines"):
    for file in files:
        if file == "example-0-fixed.py":
            corrected_pipelines.append(os.path.join(root, file))

correct_data = []
for pipeline in corrected_pipelines:
    code = read_code_from_file(pipeline)
    correct_data.append({"pipeline_code": code})

df_correct = pd.DataFrame(correct_data)
data = []
for pipeline in pipelines:
    code = read_code_from_file(pipeline)
    data.append({"pipeline_code": code})

df = pd.DataFrame(data)

# Define the ground truth labels
ground_truth = []
with open("../example_pipelines2/issues_names.txt", "r") as file:
    for line in file:
        ground_truth.append(line.strip())

ground_truth_description = []
with open("../example_pipelines2/issues_descriptions.txt", "r") as file:
    for line in file:
        ground_truth_description.append(line.strip())


def evaluate_model_predictions(
        ground_truth: dict,
        model_name: int,
) -> float:
    if model_name == 0:
        model_name = "deepseek_v3"
    elif model_name == 1:
        model_name = "deepseek_r1"
    elif model_name == 2:
        model_name = "deepseek_r1_reasoner"
    elif model_name == 3:
        model_name = "gpt-4o-mini"
    elif model_name == 4:
        model_name = "gemini-2.0-flash-001"

    predictions = []
    with open(f"{model_name}_results.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            predictions.append(row['category'])

    if len(ground_truth) != len(predictions):
        raise ValueError("The number of ground truth labels and predictions do not match.")

    # Compare predictions with ground truth
    correct = 0
    for i in range(len(ground_truth)):
        if ground_truth[i] == predictions[i]:
            correct += 1

    # Calculate accuracy
    accuracy = correct / len(ground_truth)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Optional: Save the comparison results to a new CSV file
    results = []
    for i in range(len(ground_truth)):
        results.append({
            "pipeline_index": i,
            "ground_truth": ground_truth[i],
            "prediction": predictions[i],
            "correct": ground_truth[i] == predictions[i]
        })

    results_df = pd.DataFrame(results)
    output_file = f"evaluation_results_{model_name}.csv"
    results_df.to_csv(output_file, index=False)
    print("Evaluation results saved to evaluation_results.csv")
    return accuracy


trainset = []
for i in range(len(df)):
    trainset.append(Example(code=df.iloc[i]["pipeline_code"], category=ground_truth[i], hint=ground_truth_description[i]).with_inputs("code"))

for i in range(len(df_correct)):
    trainset.append(Example(code=df_correct.iloc[i]["pipeline_code"], category="no_error", hint="This function is well-defined and should not trigger an issue.").with_inputs("code"))

print(trainset)



class CodeIssueSignature(dspy.Signature):
    """
    Signature for classifying code issues.
    The input field is 'code' and the expected outputs are 'category' (a Literal over issue_names)
    and 'confidence' (a float score).
    """
    code: str = dspy.InputField()
    category: Literal[tuple(ground_truth)] = dspy.OutputField()
    confidence: float = dspy.OutputField()


# Create the classification module that uses chain-of-thought with hint support.
classify_code_issue_with_hint = dspy.ChainOfThoughtWithHint(CodeIssueSignature)


def metric(prediction, ground_truth, trace=None):
    # Compare the predicted category to the true category.
    return prediction.category == ground_truth.category


# For example, if you want to use GPT-4o-mini:
dspy.configure(lm=dspy.LM('gemini/gemini-2.0-flash-001', api_key=gemini_api_key))
dspy.configure(experimental=True)

# Create the BootstrapFinetune optimizer.
optimizer = dspy.BootstrapFinetune(metric=metric, num_threads=4)

# Compile (i.e. finetune) your module using your training set.
optimized_classifier = optimizer.compile(classify_code_issue_with_hint, trainset=trainset)
