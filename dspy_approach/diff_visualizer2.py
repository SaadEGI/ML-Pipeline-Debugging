import csv

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.table as mpltbl

import os
import matplotlib.colors as colors  # For normalization and colormap
from model_inspection import read_code_from_file

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
        path: str
) -> float:
    if model_name == 0:
        model_name = "deepseek_v3"
    elif model_name == 1:
        model_name = "deepseek_r1_reasoner"
    elif model_name == 2:
        model_name = "gpt-4o-mini"
    elif model_name == 3:
        model_name = "gemini-2.0-flash-001"
    elif model_name == 4:
        model_name = "gpt-4o"

    predictions = []
    full_path = f"{path}/{model_name}_results.csv"
    if os.path.exists(full_path):
        with open(f"{path}/{model_name}_results.csv", "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                predictions.append((row["detected"], float(row["confidence"])))
    if len(ground_truth) != len(predictions):
        return 0.0, 0
    print(predictions)
    correct = 0
    confidence_average = 0
    for i in range(len(ground_truth)):
        if predictions[i][0] == 'True':
            correct += 1
            confidence_average += predictions[i][1]
    if correct != 0:
        confidence_average /= correct

    confidence_average = round(confidence_average, 2)
    print(correct)
    # Calculate accuracy
    accuracy = correct / len(ground_truth)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("Evaluation results saved to evaluation_results.csv")
    return accuracy, confidence_average



models = ['gpt-4o-mini', 'gemini-2.0-flash-001', 'gpt-4o']
approach_names = ['Approach 1', 'Approach 2', 'Approach 3', 'Approach 4']
approaches_paths = [
    "results_issue_input_code",
    "results_issue_input_code_dag",
    "results_issue_input_data_code",
    "results_issue_input_data_code_dag"
]
# Collect evaluation results for all models and approaches
data = []
for model_name in models:
    row = {'Model': model_name}
    for approach_name, approach_path in zip(approach_names, approaches_paths):
        accuracy, confidence_average = evaluate_model_predictions(ground_truth, model_name, approach_path)
        # Combine accuracy and confidence average into one string
        row[approach_name] = f"  Acc: {accuracy * 100:.2f}%\nConf: {confidence_average:.2f}"
    data.append(row)

# Create DataFrame
df = pd.DataFrame(data)
df.set_index('Model', inplace=True)

def plot_table(df, title=""):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('tight')
    ax.axis('off')

    # Create table from DataFrame
    table = mpltbl.table(ax, cellText=df.values, colLabels=df.columns, rowLabels=df.index,
                         cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(0.8, 1.5)

    # Set table styles
    for (row, col), cell in table.get_celld().items():
        if (row == 0) or (col == -1):  # Header cells
            cell.set_text_props(weight='bold', color='black')
        else:
            cell.set_text_props(wrap=True)

    plt.title(title)
    plt.show()

def plot_table_to_image(df, filename="table.png", title=""):
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.axis('tight')
    ax.axis('off')

    # Create table from DataFrame
    table = mpltbl.table(ax, cellText=df.values, colLabels=df.columns, rowLabels=df.index,
                         cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(0.8, 3)

    # Set table styles
    for (row, col), cell in table.get_celld().items():
        if (row == 0) or (col == -1):  # Header cells
            cell.set_text_props(weight='bold', color='black')
        else:
            cell.set_text_props(wrap=True)

    plt.title(title)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
plot_table(df, title="Approaches")

# Save the table as an image
plot_table_to_image(df, filename="model_comparison.png", title="Model Evaluation Comparison")
