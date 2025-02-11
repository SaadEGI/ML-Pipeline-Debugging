import csv

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.table as mpltbl
import matplotlib.colors as colors
import os

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

print(ground_truth)
ground_truth_description = []
with open("../example_pipelines2/issues_descriptions.txt", "r") as file:
    for line in file:
        ground_truth_description.append(line.strip())

models = ['gpt-4o-mini', 'gemini-2.0-flash-001', 'gpt-4o']
issues = ['Cross Validation Errors', 'Data Anonymization Errors', 'Data Filtering Errors', 'Data Imputation Errors',
          'Data Leakage Errors', 'Data Slicing Errors', 'Data Splitting Errors', 'Specification Bias']
approaches_paths = [
    "results_issue_input_code",
    "results_issue_input_code_dag",
    "results_issue_input_data_code",
    "results_issue_input_data_code_dag"
]
approach_names = ['Approach 1', 'Approach 2', 'Approach 3', 'Approach 4']

# Collect evaluation results for all models and approaches

data_issues_models = []


def evaluate_model_predictions_issues(model):
    issues_res = []

    for issue_index, issue in enumerate(ground_truth):
        issue_dict = [[0,0,0] for _ in range(len(approaches_paths))]
        for approach_path_index, approach_path in enumerate(approaches_paths):
            for model_index, model_name in enumerate(models):

                full_path = f"{approach_path}/{model_name}_results.csv"
                print(full_path)

                if os.path.exists(full_path):
                    with open(f"{approach_path}/{model_name}_results.csv", "r") as file:
                        reader = csv.DictReader(file)
                        # for the corresponding index in the model results and the issue indes, add if the issue was detected to the data_issues index to that specific issue index
                        for row, index in zip(reader, range(len(ground_truth))):
                            print(issue_index, row["detected"], approach_path_index, model_index)
                            print(issue_dict[approach_path_index][model_index])
                            if index == issue_index:
                                issue_dict[approach_path_index][model_index] = row["detected"]
                                break

        issues_res.append(issue_dict)

    return issues_res


data_issues_models = evaluate_model_predictions_issues("deepseek_v3")


def canculate_issue_detection_accuracy(issue_dict):
    values = 0
    for issue_index, issue in enumerate(issue_dict):
        if issue == 'True':
            values += 1

    return values / 3


print(data_issues_models)
issues_vals = []
for issue_index, issue in enumerate(data_issues_models):
    issue_value = []
    for model_index, model in enumerate(issue):
        accuracy = canculate_issue_detection_accuracy(model)
        issue_value.append(f"{accuracy * 100:.2f}%")
    issues_vals.append(issue_value)

print(issues_vals)


data_issues_models = []

df = pd.DataFrame(issues_vals, index=issues, columns=approach_names)

print(df)



def plot_table(df, title=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('tight')
    ax.axis('off')

    # Create table from DataFrame
    table = mpltbl.table(ax, cellText=df.values,
                         colLabels=df.columns,
                         rowLabels=df.index,
                         cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(0.8, 1.5)

    # Convert percentage strings to numeric values (e.g., "75.00%" -> 75.0)
    numeric_vals = df.applymap(lambda s: float(s.replace('%', '')) if isinstance(s, str) and s.endswith('%') else 0)
    norm = colors.Normalize(vmin=numeric_vals.min().min(), vmax=numeric_vals.max().max())
    cmap = plt.get_cmap('YlGn')  # Use Yellow-Green colormap

    # Iterate over table cells and set background color based on numeric value
    for (row, col), cell in table.get_celld().items():
        # Data cells: row > 0 and col >= 0 (row 0 is header; col -1 for row labels)
        if row > 0 and col >= 0:
            # Table row 1 corresponds to df row index 0, hence row-1
            try:
                value = numeric_vals.iloc[row - 1, col]
            except Exception:
                value = 0
            cell.set_facecolor(cmap(norm(value)))
            cell.set_text_props(wrap=True)
        # Bold header cells
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold', color='black')

    plt.title(title)
    plt.show()

def plot_table_to_image(df, filename="table.png", title=""):
    fig, ax = plt.subplots(figsize=(13, 12))
    ax.axis('tight')
    ax.axis('off')

    # Create table from DataFrame
    table = mpltbl.table(ax, cellText=df.values,
                         colLabels=df.columns,
                         rowLabels=df.index,
                         cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(0.8, 3)

    # Convert percentage strings to numeric values
    numeric_vals = df.applymap(lambda s: float(s.replace('%', '')) if isinstance(s, str) and s.endswith('%') else 0)
    norm = colors.Normalize(vmin=numeric_vals.min().min(), vmax=numeric_vals.max().max())
    cmap = plt.get_cmap('YlGn')

    # Apply background color based on value
    for (row, col), cell in table.get_celld().items():
        if row > 0 and col >= 0:
            try:
                value = numeric_vals.iloc[row - 1, col]
            except Exception:
                value = 0
            cell.set_facecolor(cmap(norm(value)))
            cell.set_text_props(wrap=True)
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold', color='black')

    plt.title(title)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# --- Plot and Save the Table ---
plot_table(df, title="Approaches (Highlighted by Accuracy)")
# Save the table as an image
plot_table_to_image(df, filename="issue_comparison_across_Approaches.png", title="Issue Evaluation Comparison")
