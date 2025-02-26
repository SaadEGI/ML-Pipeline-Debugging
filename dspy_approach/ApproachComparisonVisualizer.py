import csv
import os
import logging
from pathlib import Path
from typing import List, Any

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.table as mpltbl
import matplotlib.colors as colors

from model_inspection import read_code_from_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_files(root_dir: Path, target_filename: str) -> List[Path]:
    """Recursively find all files matching the target filename in the given directory."""
    return list(root_dir.rglob(target_filename))


def load_ground_truth(file_path: Path) -> List[str]:
    """Load non-empty lines from a text file as ground truth labels."""
    try:
        with file_path.open("r") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logging.error(f"Error loading ground truth from {file_path}: {e}")
        return []


def build_code_dataframe(file_list: List[Path]) -> pd.DataFrame:
    """Build a DataFrame from a list of file paths by reading their code."""
    data = [{"pipeline_code": read_code_from_file(str(fp))} for fp in file_list]
    return pd.DataFrame(data)


# Discover pipeline files
pipelines = get_files(Path("../pipelines_examples"), "example-0.py")
corrected_pipelines = get_files(Path("../example_pipelines"), "example-0-fixed.py")

df = build_code_dataframe(pipelines)
df_correct = build_code_dataframe(corrected_pipelines)

# Load ground truth labels and descriptions
ground_truth = load_ground_truth(Path("../pipelines_examples/issues_names.txt"))
ground_truth_description = load_ground_truth(Path("../pipelines_examples/issues_descriptions.txt"))
logging.info(f"Loaded {len(ground_truth)} ground truth labels.")

# Define models, issues, and approaches
models = ['gpt-4o-mini', 'gemini-2.0-flash-001', 'gpt-4o']
issues = ['Cross Validation Errors', 'Data Anonymization Errors', 'Data Filtering Errors',
          'Data Imputation Errors', 'Data Leakage Errors', 'Data Slicing Errors',
          'Data Splitting Errors', 'Specification Bias']
approaches_paths = [
    "results_issue_input_code",
    "results_issue_input_code_dag",
    "results_issue_input_data_code",
    "results_issue_input_data_code_dag"
]
approach_names = ['Approach 1', 'Approach 2', 'Approach 3', 'Approach 4']


def evaluate_model_predictions_issues() -> List[List[List[Any]]]:
    """
    For each ground truth issue, build a matrix of detection results.

    Returns a list (one per ground truth issue) where each element is a 2D list
    with dimensions [len(approaches_paths) x len(models)] containing the 'detected' value.
    """
    issues_results = []
    # Loop over each ground truth issue index
    for issue_idx in range(len(ground_truth)):
        # Initialize a matrix for this issue (rows: approaches, columns: models)
        issue_matrix = [[None for _ in range(len(models))] for _ in range(len(approaches_paths))]
        for approach_idx, approach_path in enumerate(approaches_paths):
            for model_idx, model_name in enumerate(models):
                csv_path = Path(approach_path) / f"{model_name}_results.csv"
                if csv_path.exists():
                    try:
                        with csv_path.open("r") as f:
                            reader = csv.DictReader(f)
                            # Find the row corresponding to this issue index
                            for row_idx, row in enumerate(reader):
                                if row_idx == issue_idx:
                                    issue_matrix[approach_idx][model_idx] = row["detected"]
                                    break
                    except Exception as e:
                        logging.error(f"Error processing file {csv_path}: {e}")
                        issue_matrix[approach_idx][model_idx] = None
                else:
                    logging.warning(f"File {csv_path} does not exist.")
                    issue_matrix[approach_idx][model_idx] = None
        issues_results.append(issue_matrix)
    return issues_results


def calculate_issue_detection_accuracy(detections: List[Any]) -> float:
    """
    Given a list of detection results (as strings), return the fraction of detections that are 'True'.
    """
    true_count = sum(1 for d in detections if d == 'True')
    total = len(detections)
    return true_count / total if total > 0 else 0


# Evaluate predictions (dummy parameter not used)
data_issues_models = evaluate_model_predictions_issues()
logging.info(f"Evaluation results: {data_issues_models}")

# Compute accuracy (as percentage strings) for each issue and approach
issues_accuracy = []
for issue_matrix in data_issues_models:
    approach_acc = []
    for row in issue_matrix:
        accuracy = calculate_issue_detection_accuracy(row)
        approach_acc.append(f"{accuracy * 100:.2f}%")
    issues_accuracy.append(approach_acc)

logging.info(f"Issue accuracies: {issues_accuracy}")

# Build a DataFrame with rows as issues and columns as approaches
df_issues = pd.DataFrame(issues_accuracy, index=issues, columns=approach_names)
print(df_issues)


def plot_table(df: pd.DataFrame, title: str = "") -> None:
    """Display a table with cell colors based on percentage values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('tight')
    ax.axis('off')

    table = mpltbl.table(ax, cellText=df.values,
                         colLabels=df.columns,
                         rowLabels=df.index,
                         cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(0.8, 1.5)

    # Convert percentage strings to floats (e.g., "75.00%" -> 75.0)
    numeric_vals = df.applymap(lambda s: float(s.replace('%', '')) if isinstance(s, str) and s.endswith('%') else 0)
    norm = colors.Normalize(vmin=numeric_vals.min().min(), vmax=numeric_vals.max().max())
    cmap = plt.get_cmap('YlGn')

    for (row, col), cell in table.get_celld().items():
        # Data cells (headers: row=0 or col=-1 are skipped for coloring)
        if row > 0 and col >= 0:
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


def plot_table_to_image(df: pd.DataFrame, filename: str = "table.png", title: str = "") -> None:
    """Save a table as an image with cell background colors based on percentage values."""
    fig, ax = plt.subplots(figsize=(13, 12))
    ax.axis('tight')
    ax.axis('off')

    table = mpltbl.table(ax, cellText=df.values,
                         colLabels=df.columns,
                         rowLabels=df.index,
                         cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(0.8, 3)

    numeric_vals = df.applymap(lambda s: float(s.replace('%', '')) if isinstance(s, str) and s.endswith('%') else 0)
    norm = colors.Normalize(vmin=numeric_vals.min().min(), vmax=numeric_vals.max().max())
    cmap = plt.get_cmap('YlGn')

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


# Plot and save the evaluation table
plot_table(df_issues, title="Approaches (Highlighted by Accuracy)")
plot_table_to_image(df_issues, filename="issue_comparison_across_Approaches.png",
                    title="Issue Evaluation Comparison")
