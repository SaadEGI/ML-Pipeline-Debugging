import csv
import logging
from pathlib import Path
from typing import List

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.table as mpltbl
import pandas as pd

from utils import read_code_from_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_file_paths(root_dir: Path, target_filename: str) -> List[Path]:
    """Recursively find all files with the given filename in the specified directory."""
    return list(root_dir.rglob(target_filename))


def load_file_lines(file_path: Path) -> List[str]:
    """Read non-empty lines from a text file."""
    try:
        with file_path.open("r") as file:
            return [line.strip() for line in file if line.strip()]
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return []


# Get pipelines and corrected pipelines
pipelines = get_file_paths(Path("../pipelines_examples"), "example-0.py")
corrected_pipelines = get_file_paths(Path("../example_pipelines"), "example-0-fixed.py")

# Create DataFrames from pipeline code
df_correct = pd.DataFrame([{"pipeline_code": read_code_from_file(str(p))} for p in corrected_pipelines])
df = pd.DataFrame([{"pipeline_code": read_code_from_file(str(p))} for p in pipelines])

# Load ground truth labels and descriptions
ground_truth = load_file_lines(Path("../pipelines_examples/issues_names.txt"))
ground_truth_description = load_file_lines(Path("../pipelines_examples/issues_descriptions.txt"))
logging.info(f"Loaded {len(ground_truth)} ground truth labels.")

# Define approach and issue labels
approach_names = ['gpt-4o-mini', 'gemini-2.0-flash-001', 'gpt-4o']
issue_labels = ['Cross Validation Errors', 'Data Anonymization Errors', 'Data Filtering Errors',
                'Data Imputation Errors', 'Data Leakage Errors', 'Data Slicing Errors',
                'Data Splitting Errors', 'Specification Bias']
approaches_paths = [
    "results_issue_input_code",
    "results_issue_input_code_dag",
    "results_issue_input_data_code",
    "results_issue_input_data_code_dag"
]


def evaluate_model_predictions_issues() -> List[List[List[str]]]:
    """
    Evaluate prediction results for each ground truth issue across different approaches.

    Returns:
      A list (length = number of ground truth issues) of matrices. Each matrix has
      dimensions [len(approach_names) x len(approaches_paths)] containing the detection
      result (as string) for that issue.
    """
    issues_results = []
    for issue_index in range(len(ground_truth)):
        # Initialize a matrix for the current issue
        issue_matrix = [[None for _ in range(len(approaches_paths))] for _ in range(len(approach_names))]
        for model_idx, model_name in enumerate(approach_names):
            for approach_idx, approach_path in enumerate(approaches_paths):
                csv_path = Path(approach_path) / f"{model_name}_results.csv"
                if csv_path.exists():
                    try:
                        with csv_path.open("r") as file:
                            reader = csv.DictReader(file)
                            # Iterate over rows to find the row corresponding to the current issue index
                            for row_idx, row in enumerate(reader):
                                if row_idx == issue_index:
                                    issue_matrix[model_idx][approach_idx] = row["detected"]
                                    break
                    except Exception as e:
                        logging.error(f"Error processing file {csv_path}: {e}")
                        issue_matrix[model_idx][approach_idx] = None
                else:
                    logging.warning(f"File {csv_path} does not exist.")
                    issue_matrix[model_idx][approach_idx] = None
        issues_results.append(issue_matrix)
    return issues_results


def calculate_issue_detection_accuracy(detections: List[str]) -> float:
    """
    Calculate detection accuracy from a list of detection results.

    Each detection is expected to be the string 'True' when the issue is detected.
    Returns the fraction of 'True' values.
    """
    true_count = sum(1 for d in detections if d == 'True')
    total = len(detections)
    return true_count / total if total > 0 else 0


# Evaluate predictions (dummy model parameter is unused)
data_issues_models = evaluate_model_predictions_issues()
logging.info(f"Evaluation results: {data_issues_models}")

# Compute accuracy percentages for each ground truth issue and each approach
issues_values = []
for issue_matrix in data_issues_models:
    issue_row = []
    for detection_list in issue_matrix:
        accuracy = calculate_issue_detection_accuracy(detection_list)
        issue_row.append(f"{accuracy * 100:.2f}%")
    issues_values.append(issue_row)

# Create DataFrame: rows correspond to ground truth issues, columns to approach names
df_issues = pd.DataFrame(issues_values, index=issue_labels, columns=approach_names)


def plot_table(df: pd.DataFrame, title: str = "") -> None:
    """Plot a table with cell background colors based on the numeric values extracted from percentage strings."""
    fig, ax = plt.subplots(figsize=(6, 4))
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
    plt.show()


def plot_table_to_image(df: pd.DataFrame, filename: str = "table.png", title: str = "") -> None:
    """Plot a table from a DataFrame and save it as an image file."""
    fig, ax = plt.subplots(figsize=(14, 12))
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


# Plot and save the table of evaluation results
plot_table(df_issues, title="Approaches (Highlighted by Accuracy)")
plot_table_to_image(df_issues, filename="issue_comparison.png", title="Issue Evaluation Comparison")
