import csv
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.table as mpltbl
import pandas as pd

from utils import read_code_from_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_files(root_dir: Path, target_filename: str) -> List[Path]:
    """Recursively collect files matching the target filename under the given directory."""
    return list(root_dir.rglob(target_filename))


def load_text_lines(file_path: Path) -> List[str]:
    """Return non-empty, stripped lines from a text file."""
    try:
        with file_path.open("r") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return []


def build_code_dataframe(file_list: List[Path]) -> pd.DataFrame:
    """Build a DataFrame from a list of file paths by reading their code."""
    data = [{"pipeline_code": read_code_from_file(str(fp))} for fp in file_list]
    return pd.DataFrame(data)


# Gather pipelines and corrected pipelines
pipelines = get_files(Path("../pipelines_examples"), "example-0.py")
corrected_pipelines = get_files(Path("../example_pipelines"), "example-0-fixed.py")

df = build_code_dataframe(pipelines)
df_correct = build_code_dataframe(corrected_pipelines)

# Load ground truth labels and descriptions
ground_truth = load_text_lines(Path("../pipelines_examples/issues_names.txt"))
ground_truth_description = load_text_lines(Path("../pipelines_examples/issues_descriptions.txt"))
logging.info(f"Loaded {len(ground_truth)} ground truth labels.")


def evaluate_model_predictions(ground_truth: List[str],
                               model_name: str,
                               path: str) -> Tuple[float, float]:
    """
    Evaluate model predictions against the ground truth.

    Expects a CSV file at {path}/{model_name}_results.csv with columns "detected" and "confidence".
    Returns a tuple (accuracy, confidence_average).
    """
    predictions: List[Tuple[str, float]] = []
    full_path = Path(path) / f"{model_name}_results.csv"

    if full_path.exists():
        with full_path.open("r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                predictions.append((row["detected"], float(row["confidence"])))
    else:
        logging.warning(f"File {full_path} does not exist.")
        return 0.0, 0.0

    if len(ground_truth) != len(predictions):
        logging.warning("Mismatch between ground truth entries and predictions.")
        return 0.0, 0.0

    correct = 0
    confidence_sum = 0.0
    for i in range(len(ground_truth)):
        if predictions[i][0] == 'True':
            correct += 1
            confidence_sum += predictions[i][1]
    accuracy = correct / len(ground_truth)
    confidence_average = round(confidence_sum / correct, 2) if correct > 0 else 0.0

    logging.info(f"{model_name} Accuracy: {accuracy * 100:.2f}%, "
                 f"Average Confidence: {confidence_average:.2f}")
    return accuracy, confidence_average


# Define models, approaches, and paths
models = ['gpt-4o-mini', 'gemini-2.0-flash-001', 'gpt-4o']
approach_names = ['Approach 1', 'Approach 2', 'Approach 3', 'Approach 4']
approaches_paths = [
    "results_issue_input_code",
    "results_issue_input_code_dag",
    "results_issue_input_data_code",
    "results_issue_input_data_code_dag"
]

# Collect evaluation results for all models and approaches
evaluation_data = []
for model in models:
    row = {'Model': model}
    for approach_name, approach_path in zip(approach_names, approaches_paths):
        accuracy, conf_avg = evaluate_model_predictions(ground_truth, model, approach_path)
        row[approach_name] = f"Acc: {accuracy * 100:.2f}%\nConf: {conf_avg:.2f}"
    evaluation_data.append(row)

# Create DataFrame from evaluation results
df_evaluation = pd.DataFrame(evaluation_data)
df_evaluation.set_index('Model', inplace=True)
logging.info(f"Evaluation DataFrame:\n{df_evaluation}")


def plot_table(df: pd.DataFrame, title: str = "") -> None:
    """Display a table using the DataFrame content."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('tight')
    ax.axis('off')

    table = mpltbl.table(ax, cellText=df.values, colLabels=df.columns, rowLabels=df.index,
                         cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(0.8, 1.5)

    # Set header styles
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold', color='black')
        else:
            cell.set_text_props(wrap=True)

    plt.title(title)
    plt.show()


def plot_table_to_image(df: pd.DataFrame, filename: str = "table.png", title: str = "") -> None:
    """Save a table (created from the DataFrame) as an image file."""
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.axis('tight')
    ax.axis('off')

    table = mpltbl.table(ax, cellText=df.values, colLabels=df.columns, rowLabels=df.index,
                         cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(0.8, 3)

    # Set header styles
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold', color='black')
        else:
            cell.set_text_props(wrap=True)

    plt.title(title)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


# Plot evaluation table to screen and save it as an image
plot_table(df_evaluation, title="Approaches")
plot_table_to_image(df_evaluation, filename="model_comparison.png", title="Model Evaluation Comparison")
