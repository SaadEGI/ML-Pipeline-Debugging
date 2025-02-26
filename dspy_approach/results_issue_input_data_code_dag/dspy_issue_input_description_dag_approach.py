import os
import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import dspy
from dspy import Evaluate  # if needed elsewhere
from dspy_approach.utils import read_code_from_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set base directory and load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent.parent
dotenv_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=str(dotenv_path))


def get_env_variable(key: str) -> str:
    """Retrieve environment variable value or raise an error if not set."""
    value = os.getenv(key)
    if not value:
        logging.error(f"Missing environment variable: {key}")
        raise EnvironmentError(f"Missing environment variable: {key}")
    return value


# Retrieve API keys
deepseek_api_key = get_env_variable("DEEPSEEK_API_KEY")
deepseek_r1_api_key = get_env_variable("OPENROUTER_API_KEY")
openai_api_key = get_env_variable("OPENAI_API_KEY")
gemini_api_key = get_env_variable("GEMINI_API_KEY")
together_api_key = get_env_variable("TOGETHER_API_KEY")

# Preconfigure LM instances if needed elsewhere
lm_default = dspy.LM('openai/deepseek-chat', api_key=deepseek_api_key,
                     api_base="https://api.deepseek.com")
deepseek_r1 = dspy.LM('openai/deepseek/deepseek-r1', api_key=deepseek_r1_api_key,
                      api_base="https://openrouter.ai/api/v1")


# --- File I/O and Data Preparation ---

def load_lines(file_path: Path) -> List[str]:
    """Read non-empty lines from a text file."""
    try:
        return [line.strip() for line in file_path.read_text().splitlines() if line.strip()]
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return []


def discover_files(root_dir: Path, target_filename: str) -> List[Path]:
    """Recursively discover files with the given target filename."""
    return list(root_dir.rglob(target_filename))


# Define paths for issue files
issues_names_path = BASE_DIR / "pipelines_examples" / "issues_names.txt"
issues_descriptions_path = BASE_DIR / "pipelines_examples" / "issues_descriptions.txt"
issues_impact_path = BASE_DIR / "pipelines_examples" / "issues_impact.txt"
pipeline_phase_path = BASE_DIR / "pipelines_examples" / "pipeline_phase.txt"

# Load issue-related data
issue_names = load_lines(issues_names_path)
issue_descriptions = load_lines(issues_descriptions_path)  # Loaded but not used here
issues_impact = load_lines(issues_impact_path)
pipeline_phases = load_lines(pipeline_phase_path)

# Discover pipeline and DAG files
pipelines = discover_files(BASE_DIR / "pipelines_examples", "example-0.py")
dags = discover_files(BASE_DIR / "pipelines_examples", "dag.json")
logging.info(f"Found {len(pipelines)} pipelines and {len(dags)} DAG files.")


def prepare_dataframe(pipelines: List[Path],
                      dags: List[Path],
                      issue_names: List[str],
                      issues_impact: List[str],
                      pipeline_phases: List[str]) -> pd.DataFrame:
    """
    Create a DataFrame combining code from pipelines and DAGs
    with corresponding issue, impact, and phase details.
    """
    data = []
    # Read pipeline code and corresponding DAG code.
    pipelines_sorted = sorted(pipelines)
    dags_sorted = sorted(dags)
    if len(pipelines_sorted) != len(dags_sorted):
        logging.warning("Number of pipelines and DAGs do not match; aligning by index.")

    for i, pipeline in enumerate(pipelines_sorted):
        code = read_code_from_file(str(pipeline))
        dag_code = read_code_from_file(str(dags_sorted[i])) if i < len(dags_sorted) else ""
        data.append({
            "pipeline_code": code,
            "dag": dag_code,
            "issue_name": issue_names[i] if i < len(issue_names) else "Unknown Issue",
            "issue_impact": issues_impact[i] if i < len(issues_impact) else "Unknown Impact",
            "pipeline_phase": pipeline_phases[i] if i < len(pipeline_phases) else "Unknown Phase",
            "pipeline_path": str(pipeline)
        })
    return pd.DataFrame(data)


df = prepare_dataframe(pipelines, dags, issue_names, issues_impact, pipeline_phases)


# --- dspy Signature and Prediction ---

class CodeIssue(dspy.Signature):
    """
    Analyze Python code and DAG to determine if a specific issue is present.

    Inputs:
      - code: The Python code to analyze.
      - dag: The DAG of the pipeline.
      - issue_name: The name of the issue to check.
      - potential_pipeline_phase: The pipeline phase.
      - issue_impact: The impact of the issue.

    Outputs:
      - issue_detected: Boolean indicating if the issue was found.
      - relevant_code_issue: Excerpt of relevant code.
      - fix: Suggested fix for the issue.
    """
    code: str = dspy.InputField()
    dag: str = dspy.InputField()
    issue_name: str = dspy.InputField(desc="Name of the issue to check")
    potential_pipeline_phase: str = dspy.InputField(desc="The pipeline phase")
    issue_impact: str = dspy.InputField(desc="The impact of the issue")

    issue_detected: bool = dspy.OutputField()
    relevant_code_issue: str = dspy.OutputField()
    fix: str = dspy.OutputField()


classify_code_issue = dspy.Predict(CodeIssue)

# --- Model Configuration and Classification ---

MODEL_CONFIGS: Dict[int, Dict[str, Any]] = {
    0: {
        'lm_id': 'openai/deepseek-ai/DeepSeek-V3',
        'api_key': together_api_key,
        'api_base': "https://api.together.xyz/v1",
        'cache': False,
        'model_name': "deepseek_v3"
    },
    1: {
        'lm_id': 'openai/deepseek/deepseek-r1',
        'api_key': together_api_key,
        'api_base': "https://api.together.xyz/v1",
        'cache': False,
        'model_name': "deepseek_r1"
    },
    99: {
        'lm_id': 'openai/deepseek-reasoner',
        'api_key': deepseek_api_key,
        'api_base': "https://api.deepseek.com",
        'cache': False,
        'model_name': "deepseek_r1_reasoner"
    },
    2: {
        'lm_id': 'openai/gpt-4o-mini',
        'api_key': openai_api_key,
        'cache': False,
        'model_name': "gpt-4o-mini"
    },
    3: {
        'lm_id': 'gemini/gemini-2.0-flash-001',
        'api_key': gemini_api_key,
        'cache': False,
        'model_name': "gemini-2.0-flash-001"
    },
    4: {
        'lm_id': 'openai/gpt-4o',
        'api_key': openai_api_key,
        'cache': False,
        'model_name': "gpt-4o"
    }
}


def configure_model(model_index: int) -> (dspy.LM, str):
    """
    Return a configured LM instance and its model name for the given index.
    """
    config = MODEL_CONFIGS.get(model_index)
    if not config:
        raise ValueError(f"Invalid model index: {model_index}")
    lm_instance = dspy.LM(
        config['lm_id'],
        api_key=config['api_key'],
        api_base=config.get('api_base'),
        cache=config.get('cache', True)
    )
    return lm_instance, config['model_name']


def classify_with_single_model(df: pd.DataFrame, model_index: int) -> pd.DataFrame:
    """
    Run classification for each pipeline using the specified model.

    Returns a DataFrame of results.
    """
    lm_instance, model_name = configure_model(model_index)
    dspy.configure(lm=lm_instance)

    results = []
    for idx, row in df.iterrows():
        classification = classify_code_issue(
            code=row['pipeline_code'],
            dag=row['dag'],
            issue_name=row['issue_name'],
            potential_pipeline_phase=row['pipeline_phase'],
            issue_impact=row['issue_impact']
        )
        logging.info(
            f"Model: {model_name} | Pipeline: {row['pipeline_path']} | Detected: {classification.issue_detected}")
        results.append({
            "pipeline_code": row['pipeline_path'],
            "model": model_name,
            "detected": classification.issue_detected,
            "relevant_code_issue": classification.relevant_code_issue,
            "fix": classification.fix
        })
    results_df = pd.DataFrame(results)
    output_file = Path(f"{model_name}_results.csv")
    results_df.to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")
    return results_df


# --- Save Relevant Code and Fixes ---

def save_relevant_code_and_fix(models: List[str], output_filename: str = "issue_fixes.csv") -> None:
    """
    Save relevant code excerpts and fixes for detected issues across models.
    """
    issue_fixes = []
    for model in models:
        result_file = Path(f"{model}_results.csv")
        if result_file.is_file():
            df_model = pd.read_csv(result_file)
            for _, row in df_model.iterrows():
                if row['detected']:
                    issue_fixes.append({
                        "model": model,
                        "relevant_code_issue": row['relevant_code_issue'],
                        "issue_name": row['pipeline_code'],
                        "fix": row['fix']
                    })
    if issue_fixes:
        df_issue_fixes = pd.DataFrame(issue_fixes)
        df_issue_fixes.to_csv(output_filename, index=False)
        logging.info(f"Issue fixes saved to {output_filename}")
    else:
        logging.info("No detected issues found; nothing to save.")


# --- Visualization ---

def visualize_results(model_list: List[str]) -> None:
    """
    Visualize prediction confidence per model.
    """
    visualization_data = []
    for model in model_list:
        result_file = Path(f"{model}_results.csv")
        if result_file.is_file():
            df_model = pd.read_csv(result_file)
            for idx, row in df_model.iterrows():
                visualization_data.append({
                    "model": model,
                    "pipeline_index": idx,
                    "issue_detected": row['detected'],
                    "confidence": row.get('confidence', None)
                })
    if not visualization_data:
        logging.warning("No visualization data available.")
        return

    df_vis = pd.DataFrame(visualization_data)
    df_vis['color'] = df_vis['issue_detected'].map({True: 'green', False: 'red'})
    unique_models = df_vis['model'].unique()

    fig, axes = plt.subplots(1, len(unique_models), figsize=(14, 6), sharey=True)
    for ax, model in zip(axes, unique_models):
        subset = df_vis[df_vis['model'] == model]
        ax.scatter(subset['pipeline_index'], subset['confidence'], color=subset['color'], s=100, edgecolor='k')
        ax.set_title(model)
        ax.set_xlabel("Pipeline Index")
        ax.set_ylim(0, 1.1)
        ax.grid(True)
    axes[0].set_ylabel("Confidence")
    plt.suptitle("Prediction Confidence per Model")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- Main Execution ---

if __name__ == "__main__":
    # Uncomment the desired model runs
    # 0: DeepSeek V3
    # 1: DeepSeek R1
    # 2: GPT-4o Mini
    # 3: Gemini 2.0 Flash 001
    # 4: GPT-4o

    # classify_with_single_model(df, 0)
    # classify_with_single_model(df, 1)
    classify_with_single_model(df, 2)
    classify_with_single_model(df, 3)
    classify_with_single_model(df, 4)


    # Save relevant code issues and fixes from selected models
    model_names = ["deepseek_v3", "deepseek_r1_reasoner", "gpt-4o-mini", "gemini-2.0-flash-001", "gpt-4o"]
    save_relevant_code_and_fix(model_names)

    # Visualize results for selected models
    visualize_results(model_names)
