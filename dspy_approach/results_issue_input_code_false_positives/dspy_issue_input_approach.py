import logging
import os
from pathlib import Path
from typing import List, Dict, Any

import dspy
import pandas as pd
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from dspy_approach.utils import read_code_from_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set base directory and load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
dotenv_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=str(dotenv_path))


def get_env_variable(key: str) -> str:
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

# Preconfigure some LM instances if needed elsewhere
lm_default = dspy.LM('openai/deepseek-chat', api_key=deepseek_api_key,
                     api_base="https://api.deepseek.com")
deepseek_r1 = dspy.LM('openai/deepseek/deepseek-r1', api_key=deepseek_r1_api_key,
                      api_base="https://openrouter.ai/api/v1")


# --- File loading and data preparation ---

def load_lines(file_path: Path) -> List[str]:
    """Read lines from a file, stripping whitespace."""
    try:
        return [line.strip() for line in file_path.read_text().splitlines() if line.strip()]
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return []


def discover_pipelines(root_dir: Path, target_filename: str) -> List[Path]:
    """Recursively find all files matching the target filename in the given directory."""
    return list(root_dir.rglob(target_filename))


def prepare_dataframe(pipelines: List[Path], issue_names: List[str]) -> pd.DataFrame:
    """Read each pipeline's code and combine with corresponding issue name into a DataFrame."""
    data = []
    for i, pipeline in enumerate(pipelines):
        code = read_code_from_file(str(pipeline))
        issue = issue_names[i] if i < len(issue_names) else "Unknown Issue"
        data.append({
            "pipeline_code": code,
            "issue_name": issue,
            "pipeline_path": str(pipeline)
        })
    return pd.DataFrame(data)


# Load issues and pipeline paths
issues_file = BASE_DIR / "pipelines_examples" / "issues_names.txt"
issue_names = load_lines(issues_file)

pipelines = discover_pipelines(BASE_DIR / "corrected_pipelines", "example-0-fixed.py")
logging.info(f"Found {len(pipelines)} pipeline files.")

df = prepare_dataframe(pipelines, issue_names)


# --- dspy Signature and Prediction ---

class CodeIssue(dspy.Signature):
    """
    Analyze Python code to check if a specific issue is present.

    Inputs:
      - code: The Python code to analyze.
      - issue_name: The name of the issue to check.
      - hint: A hint message to guide the analysis.

    Outputs:
      - issue_detected: Boolean indicating if the issue was found.
      - confidence: Float indicating the confidence in the result.
    """
    code: str = dspy.InputField()
    issue_name: str = dspy.InputField(desc="Name of the issue to check")
    hint: str = dspy.InputField()

    issue_detected: bool = dspy.OutputField()
    confidence: float = dspy.OutputField()


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
    99: {
        'lm_id': 'openai/deepseek/deepseek-r1',
        'api_key': deepseek_r1_api_key,
        'api_base': "https://openrouter.ai/api/v1",
        'cache': False,
        'model_name': "deepseek_r1"
    },
    1: {
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
    """Configure and return the LM instance and model name for the given index."""
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
    """Run classification for each pipeline using the specified model."""
    lm_instance, model_name = configure_model(model_index)
    dspy.configure(lm=lm_instance)

    results = []
    for idx, row in df.iterrows():
        code = row['pipeline_code']
        issue_to_check = row['issue_name']
        logging.info(f"Processing pipeline: {row['pipeline_path']} | Issue: {issue_to_check}")
        classification = classify_code_issue(
            code=code,
            issue_name=issue_to_check,
            hint="it is possible that the code pipeline does not contain the issue."
        )
        logging.info(
            f"Model: {model_name}, Detected: {classification.issue_detected}, Confidence: {classification.confidence}")
        results.append({
            "pipeline_code": row['pipeline_path'],
            "model": model_name,
            "detected": classification.issue_detected,
            "confidence": classification.confidence
        })
    results_df = pd.DataFrame(results)
    output_file = Path(f"{model_name}_results.csv")
    results_df.to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")
    return results_df


def visualize_results(model_list: List[str]) -> None:
    """Visualize prediction confidence per model."""
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
                    "confidence": row['confidence']
                })
    if not visualization_data:
        logging.warning("No visualization data available.")
        return

    df_vis = pd.DataFrame(visualization_data)
    # Map detection to colors (red for detected, green for not detected)
    df_vis['color'] = df_vis['issue_detected'].map({True: 'red', False: 'green'})
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
    plt.suptitle("Prediction per Model")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Execute classification with a chosen model (uncomment additional calls as needed)
if __name__ == "__main__":
    # Example calls (you may uncomment as needed, each number corresponds to a different model
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

    visualize_results(["deepseek_v3", "deepseek_r1_reasoner", "gpt-4o-mini", "gemini-2.0-flash-001", "gpt-4o"])
