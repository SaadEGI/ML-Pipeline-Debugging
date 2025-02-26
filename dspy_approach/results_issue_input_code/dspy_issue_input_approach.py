import os
import logging
from pathlib import Path
from typing import Dict, Any, List

import dspy
import pandas as pd
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from dspy_approach.utils import read_code_from_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent.parent
dotenv_path = BASE_DIR / ".env"
print(dotenv_path)
load_dotenv(dotenv_path=str(dotenv_path))


# Retrieve API keys
def get_env_variable(key: str) -> str:
    value = os.getenv(key)
    if not value:
        logging.error(f"Environment variable {key} not set.")
        raise EnvironmentError(f"Missing environment variable: {key}")
    return value


deepseek_api_key = get_env_variable("DEEPSEEK_API_KEY")
deepseek_r1_api_key = get_env_variable("OPENROUTER_API_KEY")
openai_api_key = get_env_variable("OPENAI_API_KEY")
gemini_api_key = get_env_variable("GEMINI_API_KEY")
together_api_key = get_env_variable("TOGETHER_API_KEY")

# Initial LM configuration
default_lm = dspy.LM(
    'openai/deepseek-chat',
    api_key=deepseek_api_key,
    api_base="https://api.deepseek.com"
)


# Load issue names
def load_issue_names(file_path: Path) -> List[str]:
    with file_path.open("r") as file:
        return [line.strip() for line in file if line.strip()]


issue_names_path = BASE_DIR / "pipelines_examples" / "issues_names.txt"
issue_names = load_issue_names(issue_names_path)


# Discover pipelines (example-0.py files)
def discover_pipelines(root_dir: Path, filename: str) -> List[Path]:
    return [Path(root) / file
            for root, _, files in os.walk(root_dir)
            for file in files if file == filename]


pipelines = discover_pipelines(BASE_DIR / "pipelines_examples", "example-0.py")


# Prepare data for processing
def prepare_data(pipelines: List[Path], issues: List[str]) -> pd.DataFrame:
    data = []
    for i, pipeline in enumerate(pipelines):
        code = read_code_from_file(str(pipeline))
        # Associate issue name if available
        issue_name = issues[i] if i < len(issues) else "Unknown Issue"
        data.append({"pipeline_code": code, "issue_name": issue_name, "pipeline_path": str(pipeline)})
    return pd.DataFrame(data)


df = prepare_data(pipelines, issue_names)


# Define a dspy signature for code issue classification
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
    issue_detected: bool = dspy.OutputField()
    confidence: float = dspy.OutputField()


classify_code_issue = dspy.Predict(CodeIssue)

# Dictionary to map model indices to configurations
MODEL_CONFIGS: Dict[int, Dict[str, Any]] = {
    0: {
        'lm_id': 'openai/deepseek-ai/DeepSeek-V3',
        'api_key': together_api_key,
        'api_base': "https://api.together.xyz/v1",
        'cache': False,
        'model_name': "deepseek_v3"
    },
    1: {
        'lm_id': 'openai/deepseek-ai/DeepSeek-R1',
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
    lm_instance, model_name = configure_model(model_index)
    dspy.configure(lm=lm_instance)

    results = []
    for idx, row in df.iterrows():
        classification = classify_code_issue(code=row['pipeline_code'], issue_name=row['issue_name'])
        logging.info(f"Model: {model_name}, Pipeline: {row['pipeline_path']}, "
                     f"Detected: {classification.issue_detected}, Confidence: {classification.confidence}")
        results.append({
            "pipeline_path": row['pipeline_path'],
            "model": model_name,
            "detected": classification.issue_detected,
            "confidence": classification.confidence
        })
    results_df = pd.DataFrame(results)
    output_file = Path(f"{model_name}_results.csv")
    results_df.to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")
    return results_df

def visualize_results(models: List[str]):
    visualization_data = []
    for model in models:
        results_file = Path(f"{model}_results.csv")
        if results_file.is_file():
            df_model = pd.read_csv(results_file)
            for idx, row in df_model.iterrows():
                visualization_data.append({
                    "model": model,
                    "pipeline_index": idx,
                    "issue_detected": row['detected'],
                    "confidence": row['confidence']
                })

    if not visualization_data:
        logging.warning("No data available for visualization.")
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
    plt.suptitle("Prediction per Model")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Example calls (you may uncomment as needed, each number corresponds to a different model
# 0: DeepSeek V3
# 1: DeepSeek R1
# 2: GPT-4o Mini
# 3: Gemini 2.0 Flash 001
# 4: GPT-4o

# classify_with_single_model(df, 0)
# classify_with_single_model(df, 1)
classify_with_single_model(df, 2)
#classify_with_single_model(df, 3)
#classify_with_single_model(df, 4)





# Call visualization for models of interest
visualize_results(["deepseek_v3", "deepseek_r1_reasoner", "gpt-4o-mini", "gemini-2.0-flash-001", "gpt-4o"])
