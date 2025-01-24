import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
client_deepseek = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
client_deepseek_r1 = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1/")


def read_code_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            code = file.read()
        return code
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return ""
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def add_line_numbers(code):
    """Add aligned line numbers to code display"""
    lines = code.split('\n')
    if not lines:
        return ""

    # Calculate padding based on total lines
    line_count = len(lines)
    num_width = len(str(line_count))

    # Create formatted lines with numbers
    numbered_lines = [
        f"{i + 1:>{num_width}}| {line}"
        for i, line in enumerate(lines)
    ]

    return '\n'.join(numbered_lines)


def create_prompt_with_issue_description(issue_description, issue_description_full, code):
    prompt = f"""
    You are an AI assistant that analyzes Python machine learning pipeline code to identify whether a specific issue is present.

    The code includes line numbers (e.g., '1| ...') for reference.
    When identifying issues:
    - ALWAYS specify line numbers using the provided numbering
    - The issue may affect MULTIPLE LINES - list ALL relevant line numbers
    ```python
    {code}
    ```

    Check specifically for {issue_description.lower()}, here is a description of the issue: {issue_description_full}
    
    Please output the results in the following JSON format (without any code fences or additional text):

    {{
    "issue_detected": boolean,
    "affected_lines": [numbers]  #Array of ALL affected line numbers, e.g., [5, 8-10, 14]
    }}
    Do NOT include any code blocks, markdown, or additional explanations. Output ONLY the JSON.
    """
    return prompt


def check_code_with_gpt4(prompt):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
    )
    output = response.choices[0].message.content
    return output


def check_code_with_deepseek(prompt, reason):
    # reason yes or no
    if reason == 0:
        response = client_deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            stream=False
        )
        output = response.choices[0].message.content
        return output
    else:
        response = client_deepseek_r1.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}]

        )
        output = response.choices[0].message.content
        return output


def analyse_output(output, model_name, pipeline_number):
    feedback = json.loads(output)
    return_values = [feedback["issue_detected"], feedback["affected_lines"], model_name,
                     pipeline_number]
    return return_values


def clean_response(output):
    # Use regex to find JSON content between code fences
    json_pattern = re.compile(
        r"```(?:json)?\s*(\{.*?\}|$$.*?$$)\s*```", re.DOTALL
    )
    match = json_pattern.search(output)
    if match:
        json_content = match.group(1)
    else:
        # If no code fences, try to find JSON content in the whole string
        json_pattern = re.compile(r"(\{.*?\}|$$.*?$$)", re.DOTALL)
        match = json_pattern.search(output)
        if match:
            json_content = match.group(1)
        else:
            # Unable to find JSON content
            raise ValueError("No JSON content found in the model output.")

    # Clean up the JSON string
    json_content = json_content.strip()

    return json_content


def save_numbered_code(numbered_code, pipeline_number, output_dir="numbered_references"):
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create filename with pipeline number
    filename = f"pipeline_{pipeline_number}_numbered.txt"
    filepath = os.path.join(output_dir, filename)

    # Save to text file
    with open(filepath, "w") as f:
        f.write(numbered_code)

    return filepath


def process_model_output(output, model_name, index_i, clean=False):
    cleaned = clean_response(output) if clean else output
    data = json.loads(cleaned)
    data.update({
        "model_name": model_name,
        "pipeline_number": index_i
    })
    return data


def main():
    pipelines = ["corrupted_pipelines/example-0.py", "corrupted_pipelines/example-0-annotation.py"]
    issue_descriptions = ["Aggregation errors", "Annotation errors"]
    issue_descriptions_full = [
        "aggregation errors occur when data is grouped or transformed in a way that introduces inaccuracies or misrepresents the underlying data distribution"
        ,
        "Annotation errors arise when the labeling of data points introduces inaccuracies or biases into the dataset."]
    output_res = []
    feedback_list = []

    with open('results/output_res.json') as f:
        output_res = json.load(f)
    models = ['Ground Truth', 'OpenAI GPT 4', 'Deepseek V3', 'Deepseek R1']

    data = {
        'Model': models,
        'Class_NoVuln': ["Pipeline w/ Aggr Err", "Pipeline w/ Aggr Err", "Pipeline w/ Aggr Err",
                         "Pipeline w/ Aggr Err"],
        'Class_Vuln': ["Pipeline w/ Ann Err", "Pipeline w/ Ann Err", "Pipeline w/ Ann Err", "Pipeline w/ Ann Err"],
        'Aggr_detected': ["-", 0, 0, 0],
        'Aggr_Lines': ["[28, 31]", 0, 0, 0],
        'Ann_detected': ["-", 0, 0, 0],
        'Ann_Lines': ["[27, 28]", 0, 0, 0]
    }

    model_mapping = {
        "gpt4": "OpenAI GPT 4",
        "DeepSeek V3": "Deepseek V3",
        "DeepSeek R1": "Deepseek R1"
    }

    for entry in output_res:
        issue_detected, affected_lines, model_key, pipeline_num = entry
        model_name = model_mapping.get(model_key, model_key)

        if model_name not in models:
            continue

        idx = models.index(model_name)

        if pipeline_num == 0:  # Aggregation pipeline
            data['Aggr_detected'][idx] = "Yes" if issue_detected else "No"
            data['Aggr_Lines'][idx] = affected_lines
        elif pipeline_num == 1:  # Annotation pipeline
            data['Ann_detected'][idx] = "Yes" if issue_detected else "No"
            data['Ann_Lines'][idx] = affected_lines

    df = pd.DataFrame(data)

    # Helper function to format different value types
    def format_value(value):
        if isinstance(value, (list, np.ndarray)):  # Handle arrays
            return ', '.join(map(str, value))
        elif isinstance(value, (int, float)):  # Handle numbers
            return f"{value:.2f}" if not float(value).is_integer() else f"{int(value)}"
        else:  # Handle strings and others
            return str(value)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('tight')
    ax.axis('off')

    # Prepare data for table
    table_vals = []
    for index, model in enumerate(df['Model']):
        table_vals.append([
            model,
            df['Class_NoVuln'][index],
            format_value(df['Aggr_detected'][index]),
            format_value(df['Aggr_Lines'][index])
        ])
        table_vals.append([
            '',  # empty for model column in second row
            df['Class_Vuln'][index],
            format_value(df['Ann_detected'][index]),
            format_value(df['Ann_Lines'][index])
        ])

    # Column headers and table title
    col_labels = ['Model', 'Pipeline', 'Issue Detected', 'Issue Lines']
    table_title = 'TABLE I: Comparison among the different approaches.'

    # Create the table
    table = ax.table(
        cellText=table_vals,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        edges='horizontal'
    )

    # Table styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    table.auto_set_column_width(col=list(range(len(col_labels))))

    for (pos, cell) in table.get_celld().items():
        if pos[0] == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#F0F0F0')

    ax.axhline(y=len(table_vals), color='black', linewidth=1.0, clip_on=False)
    ax.axhline(y=1, color='black', linewidth=0.7, linestyle='-', clip_on=False)

    # Title
    ax.set_title(table_title, y=1.05, fontsize=12)

    plt.tight_layout()

    if os.path.isfile('results/table.png'):
        i = 1
        while os.path.isfile(f"results/table_{i}.png"):
            i += 1
        fig.savefig(f'results/table_{i}.png')
    else:
        fig.savefig('results/table_1.png')

    plt.show()


if __name__ == "__main__":
    main()
