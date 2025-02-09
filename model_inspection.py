import json
import os
import re

import openai
from dotenv import load_dotenv
from openai import OpenAI
import dspy_run

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
client_deepseek = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
client_deepseek_r1 = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1/")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)


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

    line_count = len(lines)
    num_width = len(str(line_count))

    numbered_lines = [
        f"{i + 1:>{num_width}}| {line}"
        for i, line in enumerate(lines)
    ]

    return '\n'.join(numbered_lines)


def create_prompt_with_issue_description(issue_name, issue_description, code, dag=None, aggregated_stats=None):
    prompt = f"""
    You are an AI assistant that analyzes Python machine learning pipeline code to identify whether a specific issue is present.

    The code includes line numbers (e.g., '1| ...') for reference.
    When identifying issues:
    - ALWAYS specify line numbers using the provided numbering
    - The issue may affect MULTIPLE LINES - list ALL relevant line numbers
    ```python
    {code}
    ```
    """
    if dag:
        prompt += f"The DAG structure of the code is the following json file:\n{dag}\n"
    if aggregated_stats:
        prompt += f"The aggregated statistics of the code are the following json file:\n{aggregated_stats}\n"

    prompt += f"""
    Check specifically for {issue_name.lower()}, and detect if the issue is present.
    """
    if issue_description:
        prompt += """here is a description of the issue: {issue_description_full}"""

    prompt += """
    Please output the results in the following JSON format (without any code fences or additional text):

    {{
    "issue_detected": boolean,
    "affected_lines": [numbers] 
    }}
    Do NOT include any code blocks, markdown, or additional explanations. Output ONLY the JSON.
    """
    return prompt



def create_prompt_with_dag(issue_name, issue_description, dag=None, aggregated_stats=None):
    prompt = f"""
    You are an AI assistant that analyzes Python machine learning pipeline code converted to DAG to identify whether a specific issue is present.

    When identifying issues:
    - ALWAYS specify line numbers using the provided numbering
    - The issue may affect MULTIPLE LINES - list ALL relevant line numbers
    """
    if dag:
        prompt += f"The AST Tree of the code is the following json file:\n{dag}\n"
    if aggregated_stats:
        prompt += f"The aggregated statistics of the code are the following json file:\n{aggregated_stats}\n"

    prompt += f"""
    Check specifically for {issue_name.lower()}, and detect if the issue is present.
    """
    if issue_description:
        prompt += """here is a description of the issue: {issue_description_full}"""

    prompt += """
    Please output the results in the following JSON format (without any code fences or additional text):

    {{
    "issue_detected": boolean,
    "affected_lines": [numbers] 
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


def check_code_with_openrouter(prompt):
    response = client.chat.completions.create(
        model="deepseek/deepseek-r1",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
    )
    output = response.choices[0].message.content
    return output


def analyse_output(output, model_name, pipeline_number):
    feedback = json.loads(output)
    return_values = [feedback["issue_detected"], feedback["affected_lines"], model_name,
                     pipeline_number]
    return return_values


def clean_response(output):
    json_pattern = re.compile(
        r"```(?:json)?\s*(\{.*?\}|$$.*?$$)\s*```", re.DOTALL
    )
    match = json_pattern.search(output)
    if match:
        json_content = match.group(1)
    else:

        json_pattern = re.compile(r"(\{.*?\}|$$.*?$$)", re.DOTALL)
        match = json_pattern.search(output)
        if match:
            json_content = match.group(1)
        else:

            raise ValueError("No JSON content found in the model output.")

    json_content = json_content.strip()

    return json_content


def save_numbered_code(numbered_code, pipeline_number, output_dir="numbered_references"):
    os.makedirs(output_dir, exist_ok=True)

    filename = f"pipeline_{pipeline_number}_numbered.txt"
    filepath = os.path.join(output_dir, filename)

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


def save_raw_output(model_output, pipeline_id, model_name, configuration, output_dir="raw_results"):
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{pipeline_id}_{model_name}_{configuration}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(model_output)

    return filepath


def main():
    issue_names = []
    issue_descriptions = []

    with open("example_pipelines/issues_names.txt", "r") as file:
        for line in file:
            issue_names.append(line.strip())

    with open("example_pipelines/issues_descriptions.txt", "r") as file:
        for line in file:
            issue_descriptions.append(line.strip())

    pipelines = []

    for root, dirs, files in os.walk("example_pipelines"):
        for file in files:
            if file == "example-0.py":
                pipelines.append(os.path.join(root, file))
    dags = []
    for root, dirs, files in os.walk("example_pipelines"):
        for file in files:
            if file == "dag_2.json":
                dags.append(os.path.join(root, file))
    output_res = []
    feedback_list = []

    for i, pipeline in enumerate(pipelines):
        original_code = read_code_from_file(pipeline)
        numbered_code = add_line_numbers(original_code)
        save_numbered_code(numbered_code, issue_names[i])
        prompt = create_prompt_with_issue_description(issue_names[i], issue_descriptions[i], numbered_code)
        prompt_with_dag = create_prompt_with_dag(issue_names[i], issue_descriptions[i], dags[i])

        # OpenAI GPT-4
        output_gpt4 = check_code_with_gpt4(prompt)
        save_raw_output(output_gpt4, i, "gpt4", issue_names[i], "raw_results")
        output_gpt4 = check_code_with_gpt4(prompt_with_dag)
        save_raw_output(output_gpt4, i, "gpt4_with_dag", issue_names[i], "raw_results")
        # with open(f"results/output_gpt4_{i}.json", "w") as f:
        #    f.write(output_gpt4)
        # gpt4_data = process_model_output(output_gpt4, "gpt4", i)
        # feedback_list.append(gpt4_data)
        # output_res.append(analyse_output(output_gpt4, "OpenAI GPT 4", i))

        # DeepSeek V3
        output_v3 = check_code_with_deepseek(prompt, 0)
        save_raw_output(output_v3, i, "deepseek_v3", issue_names[i], "raw_results")
        output_v3 = check_code_with_deepseek(prompt_with_dag, 0)
        save_raw_output(output_v3, i, "deepseek_v3_with_dag", issue_names[i], "raw_results")
        # with open(f"results/output_v3_{i}.json", "w") as f:
        #    f.write(output_v3)
        # v3_data = process_model_output(output_v3, "DeepSeek V3", i, clean=True)
        # feedback_list.append(v3_data)
        # output_res.append(analyse_output(output_v3, "DeepSeek V3", i))

        # DeepSeek R1

        output_r1 = check_code_with_openrouter(prompt)
        save_raw_output(output_r1, i, "deepseek_r1", issue_names[i], "raw_results")
        output_r1 = check_code_with_openrouter(prompt_with_dag)
        save_raw_output(output_r1, i, "deepseek_r1_with_dag", issue_names[i], "raw_results")
        # with open(f"results/output_r1_{i}.json", "w") as f:
        #    f.write(output_r1)
        # r1_data = process_model_output(output_r1, "DeepSeek R1", i, clean=True)
        # feedback_list.append(r1_data)
        # output_res.append(analyse_output(output_r1, "DeepSeek R1", i))

        # with open(f"results/output_models.json", "w") as f:
        #    json.dump(feedback_list, f, indent=4)
        # with open(f"results/output_res.json", "w") as f:
        #    json.dump(output_res, f, indent=4)
        if i == 0:
            break


if __name__ == "__main__":
    main()
