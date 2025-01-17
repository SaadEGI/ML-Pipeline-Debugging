import openai
import json
import os


openai.api_key = os.getenv("OPENAI_API_KEY")


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


def analyze_code(prompt):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    output = response.choices[0].message.content
    return output

def create_prompt_with_issue_description(issue_description, code):
    prompt = f"""
          You are an AI assistant that analyzes Python machine learning pipeline code to identify potential issues related to the code, data, and transformations, and provide actionable feedback for correction.

          Given the following pipeline code:{os.linesep}
          ```python{os.linesep}
          {code}{os.linesep}
          ```
          The following issue description is provided:{os.linesep}{issue_description}{os.linesep}

          Analyze the code and consider the issue description. Identify any potential problems, and provide a list of detected issues with suggested fixes.

          Please output the results in the following JSON format:{os.linesep}
          [
            {{
              "issue_description": "Description of the problem",
              "suggested_fix": "Suggestion on how to fix the problem",
              "code_snippet": "Optional code illustrating the fix"
            }},
            ...
          ]
          """
    return prompt


def create_prompt(code):
    prompt = f"""
            You are an AI assistant that analyzes Python machine learning pipeline code to identify potential issues related to the code, data, and transformations, and provide actionable feedback for correction.

            Given the following pipeline code:

            {code}
            ```

            Analyze the code and Identify any potential problems, and provide a list of detected issues with suggested fixes.

            Please output the results in the following JSON format:

            [
              {{
                "issue_description": "Description of the problem",
                "suggested_fix": "Suggestion on how to fix the problem",
                "code_snippet": "Optional code illustrating the fix"
              }},
              ...
            ]
          """
    return prompt

def create_evaluation_prompt(original_code, corrected_code, feedback_list):


    issues_and_fixes = "\n\n".join([
        f"Issue {i + 1}:\nDescription: {feedback['issue_description']}\nSuggested Fix: {feedback['suggested_fix']}"
        for i, feedback in enumerate(feedback_list)
    ])

    prompt = f"""
    You are an AI assistant specialized in **evaluating** machine learning pipeline code corrections.

    **Your Sole Focus**: Determine whether the **previously identified issues** have been resolved in the provided **100% corrected code**.
    
    **Important Instructions**:
    
    - **Do NOT** find new issues or suggest additional improvements.
    - **Only** assess the resolution of the issues you previously identified.

    **Original Code and Issues:**
    
    Here is the original pipeline code which you analyzed before, and the issues you previously identified with your analysis:    
    {original_code}
    
    Your Previous Analysis (Issues & Fixes):
 
    {issues_and_fixes}
    
    Now, here's the ground truth, a 100% corrected version of the same pipeline code:
    
    {corrected_code}
    
    Evaluation Task:

    Your task is to strictly compare your previous issues with the corrected code. For each issue you previously identified, indicate whether that specific issue was resolved in the 100% correct pipeline.

    Do NOT look for new problems or suggest additional fixes.
    Focus ONLY on the issues you previously identified.

    
    Output Format

    Provide the following for each issue in JSON format:
    
    json
    
    [
      {{
        "issue_number": 1,
        "issue_description": "Exact description of the first issue as you previously described it.",
        "resolved": "Yes" or "No",
        "explanation": "Brief explanation based on the corrected code of why the issue was resolved or why it remains unresolved. Focus on what you previously identified, not on new problems."
      }},
      {{
        "issue_number": 2,
        "issue_description": "Exact description of the second issue as you previously described it.",
        "resolved": "Yes" or "No",
        "explanation": "Brief explanation based on the corrected code of why the issue was resolved or why it remains unresolved. Focus on what you previously identified, not on new problems."
      }},
      ...
    ] 
    Remember: Focus only on evaluating whether the issues you previously identified have been resolved in the corrected code. Do NOT analyze for new issues.  
    """
    return prompt

def evaluate_corrections(prompt):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant for code evaluation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    output = response.choices[0].message.content
    return output

def parse_feedback(model_output):
    try:
        feedback = json.loads(model_output)
        return feedback
    except json.JSONDecodeError as e:
        print("Failed to parse model output:", e)
    return None

def parse_evaluation_output(output):
    try:
        evaluation_results = json.loads(output)
        return evaluation_results
    except json.JSONDecodeError as e:
        print("Failed to parse evaluation output:", e)
        return None


def main():
    pipeline_files = ["corrupted_pipelines/example-0.py", "corrupted_pipelines/example-0-annotation.py"]
    corrected_files = ["corrected_pipelines/example-0-fixed.py", "corrected_pipelines/example-0-fixed-annotation.py"]
    issue_descriptions = ["Aggregation Errors", "Annotation Errors"]

    # without issue specification input
    for i, pipeline_file in enumerate(pipeline_files):
        original_code = read_code_from_file(pipeline_file)
        corrected_code = read_code_from_file(corrected_files[i])
        prompt = create_prompt(original_code)
        model_output = analyze_code(prompt)
        feedback_list = parse_feedback(model_output)

        with open(f"feedback/initial_feedback_pipeline_{i}_without_issue_description.json", "w") as f:
            json.dump(feedback_list, f, indent=2)
        print("Feedback saved to feedback.json")
        evaluation_prompt = create_evaluation_prompt(original_code, corrected_code, feedback_list)
        evaluation_output = evaluate_corrections(evaluation_prompt)
        evaluation_results = parse_evaluation_output(evaluation_output)
        if evaluation_results is not None:
            # save the results to a file with unique name
            with open(f"results/model_evaluation_results_pipeline_{i}_without_issue_description.json", "w") as f:
                json.dump(evaluation_results, f, indent=2)
            print("Evaluation Results saved to evaluation_results.json")
        else:
            print("No evaluation results to display.")

    # with issue specification input
    for i, pipeline_file in enumerate(pipeline_files):
        for j, issue_description in enumerate(issue_descriptions):
            original_code = read_code_from_file(pipeline_file)
            corrected_code = read_code_from_file(corrected_files[i])
            prompt = create_prompt_with_issue_description(issue_description, original_code)
            model_output = analyze_code(prompt)
            feedback_list = parse_feedback(model_output)
            with open(f"feedback/initial_feedback_pipeline_{i}_with_issue_description_{issue_description}.json", "w") as f:
                json.dump(feedback_list, f, indent=2)
            print("Feedback saved to feedback.json")
            evaluation_prompt = create_evaluation_prompt(original_code, corrected_code, feedback_list)
            evaluation_output = evaluate_corrections(evaluation_prompt)
            evaluation_results = parse_evaluation_output(evaluation_output)
            if evaluation_results is not None:
                # save the results to a file with unique name
                with open(f"results/model_evaluation_results_pipeline_{i}_with_{issue_description}.json",
                          "w") as f:
                    json.dump(evaluation_results, f, indent=2)
                print("Evaluation Results saved to evaluation_results.json")
            else:
                print("No evaluation results to display.")


if __name__ == "__main__":
    main()