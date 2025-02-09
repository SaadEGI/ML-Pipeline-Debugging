import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI



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