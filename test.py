import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # Required for array handling

# imort json file and convert to pandas dataframe
import json
with open('results/output_res.json') as f:
    output_res = json.load(f)
models = ['Ground Truth', 'gpt4', 'DeepSeek V3', 'DeepSeek R1']

data2 = {
        'Model': models,
        'Class_NoVuln': ["Pipeline w/ Aggr Err", "Pipeline w/ Aggr Err", "Pipeline w/ Aggr Err","Pipeline w/ Aggr Err"],
        'Class_Vuln': ["Pipeline w/ Ann Err", "Pipeline w/ Ann Err", "Pipeline w/ Ann Err", "Pipeline w/ Ann Err"],
        'Aggr_detected': ["-", 0, 0, 0],
        'Aggr_Lines': [[28, 31],  0, 0, 0],
        'Ann_detected': ["-", 0, 0, 0],
        'Ann_Lines': [[27, 28], 0, 0, 0]
    }
# output res looks like this [[True, 0.9, [31], 'gpt4', 0], [True, 0.9, [31], 'DeepSeek V3', 0], [True, 0.95, [31], 'DeepSeek R1', 0], [False, 1, [], 'gpt4', 1], [False, 1.0, [], 'DeepSeek V3', 1], [False, 0.1, [], 'DeepSeek R1', 1]], first one is issue detected or not,
# second one is the confidence, third one is the line number, fourth one is the model name, fifth one is the pipeline or annotation
# recorrect the data based on the class no vuln and class vuln do not change
for i in range(len(output_res)):
    if output_res[i][4] == 0:
        if output_res[i][0] == True:
            data2['Aggr_detected'][models.index(output_res[i][3])] = output_res[i][1]
            data2['Aggr_Lines'][models.index(output_res[i][3])] = output_res[i][2]
        else:
            data2['Ann_detected'][models.index(output_res[i][3])] = output_res[i][1]
            data2['Ann_Lines'][models.index(output_res[i][3])] = output_res[i][2]
print(data2)
print(output_res)

# Data from the table
data = {
    'Model': ['Ground Truth', 'OpenAI GPT 4', 'Deepseek V3', 'Deepseek R1'],
    'Class_NoVuln': ["Pipeline w/ Aggr Err", "Pipeline w/ Aggr Err", "Pipeline w/ Aggr Err","Pipeline w/ Aggr Err"],
    'Class_Vuln': ["Pipeline w/ Ann Err", "Pipeline w/ Ann Err", "Pipeline w/ Ann Err", "Pipeline w/ Ann Err"],
    'Aggr_detected': ["-", 0.97, 0.96, 0.93],
    'Ann_detected': [0.45, 0.33, 0.37, 0.58],
    'Aggr_Lines': [0.96, 0.76, 0.83, 0.95],
    'Ann_Lines': [[31,5], 0.85, 0.73, 0.53]  # Now supports integers and arrays
}

df = pd.DataFrame(data)

# Helper function to format different value types
def format_value(value):
    if isinstance(value, (list, np.ndarray)):  # Handle arrays
        return ', '.join(map(str, value))
    elif isinstance(value, (int, float)):     # Handle numbers
        return f"{value:.2f}" if not float(value).is_integer() else f"{int(value)}"
    else:                                     # Handle strings and others
        return str(value)

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))
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
table_title = 'TABLE II: Comparison among the different approaches.'

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

# Header styling
for (pos, cell) in table.get_celld().items():
    if pos[0] == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#F0F0F0')

# Border lines (removed middle horizontal lines)
ax.axhline(y=0, color='black', linewidth=1.0, clip_on=False)          # Top line
ax.axhline(y=len(table_vals), color='black', linewidth=1.0, clip_on=False)  # Bottom line
ax.axhline(y=1, color='black', linewidth=0.7, linestyle='-', clip_on=False)  # Header separator

# Title
ax.set_title(table_title, y=1.05, fontsize=12)

plt.tight_layout()
plt.show()
