import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

result_files = ["deepseek_v3_results.csv", "gemini-2.0-flash-001_results.csv",
                "gpt-4o-mini_results.csv", "deepseek_r1_reasoner_results.csv"]

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the ground truth file
with open("../example_pipelines2/issues_names.txt", "r") as file:
    ground_truth = file.read().splitlines()

# List of result files

# Initialize an empty DataFrame to store combined results
combined_results = pd.DataFrame()

# Process each result file
for file in result_files:
    # Load the result file
    df = pd.read_csv(file)

    # Add the ground truth column
    df["ground_truth"] = ground_truth

    # Add a column to check if the model's category matches the ground truth
    df["is_correct"] = df["category"] == df["ground_truth"]

    # Add a column to identify the model (based on the filename)
    df["model"] = file.replace(".csv", "")

    # Append to the combined results DataFrame
    combined_results = pd.concat([combined_results, df], ignore_index=True)


# Initialize an empty DataFrame to store the comparison table
comparison_table = pd.DataFrame()

# Process each result file
for file in result_files:
    # Load the result file
    df = pd.read_csv(file)

    # Add the ground truth column
    df["ground_truth"] = ground_truth

    # Add a column to check if the model's category matches the ground truth
    df["is_correct"] = df["category"] == df["ground_truth"]

    # Add a column to identify the model (based on the filename)
    df["model"] = file.replace(".csv", "")

    # Append to the comparison table
    comparison_table = pd.concat([comparison_table, df], ignore_index=True)

# Select relevant columns for the table
comparison_table = comparison_table[["pipeline_code", "model", "category", "ground_truth", "is_correct"]]

# Display the comparison table
print("Comparison Table:")
print(comparison_table)

# Save the table to a CSV file (optional)
comparison_table.to_csv("comparison_table.csv", index=False)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the comparison table (if not already loaded)
comparison_table = pd.read_csv("comparison_table.csv")

# Visualization 1: Accuracy of Each Model
plt.figure(figsize=(10, 6))
accuracy = comparison_table.groupby("model")["is_correct"].mean().reset_index()
sns.barplot(x="model", y="is_correct", data=accuracy, palette="viridis")
plt.title("Accuracy of Each Model")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)  # Set y-axis limits for accuracy
plt.show()

# Visualization 2: Distribution of Correct vs Incorrect Predictions for Each Model
plt.figure(figsize=(12, 6))
sns.countplot(x="model", hue="is_correct", data=comparison_table, palette="viridis")
plt.title("Distribution of Correct vs Incorrect Predictions for Each Model")
plt.xlabel("Model")
plt.ylabel("Count")
plt.legend(title="Correct Prediction", labels=["Incorrect", "Correct"])
plt.show()

# Set visual style
sns.set_theme(style="whitegrid", palette="pastel")
plt.figure(figsize=(14, 10))

# 1. Overall Accuracy Plot (Fixed hue/palette warning)
plt.subplot(2, 2, 1)
accuracy_plot = sns.barplot(
    data=combined_results,
    x="model",
    y="is_correct",
    hue="model",  # Added hue parameter
    palette="viridis",
    estimator="mean",
    errorbar=None,
    legend=False  # Disabled legend
)
plt.title("Overall Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
accuracy_plot.bar_label(accuracy_plot.containers[0], fmt='%.2f')

# 2. Category-wise Accuracy Heatmap
plt.subplot(2, 2, 2)
heatmap_data = combined_results.groupby(['ground_truth', 'model'])['is_correct'].mean().unstack()
sns.heatmap(
    heatmap_data.T,
    annot=True,
    fmt=".2f",  # Proper format for float values
    cmap="YlGnBu",
    cbar_kws={'label': 'Accuracy'}
)
plt.title("Category-wise Accuracy")
plt.xlabel("Ground Truth Category")
plt.ylabel("Model")
plt.xticks(rotation=45)

# 3. Confusion Matrices Grid (Fixed format error)
plt.subplot(2, 1, 2)
# Convert counts to integers and fill NaN with 0
confusion_data = combined_results.groupby(['model', 'ground_truth', 'category']).size().unstack().fillna(0).astype(int)
sns.heatmap(
    confusion_data,
    cmap="Blues",
    annot=True,
    fmt="d",  # Now safe for integers
    cbar_kws={'label': 'Count'}
)
plt.title("Prediction Distribution Across Categories")
plt.xlabel("Predicted Category")
plt.ylabel("Model & Ground Truth")

plt.tight_layout()
plt.show()
