import pandas as pd

# Load both CSV files
gpt_df = pd.read_csv("gpt.csv")
llama_df = pd.read_csv("llama.csv")

# Select and rename relevant columns
gpt_df = gpt_df[["dataset_name", "auc", "ece"]].rename(columns={
    "auc": "gpt_auc",
    "ece": "gpt_ece"
})

llama_df = llama_df[["dataset_name", "auc", "ece"]].rename(columns={
    "auc": "llama_auc",
    "ece": "llama_ece"
})

# Merge the two DataFrames on 'dataset_name'
merged_df = pd.merge(gpt_df, llama_df, on="dataset_name")

# Optional: Save to CSV or LaTeX
merged_df.to_csv("joined_results.csv", index=False)
# print(merged_df.to_latex(index=False, float_format="%.3f"))

# Print the merged DataFrame
print(merged_df)
