import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


plt.rc("axes", titlesize=20)
plt.rc("axes", labelsize=20)
plt.rc("font", size=14)

# Load the CSV files into dataframes
gpt_df = pd.read_csv("gpt.csv")
llama_df = pd.read_csv("llama.csv")
gpt4o_df = pd.read_csv("gpt_4o.csv")
mistral_df = pd.read_csv("mistral.csv")

# Add a column to specify the model type
gpt_df["model"] = "GPT"
llama_df["model"] = "Llama"
gpt4o_df["model"] = "GPT 4o"
mistral_df["model"] = "Mistral"

for name in ["AUC", "ECE"]:
    for llm, df in [("GPT", gpt_df), ("Llama", llama_df), ("GPT 4o", gpt4o_df), ("Mistral", mistral_df)]:
        plt.figure(figsize=(6, 4))
        # plt.boxplot(l, vert=True)
        plt.hist(df[name.lower()], bins=10, color="skyblue", edgecolor="black")
        # plt.title(f'Histogram of {name.upper()}, {llm}')
        plt.xlabel(f"{name}")
        plt.ylabel("Frequency")
        # plt.xticks([1], [name])  # Label for the single box
        # plt.grid(True)
        plt.savefig(f"results_section_1/histogram_{name}_{llm}.pdf", format="pdf", bbox_inches="tight")
        plt.clf()


# # Combine both dataframes into one
# df = pd.concat([gpt_df, llama_df], ignore_index=True)

# # First Image: Box plots for AUC scores
# fig1, axes1 = plt.subplots(1, 2, figsize=(12, 6))

# # Box plot for GPT AUC
# sns.boxplot(x='model', y='auc', data=gpt_df, ax=axes1[0])
# axes1[0].set_title('GPT AUC')
# axes1[0].set_ylabel('AUC')

# # Box plot for Llama AUC
# sns.boxplot(x='model', y='auc', data=llama_df, ax=axes1[1])
# axes1[1].set_title('Llama AUC')
# axes1[1].set_ylabel('AUC')

# # Set the same y-axis limits for both plots

# for ax in axes1:
#     ax.set_ylim(0, 1)

# # Adjust layout and save figure
# plt.tight_layout()
# fig1.savefig('auc_comparison.pdf', format="pdf", bbox_inches="tight")

# # Second Image: Box plots for ECE scores
# fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))

# # Box plot for GPT ECE
# sns.boxplot(x='model', y='ece', data=gpt_df, ax=axes2[0])
# axes2[0].set_title('GPT ECE')
# axes2[0].set_ylabel('ECE')

# # Box plot for Llama ECE
# sns.boxplot(x='model', y='ece', data=llama_df, ax=axes2[1])
# axes2[1].set_title('Llama ECE')
# axes2[1].set_ylabel('ECE')

# for ax in axes2:
#     ax.set_ylim(0, 1)

# # Adjust layout and save figure
# plt.tight_layout()
# fig2.savefig('ece_comparison.pdf', format="pdf", bbox_inches="tight")
