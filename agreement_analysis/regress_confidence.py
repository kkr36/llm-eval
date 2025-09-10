import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


plt.rc("axes", titlesize=26)
plt.rc("axes", labelsize=26)
plt.rc("font", size=20)

if __name__ == "__main__":
    label_csv = pd.read_csv("dataset_labels.csv")
    categories = label_csv["label"].unique()
    color_map = {category: color for category, color in zip(categories, plt.cm.Set1.colors)}
    # experiment_cols = ["average_risk_score", "std_risk_scores", "confidence_score", "std_confidence_score", "masking", "std_masking", "predicted_auc", "probs", "score", "ece"][:-1]
    experiment_cols = ["std_risk_scores", "average_risk_score", "confidence_score", "predicted_auc", "probs", "score"]
    gpt_stats = []
    llama_stats = []

    for llm_path in ["gpt", "llama", "gpt_4o", "mistral"]:
        # for llm_path in ["results_list_template", "results_text_template"]:

        llm_csv = pd.read_csv(f"{llm_path}.csv")
        merged_df = pd.merge(llm_csv, label_csv, on="dataset_name", how="inner")

        for i, col1 in enumerate(experiment_cols):
            for j, col2 in enumerate(experiment_cols[i + 1 :]):
                x_col = merged_df[col1]
                y_col = merged_df[col2].tolist()

                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score

                model = LinearRegression()
                X, y = np.array(x_col).reshape(-1, 1), np.array(y_col)
                model.fit(X, y)
                # Predict
                y_pred = model.predict(X)

                # Calculate R²
                r_squared = r2_score(y, y_pred)

                plt.scatter(X, y, label="Data points")
                # for category in categories:
                #     subset = merged_df[merged_df['label'] == category]
                #     plt.scatter(subset[col], subset['auc'], label=category, color=color_map[category])
                # plt.plot(X, y_pred, color='red', label=f'Best fit line\n$R^2$ = {r_squared:.3f}')
                plt.plot(X, y_pred, color="red")
                # if col != "confidence_score" and col != "score":
                plt.text(0.9, 0, rf"$R^2=${r_squared:.3f}", ha="right", va="bottom", transform=plt.gca().transAxes)
                # else:
                #     plt.text(.68,0,fr'$R^2=${r_squared:.3f}',ha='right',va='bottom',transform=plt.gca().transAxes)

                # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                # plt.title(f"R^2 = {round(r_squared, 3)}")
                # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.xlabel(col1)
                plt.ylabel(col2)
                plt.tight_layout()
                import os

                if not os.path.exists(f"regress_confidence/{llm_path}"):
                    os.makedirs(f"regress_confidence/{llm_path}")
                plt.savefig(f"regress_confidence/{llm_path}/{col1}_{col2}.pdf", format="pdf", bbox_inches="tight")
                plt.clf()
