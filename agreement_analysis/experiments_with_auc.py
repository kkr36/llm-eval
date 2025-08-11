import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

plt.rc("axes",titlesize=26)
plt.rc("axes",labelsize=26)
plt.rc("font",size=20)

exp_name_map = {
    "average_risk_score": "Avg. MCP",
    "std_risk_scores": "Std. Risk Scores",
    "confidence_score": "Avg. Confidence",
    "std_confidence_score": "Std. Confidence",
    "masking": "Avg. Masking",
    "std_masking": "Std. Masking",
    "predicted_auc": "Direct AUC Prediction",
    "probs": "Decimal Scoring (0-1)",
    "score": "Integer Scoring (1-5)",
    "ece": "ECE"
}

if __name__ == "__main__":
    label_csv = pd.read_csv("dataset_labels.csv")
    categories = label_csv['label'].unique()
    color_map = {category: color for category, color in zip(categories, plt.cm.Set1.colors)}
    experiment_cols = ["average_risk_score", "std_risk_scores", "confidence_score", "std_confidence_score", "masking", "std_masking", "predicted_auc", "probs", "score", "ece"][:-1]
    gpt_stats = []
    llama_stats = []

    for llm_path in ["gpt", "llama", "gpt_4o", "mistral"][-2:]:
    # for llm_path in ["results_list_template", "results_text_template"]:

        llm_csv = pd.read_csv(f"{llm_path}.csv")
        merged_df = pd.merge(llm_csv, label_csv, on='dataset_name', how='inner')

        for col in experiment_cols:
            # if col != "std_risk_scores": continue
            # if (col=="masking" and llm_path=="llama") or (col=="std_masking" and llm_path=="llama"): continue
            x_col = merged_df[col]
            y_col = merged_df['auc'].tolist()

            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            model = LinearRegression()
            X, y  = np.array(x_col).reshape(-1,1), np.array(y_col)
            model.fit(X, y)
            # Predict
            y_pred = model.predict(X)

            # Calculate R²
            r_squared = r2_score(y, y_pred)
            if llm_path == "gpt":
                gpt_stats.append(r_squared)
            else:
                llama_stats.append(r_squared)

            plt.scatter(X, y, label='Data points')
            # for category in categories:
            #     subset = merged_df[merged_df['label'] == category]
            #     plt.scatter(subset[col], subset['auc'], label=category, color=color_map[category])
            # plt.plot(X, y_pred, color='red', label=f'Best fit line\n$R^2$ = {r_squared:.3f}')
            plt.plot(X, y_pred, color='red')
            if col != "confidence_score" and col != "score":
                plt.text(.9,0,fr'$R^2=${r_squared:.3f}',ha='right',va='bottom',transform=plt.gca().transAxes)
            else:
                plt.text(.68,0,fr'$R^2=${r_squared:.3f}',ha='right',va='bottom',transform=plt.gca().transAxes)

            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.title(f"R^2 = {round(r_squared, 3)}")
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel(exp_name_map[col])
            plt.ylabel(f"{llm_path.upper()} AUC")
            plt.tight_layout()
            plt.savefig(f"{llm_path}_auc/{col}.pdf", format="pdf", bbox_inches="tight")
            # plt.savefig(f"llama_auc/{col}.pdf", format="pdf", bbox_inches="tight")
            plt.clf()
        
    # data = [
    #     ['llama'] + llama_stats,
    #     ['gpt'] + gpt_stats
    # ]

    # df = pd.DataFrame(data, columns=['llm'] + experiment_cols)

    # # Save to CSV
    # df.to_csv('r2_table.csv', index=False)
        