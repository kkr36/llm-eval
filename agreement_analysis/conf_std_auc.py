import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    label_csv = pd.read_csv("dataset_labels.csv")
    categories = label_csv['label'].unique()
    color_map = {category: color for category, color in zip(categories, plt.cm.Set1.colors)}

    for llm_path in ["gpt", "llama"]:

        merged_df = pd.read_csv(f"{llm_path}.csv")

        x_col = merged_df['confidence_score']
        y_col = merged_df['std_risk_scores'].tolist()

        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        model = LinearRegression()
        X, y  = np.array(x_col).reshape(-1,1), np.array(y_col)
        model.fit(X, y)
        # Predict
        y_pred = model.predict(X)

        # Calculate R²
        r_squared = r2_score(y, y_pred)

        plt.scatter(x_col, y_col, c=merged_df["auc"], label='Data points')
        plt.plot(X, y_pred, color='red', label=f'Best fit line\n$R^2$ = {r_squared:.3f}')
        plt.colorbar(label='auc', orientation='horizontal')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.title(f"R^2 = {round(r_squared, 3)}")
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(f"average confidence score")
        plt.ylabel(f"std risk scores")
        plt.tight_layout()
        plt.savefig(f"{llm_path}_auc/conf_std_auc.png")
        plt.clf()
    