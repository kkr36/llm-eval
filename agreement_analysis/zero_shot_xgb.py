import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

plt.rc("axes",titlesize=20)
plt.rc("axes",labelsize=20)
plt.rc("font",size=15)

if __name__ == "__main__":
    xgb_csv = pd.read_csv("xgb.csv")
    for llm_path in ["gpt", "llama", "gpt_4o", "mistral"][2:]:
        llm_csv = pd.read_csv(f"{llm_path}.csv")
        merged_df = pd.merge(llm_csv, xgb_csv, on='dataset_name', how='inner')
        x_col = merged_df['xgb_auc'].tolist()
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
        plt.figure(figsize=(5, 5))
        plt.scatter(X, y, label='Data points')
        plt.plot(X, y_pred, color='red', label=f'Best fit line\n$R^2$ = {r_squared:.3f}')
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(f"XGB AUC")
        plt.ylabel(f"{llm_path.upper()} AUC")
        plt.text(.68,0,fr'$R^2=${r_squared:.3f}',ha='right',va='bottom',transform=plt.gca().transAxes)
        plt.plot([i/10 for i in range(4,11)], [i/10 for i in range(4,11)], linestyle=':', color='black', label='y = x')
        plt.xlim((.4,1.05))
        plt.ylim((.4,1.05))
        plt.tight_layout()
        plt.savefig(f"results_section_1/{llm_path}_xgb_correlation.pdf", format="pdf", bbox_inches="tight")
        plt.clf()
        