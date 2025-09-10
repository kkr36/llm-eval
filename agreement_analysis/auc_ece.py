import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


if __name__ == "__main__":
    for llm in ["gpt", "llama"]:
        csv = pd.read_csv(f"{llm}.csv")

        x_col = csv["auc"]
        y_col = csv["ece"].tolist()

        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        model = LinearRegression()
        X, y = np.array(x_col).reshape(-1, 1), np.array(y_col)
        model.fit(X, y)
        # Predict
        y_pred = model.predict(X)

        # Calculate R²
        r_squared = r2_score(y, y_pred)

        plt.scatter(x_col, y_col, label="Data points")
        plt.plot(X, y_pred, color="red", label=f"Best fit line\n$R^2$ = {r_squared:.3f}")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        # plt.title(f"R^2 = {round(r_squared, 3)}")
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(f"auc scores")
        plt.ylabel(f"ece scores")
        plt.tight_layout()
        plt.savefig(f"{llm}_auc/auc_ece.png")
        plt.clf()
