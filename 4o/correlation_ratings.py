import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


expnames = ["auc", "auc_logprob", "probs", "probs_logprob", "score", "score_logprob"]

if __name__ == "__main__":
    # zero shot scores
    zero_shot_path = "zero_shot.csv"
    zero_shot_data = pd.read_csv(zero_shot_path)

    for expname in expnames:
        csvpath = f"results/{expname}.csv"
        data = pd.read_csv(csvpath)
        full_data = pd.merge(zero_shot_data, data, on="dataset_name", how="inner")
        assert len(full_data == len(zero_shot_data))

        auc, results = full_data["auc"].tolist(), full_data["rating"].tolist()

        # plotting

        xname = expname if "auc" not in expname else expname.replace("auc", "predicted_auc")

        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        model = LinearRegression()
        X, y = np.array(results).reshape(-1, 1), np.array(auc)
        model.fit(X, y)
        # Predict
        y_pred = model.predict(X)

        # Calculate R²
        r_squared = r2_score(y, y_pred)

        print(f"R-squared: {r_squared}")

        # plot datapoints
        plt.scatter(X, y, c=full_data["label_imbalance"].tolist(), label="Data points")
        plt.colorbar(label="Imbalance", orientation="horizontal")
        plt.plot(X, y_pred, color="red", label=f"Best fit line\n$R^2$ = {r_squared:.3f}")
        plt.xlabel(xname)
        plt.ylabel("auc score")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f"correlation_{xname}.png")
        plt.clf()
