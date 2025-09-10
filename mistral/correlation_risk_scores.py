import json
import os
import pdb
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


risk_scores_dir = Path("llama_zero_shot")
r2_vals = []
ece_vals = []
auc_vals = []
datapoints = []

if __name__ == "__main__":
    for f in os.listdir(risk_scores_dir):
        assert len(os.listdir(risk_scores_dir / f)) == 1
        timestamp = os.listdir(risk_scores_dir / f)[0]
        pre_llm_dir = Path(f"{risk_scores_dir}/{f}/{timestamp}/reentry")
        assert len(os.listdir(pre_llm_dir)) == 1
        results_dir = pre_llm_dir / os.listdir(pre_llm_dir)[0]
        for results_file in os.listdir(results_dir):
            if results_file[-5:] == ".json":  # metrics
                with open(results_dir / results_file, "r") as fi:
                    metrics_json = json.load(fi)
            if results_file[-4:] == ".csv":  # risk scores
                risk_score_df = pd.read_csv(results_dir / results_file)

        avg_risk = np.mean(np.abs(risk_score_df["risk_score"] - np.round(risk_score_df["risk_score"])))
        std_risk = np.std(risk_score_df["risk_score"])
        std_risk2 = np.std(np.abs(risk_score_df["risk_score"] - np.round(risk_score_df["risk_score"])))
        label_imbalance = max(np.mean(risk_score_df["label"]), 1 - np.mean(risk_score_df["label"]))
        auc = metrics_json["roc_auc"]
        ece = metrics_json["ece"]

        # regress risk scores on label
        risk_scores = risk_score_df["risk_score"].tolist()
        labels = risk_score_df["label"].tolist()

        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        model = LinearRegression()
        X, y = np.array(risk_scores).reshape(-1, 1), np.array(labels)
        model.fit(X, y)
        # Predict
        y_pred = model.predict(X)

        # Calculate R²
        r_squared = r2_score(y, y_pred)

        print(f"R-squared: {r_squared}")

        # plot datapoints
        plt.scatter(risk_scores, labels, label="Data points")
        plt.plot(X, y_pred, color="red", label=f"Best fit line\n$R^2$ = {r_squared:.3f}")
        plt.xlabel("risk score")
        plt.ylabel("label")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f"calibration_regression/{f.split('_')[0]}.png")
        plt.clf()

        r2_vals.append(r_squared)
        ece_vals.append(ece)
        auc_vals.append(auc)

        # if auc < 0.5 and std_risk > 0.4: continue

        # add average risk, metric to datapoints
        # datapoints.append([avg_risk, auc / xgb_map[f.split("_")[0]], std_risk, label_imbalance])
        datapoints.append([avg_risk, auc, std_risk, label_imbalance])

    # box plot r2, ece, auc
    for l, name in [(r2_vals, "r2"), (ece_vals, "ece"), (auc_vals, "auc")]:
        # Create a box plot for the single list
        plt.figure(figsize=(6, 4))
        plt.boxplot(l, vert=True)
        plt.title(f"Box Plot of {name}")
        plt.ylabel(f"{name}")
        plt.xticks([1], [name])  # Label for the single box
        plt.grid(True)
        plt.savefig(f"calibration_regression/box_plot_{name}.png")

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    model = LinearRegression()
    X, y = np.array([x[2] for x in datapoints]).reshape(-1, 1), np.array([x[1] for x in datapoints])
    model.fit(X, y)
    # Predict
    y_pred = model.predict(X)

    # Calculate R²
    r_squared = r2_score(y, y_pred)

    print(f"R-squared: {r_squared}")

    # plot datapoints
    plt.scatter([(x[0]) for x in datapoints], [x[1] for x in datapoints], c=[x[-1] for x in datapoints])
    plt.colorbar(label="Imbalance")
    plt.xlabel("average(risk score - round(risk score))")
    plt.ylabel("auc score")
    plt.savefig("risk_score_correlation.png")
    plt.clf()

    plt.scatter(X, y, c=[x[-1] for x in datapoints], label="Data points")
    plt.colorbar(label="Imbalance", orientation="horizontal")
    plt.plot(X, y_pred, color="red", label=f"Best fit line\n$R^2$ = {r_squared:.3f}")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("sd(risk score)")
    plt.ylabel("auc score")
    plt.tight_layout()
    plt.savefig("risk_std_correlation.png")
    plt.clf()

    plt.scatter([x[3] for x in datapoints], [x[1] for x in datapoints])
    plt.xlabel("frequency of majority class")
    plt.ylabel("auc score")
    plt.savefig("imbalance_correlation.png")
    plt.clf()
