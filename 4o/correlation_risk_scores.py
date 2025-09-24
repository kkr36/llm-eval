import json
import os
import pdb
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


risk_scores_dir = Path("past_results/gpt_zero_shot")
r2_vals = []
ece_vals = []
auc_vals = []
datapoints = []

if __name__ == "__main__":
    xgb_vals = pd.read_csv("xgb.csv")
    xgb_map = {name: auc for (name, auc) in zip(xgb_vals["dataset_name"].tolist(), xgb_vals["auc"].tolist())}

    for f in os.listdir(risk_scores_dir):
        assert len(os.listdir(risk_scores_dir / f)) == 1
        timestamp = os.listdir(risk_scores_dir / f)[0]
        openai_dir = Path(f"{risk_scores_dir}/{f}/{timestamp}/reentry/openai")
        assert len(os.listdir(openai_dir)) == 1
        bench_dir = os.listdir(openai_dir)[0]
        results_dir = openai_dir / bench_dir
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
        absolute_error = np.abs(risk_score_df["risk_score"] - risk_score_df["label"])

        model = LinearRegression()
        X, y = np.array(risk_scores).reshape(-1, 1), np.array(absolute_error)
        model.fit(X, y)
        # Predict
        y_pred = model.predict(X)

        # Calculate R²
        r_squared = r2_score(y, y_pred)

        print(f"R-squared: {r_squared}")

        # plot datapoints
        plt.scatter(risk_scores, absolute_error, label="Data points")
        plt.plot(X, y_pred, color="red", label=f"Best fit line\n$R^2$ = {r_squared:.3f}")
        plt.xlabel("risk score")
        plt.ylabel("label")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f"calibration_regression/{f.split('_')[0]}_abs_err.png")
        plt.clf()

        r2_vals.append(r_squared)
        ece_vals.append(ece)
        auc_vals.append(auc)
        # if auc < 0.5 and std_risk > 0.4: continue

        # add average risk, metric to datapoints
        # datapoints.append([avg_risk, auc / xgb_map[f.split("_")[0]], std_risk, label_imbalance])
        datapoints.append([avg_risk, auc, std_risk, label_imbalance])

        # box plot r2, ece, auc
    for l, name in [(r2_vals, "r2"), (ece_vals, "ECE"), (auc_vals, "AUC")]:
        # Create a box plot for the single list
        plt.figure(figsize=(6, 4))
        plt.boxplot(l, vert=True)
        plt.title(f"Box Plot of {name}, GPT-4o-mini")
        plt.ylabel(f"{name}")
        plt.xticks([1], [name])  # Label for the single box
        plt.grid(True)
        plt.savefig(f"calibration_regression/box_plot_{name}.png")
        plt.clf()

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
    plt.savefig("_figs_gpt/risk_score_correlation.png")
    plt.clf()

    plt.scatter(X, y, c=[x[-1] for x in datapoints], label="Data points")
    plt.colorbar(label="Imbalance", orientation="horizontal")
    plt.plot(X, y_pred, color="red", label=f"Best fit line\n$R^2$ = {r_squared:.3f}")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("sd(risk score)")
    plt.ylabel("auc score")
    plt.tight_layout()
    plt.savefig("_figs_gpt/risk_std_correlation.png")
    plt.clf()

    plt.scatter([x[3] for x in datapoints], [x[1] for x in datapoints])
    plt.xlabel("frequency of majority class")
    plt.ylabel("auc score")
    plt.savefig("_figs_gpt/imbalance_correlation.png")
    plt.clf()
