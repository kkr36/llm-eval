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
confidence_scores_dir = Path("past_results/gpt_confidence")
risk_score_results = {}
confidence_results = {}

if __name__ == "__main__":
    xgb_vals = pd.read_csv("xgb.csv")
    xgb_map = {name: auc for (name, auc) in zip(xgb_vals["dataset_name"].tolist(), xgb_vals["auc"].tolist())}

    # get all risk score info
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
        label_imbalance = max(np.mean(risk_score_df["label"]), 1 - np.mean(risk_score_df["label"]))
        auc = metrics_json["roc_auc"]

        # add average risk, metric to datapoints
        # risk_score_results[f.split("_")[0]] = [avg_risk, auc / xgb_map[f.split("_")[0]], std_risk, label_imbalance]
        risk_score_results[f.split("_")[0]] = [avg_risk, auc, std_risk, label_imbalance]

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(risk_score_results, orient="index")

    # Optional: rename columns
    df.columns = ["average_risk_score", "auc", "std_risk_scores", "label_imbalance"]
    # Write to CSV
    df = df.reset_index().rename(columns={"index": "dataset_name"})

    df.to_csv("zero_shot.csv", index=False)

    # get all confidence score info

    for f in os.listdir(confidence_scores_dir):
        assert len(os.listdir(confidence_scores_dir / f)) == 1
        timestamp = os.listdir(confidence_scores_dir / f)[0]
        openai_dir = Path(f"{confidence_scores_dir}/{f}/{timestamp}/confidence")
        confidence_scores_df = pd.read_csv(openai_dir / "uncertainty_scores.csv")

        confidence_results[f.split("_")[0]] = [
            np.mean(confidence_scores_df["uncertainty_score"]),
            np.std(confidence_scores_df["uncertainty_score"]),
        ]

    assert len(confidence_results) == len(risk_score_results)

    keys = [key for key in confidence_results]
    keys.sort()

    model = LinearRegression()
    X, y = (
        np.array([(confidence_results[key][0]) for key in keys]).reshape(-1, 1),
        np.array([risk_score_results[key][1] for key in keys]),
    )
    model.fit(X, y)
    # Predict
    y_pred = model.predict(X)

    # Calculate R²
    r_squared = r2_score(y, y_pred)

    print(f"R-squared: {r_squared}")

    # plot datapoints
    plt.scatter(X, y, c=[risk_score_results[x][-1] for x in keys], label="Data points")
    plt.colorbar(label="Imbalance", orientation="horizontal")
    plt.plot(X, y_pred, color="red", label=f"Best fit line\n$R^2$ = {r_squared:.3f}")
    plt.xlabel("average confidence score")
    plt.ylabel("auc score")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("confidence_score_correlation.png")
    plt.clf()

    X2 = np.array([(confidence_results[key][1]) for key in keys]).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X2, y)
    # Predict
    y_pred = model.predict(X2)

    # Calculate R²
    r_squared = r2_score(y, y_pred)

    print(f"R-squared: {r_squared}")
    plt.scatter(X2, y, c=[risk_score_results[x][-1] for x in keys], label="Data points")
    plt.colorbar(label="Imbalance", orientation="horizontal")
    plt.plot(X2, y_pred, color="red", label=f"Best fit line\n$R^2$ = {r_squared:.3f}")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("std confidence score")
    plt.ylabel("auc score")
    plt.tight_layout()
    plt.savefig("std_confidence_score_correlation.png")
    plt.clf()
