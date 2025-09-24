import json
import os
import pdb
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


risk_scores_dir = Path("llama_zero_shot")
confidence_scores_dir = Path("results")
risk_score_results = {}
confidence_results = {}
masking_lists = {}

if __name__ == "__main__":
    ### RISK SCORES ###
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
        label_imbalance = max(np.mean(risk_score_df["label"]), 1 - np.mean(risk_score_df["label"]))
        auc = metrics_json["roc_auc"]

        # add average risk, metric to datapoints
        risk_score_results[f.split("_")[0]] = [avg_risk, auc, std_risk, label_imbalance]

    # get all confidence score info

    for f in os.listdir(confidence_scores_dir):
        assert len(os.listdir(confidence_scores_dir / f)) == 1
        timestamp = os.listdir(confidence_scores_dir / f)[0]
        openai_dir = Path(f"{confidence_scores_dir}/{f}/{timestamp}")
        masking_pickle = openai_dir / "all_results.pickle"
        with open(masking_pickle, "rb") as handle:
            masking_results = pickle.load(handle)

        confidence_results[f.split("_")[0]] = [
            np.mean([masking_results[x]["roc_auc"] for x in masking_results]),
            np.std([masking_results[x]["roc_auc"] for x in masking_results]),
        ]
        masking_lists[f.split("_")[0]] = [masking_results[x]["roc_auc"] for x in masking_results]

    keys = [key for key in confidence_results]
    keys.sort()

    # box plots of roc over cols, per dataset
    # Create horizontal boxplots
    fig, ax = plt.subplots(figsize=(10, 15))  # Adjust height for label clarity
    ax.boxplot([masking_lists[key] for key in keys], vert=False, patch_artist=True)

    # Set y-axis labels
    ax.set_yticks(range(1, len(masking_lists) + 1))
    ax.set_yticklabels(list(keys))

    ax.set_xlabel("AUC")
    ax.set_title("Variation over Columns in Masking Task, Llama")
    plt.tight_layout()
    plt.savefig("masking_dist.png")
    plt.clf()

    assert len(confidence_results) == len(risk_score_results)

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
    plt.xlabel("average auc over features")
    plt.ylabel("auc score")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("mean_masking.png")
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
    plt.xlabel("std auc over columns")
    plt.ylabel("auc score")
    plt.tight_layout()
    plt.savefig("std_masking.png")
    plt.clf()
