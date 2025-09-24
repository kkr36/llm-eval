import json
import os
import pdb
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


plt.rc("axes", titlesize=20)
plt.rc("axes", labelsize=20)
plt.rc("font", size=15)

risk_scores_dir = Path("past_results/gpt_zero_shot")
r2_vals = []
ece_vals = []
auc_vals = []
datapoints = []
positive_datapoints = []
negative_datapoints = []

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

        datapoints = []

        # if "airline" in f: continue

        for window_size in range(500, 0, -1):
            lower_bound = 0.5 - (window_size / 1000.0)
            upper_bound = 0.5 + (window_size / 1000.0)

            subset = risk_score_df[
                (risk_score_df["risk_score"] >= upper_bound) | (risk_score_df["risk_score"] <= lower_bound)
            ]


            risk_scores = subset["risk_score"].tolist()
            labels = subset["label"].tolist()

            accuracy = 1 - np.mean(np.abs(np.round(risk_scores) - np.array(labels)))

            # majority_pct_subset = max(np.mean(subset["label"]), 1-np.mean(subset["label"]))
            majority_pct_df = max(np.mean(risk_score_df["label"]), 1 - np.mean(risk_score_df["label"]))

            # if window_size >= 48 and accuracy <= .30: import pdb; pdb.set_trace()
            if len(subset) > 0:
                datapoints.append([window_size, (accuracy - majority_pct_df) / (1 - majority_pct_df), accuracy])

        if min([x[1] for x in datapoints]) < 0:
            negative_datapoints.append(datapoints)
        else:
            positive_datapoints.append(datapoints)

    plt.figure(figsize=(6, 6))
    # plt.ylim((0,1))
    for series in negative_datapoints:
        plt.plot([x[0] / 1000 + 0.5 for x in series], [x[2] for x in series], color="blue")
    plt.xlabel("Max Class Probability ≥ Threshold")
    plt.ylabel("Accuracy")
    plt.savefig("negatives.pdf", format="pdf", bbox_inches="tight")
    plt.clf()

    plt.figure(figsize=(6, 6))
    # plt.ylim((0,1))
    for series in positive_datapoints:
        plt.plot([x[0] / 1000 + 0.5 for x in series], [x[2] for x in series], color="blue")
    plt.xlabel("Max Class Probability ≥ Threshold")
    plt.ylabel("Accuracy")
    plt.savefig("positives.pdf", format="pdf", bbox_inches="tight")
    plt.clf()
    # datapoints.append([window_size, accuracy, len(subset)])

    # if "brfssdiab" in f: import pdb; pdb.set_trace()

    # plot accuracies against window size

    #     plt.plot([x[0] / 1000 for x in datapoints], [x[1] for x in datapoints])
    # plt.xlabel("window size (larger = only look at extremes)")
    # plt.ylabel("accuracy")
    #     # plt.savefig(f"auc_like_plots/{f.split('.')[0]}.png")
    # plt.savefig("_all.png")

    # plt.clf()

    # x, y1, y2 = [x[0] / 1000 for x in datapoints], [x[1] for x in datapoints], [x[2] for x in datapoints]

    # fig, ax1 = plt.subplots()

    # # Plot y1 on the primary y-axis
    # color = 'tab:blue'
    # ax1.set_xlabel("window size (larger = only look at extremes)")
    # ax1.set_ylabel('Accuracy', color=color)
    # ax1.plot(x, y1, color=color)
    # ax1.tick_params(axis='y', labelcolor=color)

    # # Create a secondary y-axis
    # ax2 = ax1.twinx()

    # # Plot y2 on the secondary y-axis
    # color = 'tab:red'
    # ax2.set_ylabel('number of datapoints', color=color)
    # ax2.plot(x, y2, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    # plt.savefig(f"auc_like_plots/{f.split('.')[0]}.png")

    # plt.clf()
