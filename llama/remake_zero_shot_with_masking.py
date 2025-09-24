import json
import os
import pdb
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score


plt.rc("axes", titlesize=26)
plt.rc("axes", labelsize=26)
plt.rc("font", size=20)

if __name__ == "__main__":
    all_results_dir = Path("all_zero_shot")
    ece_vals, auc_vals = [], []
    datapoints = []
    risk_scores_list = []
    labels_list = []
    binned_labels_list = []
    quantile_labels_list = []
    bins = np.linspace(0.0, 1.0, 101)
    zero_shot_folders = sorted(os.listdir(all_results_dir))
    for zero_shot_folder in zero_shot_folders:
        for results_file in os.listdir(all_results_dir / zero_shot_folder):
            if results_file[-5:] == ".json":  # metrics
                with open(all_results_dir / zero_shot_folder / results_file, "r") as fi:
                    metrics_json = json.load(fi)
            if results_file[-4:] == ".csv":  # risk scores
                csv_file = all_results_dir / zero_shot_folder / results_file
                risk_score_df = pd.read_csv(all_results_dir / zero_shot_folder / results_file)
        avg_risk = np.mean(np.abs(risk_score_df["risk_score"] - np.round(risk_score_df["risk_score"])))
        std_risk = np.std(risk_score_df["risk_score"])
        label_imbalance = max(np.mean(risk_score_df["label"]), 1 - np.mean(risk_score_df["label"]))
        auc = metrics_json["roc_auc"]
        ece = metrics_json["ece"]
        if "_binary" not in str(csv_file):
            risk_scores_list.append(risk_score_df["risk_score"].tolist())
            labels_list.append(risk_score_df["label"].tolist())

        ece_vals.append(ece)
        auc_vals.append(auc)
        dset = metrics_json["plots"]["calibration_curve_path"].split("/")[5]
        dset_name = dset.split("_")[0]

        # bin labels in .01 size bins; add to binned_labels_list
        counts, _ = np.histogram(risk_score_df["risk_score"].tolist(), bins=bins)
        binned_labels_list.append([dset, std_risk] + (counts / len(risk_score_df)).tolist())

        # Compute quantile edges for 5 bins (i.e., 0%, 20%, 40%, ..., 100%)
        quantile_edges = np.quantile(risk_score_df["risk_score"].tolist(), q=np.linspace(0, 1, 201))
        quantile_labels_list.append([dset_name, std_risk] + quantile_edges.tolist())

        # get the xgb score for that column
        # if original task use xgb csv; else use the masking pickle
        if "classification" in dset:
            xgb_csv = pd.read_csv("xgb.csv")
            xgb_dict = {
                dataset: xgb for (dataset, xgb) in zip(xgb_csv["dataset_name"].tolist(), xgb_csv["auc"].tolist())
            }
            xgb_auc = xgb_dict[dset_name]
        else:
            assert "masking" in dset
            with open(f"xgb_pickles/{dset_name}.pickle", "rb") as handle:
                masking_pickle = pickle.load(handle)
            col_name = metrics_json["plots"]["calibration_curve_path"].split("/")[7]
            xgb_auc = masking_pickle[col_name]["xgb_auc"]

        datapoints.append([dset_name, avg_risk, auc, std_risk, label_imbalance, ece, xgb_auc])
    output_dir = Path("remake_zero_shot_with_masking")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # box plot r2, ece, auc
    for l, name in [(ece_vals, "ECE"), (auc_vals, "AUC")]:
        # Create a box plot for the single list
        plt.figure(figsize=(6, 4))
        # plt.boxplot(l, vert=True)
        plt.hist(l, bins=10, color="skyblue", edgecolor="black")
        plt.title(f"Histogram of {name}, GPT-4o-mini")
        plt.ylabel(f"{name}")
        # plt.xticks([1], [name])  # Label for the single box
        # plt.grid(True)
        plt.savefig(f"remake_zero_shot_with_masking/histogram_{name}.pdf", format="pdf", bbox_inches="tight")
        plt.clf()

    # from sklearn.linear_model import LinearRegression
    # from sklearn.metrics import r2_score
    # model = LinearRegression()
    X, y = np.array([x[2] for x in datapoints]).reshape(-1, 1), np.array([x[1] for x in datapoints])
    # model.fit(X, y)
    # # Predict
    # y_pred = model.predict(X)

    # # Calculate R²
    # r_squared = r2_score(y, y_pred)

    # print(f"R-squared: {r_squared}")

    # plot datapoints
    plt.scatter([(x[0]) for x in datapoints], [x[1] for x in datapoints])
    # plt.colorbar(label='Imbalance')
    plt.xlabel("average(risk score - round(risk score))")
    plt.ylabel("auc score")
    plt.savefig("remake_zero_shot_with_masking/risk_score_correlation.pdf", format="pdf", bbox_inches="tight")
    plt.clf()

    # Create plot with LOESS smoothing and confidence interval
    plt.figure(figsize=(8, 6))
    sns.regplot(x=X, y=y, lowess=True, scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})

    plt.title("Scatter Plot with LOESS Curve and Confidence Interval")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig("remake_zero_shot_with_masking/risk_std_correlation.pdf", format="pdf", bbox_inches="tight")
    plt.clf()

    # plt.scatter(X, y, label='Data points')
    # # plt.colorbar(label='Imbalance', orientation='horizontal')
    # plt.plot(X, y_pred, color='red', label=f'Best fit line\n$R^2$ = {r_squared:.3f}')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.xlabel("sd(risk score)")
    # plt.ylabel("auc score")
    # plt.tight_layout()
    # plt.savefig("remake_zero_shot_with_masking/risk_std_correlation.pdf", format="pdf", bbox_inches="tight")
    # plt.clf()

    plt.scatter([x[3] for x in datapoints], [x[1] for x in datapoints])
    plt.xlabel("frequency of majority class")
    plt.ylabel("auc score")
    plt.savefig("remake_zero_shot_with_masking/imbalance_correlation.pdf", format="pdf", bbox_inches="tight")
    plt.clf()

    # failure prediction -- for each of the (probs, labels) pairs, get the mcp, and see how well this predicts accuracy
    aucs = []
    accs = []
    base_aucs = []

    for i, (probs, labels) in enumerate(zip(risk_scores_list, labels_list)):
        # pdb.set_trace()
        mcp = np.maximum(np.array(probs), 1 - np.array(probs))
        accuracy = labels == np.round(probs)
        base_acc = np.mean(accuracy)
        base_auc = roc_auc_score(labels, probs)
        auc = roc_auc_score(accuracy, mcp)
        aucs.append(auc)
        accs.append(base_acc)
        base_aucs.append(base_auc)
    print(f"avg fail pred auc {np.median(aucs)}")
    plt.subplots(figsize=(6, 6))
    plt.hist(aucs, bins=30)
    plt.xlabel("AUC of Failure Prediction")
    plt.ylabel("Frequency")
    plt.savefig("remake_zero_shot_with_masking/auc_hist_llama.pdf", format="pdf", bbox_inches="tight")
    plt.clf()

    model = LinearRegression()
    X, y = np.array(base_aucs).reshape(-1, 1), np.array(aucs)
    model.fit(X, y)
    # Predict
    y_pred = model.predict(X)
    r_squared = r2_score(y, y_pred)

    plt.subplots(figsize=(6, 6))
    plt.plot(X, y_pred, color="red", label=f"Best fit line\n$R^2$ = {r_squared:.3f}")
    plt.text(0.68, 0, rf"$R^2=${r_squared:.3f}", ha="right", va="bottom", transform=plt.gca().transAxes)
    plt.scatter(base_aucs, aucs)
    plt.ylim((0.3, 1))
    plt.xlim((0.3, 1))
    plt.plot(
        [i / 10 for i in range(4, 11)], [i / 10 for i in range(4, 11)], linestyle=":", color="black", label="y = x"
    )

    plt.xlabel("Base AUC")
    plt.ylabel("AUC of Failure Prediction")
    plt.savefig("remake_zero_shot_with_masking/base_auc_auc_llama.pdf", format="pdf", bbox_inches="tight")
    plt.clf()

    plt.subplots(figsize=(6, 6))
    plt.scatter(accs, aucs)
    plt.xlabel("Base Accuracy")
    plt.ylabel("AUC of Failure Prediction")
    plt.savefig("remake_zero_shot_with_masking/base_acc_auc_llama.pdf", format="pdf", bbox_inches="tight")
    plt.clf()

    # overlay 31 calibration curves on top of one another

    # Create two separate figures
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    fig2, ax2 = plt.subplots(figsize=(6, 6))

    # Plot calibration curves for each (probs, labels) pair
    n_blue = 0
    n_red = 0
    n_non_crosses = 0
    n_crosses = 0
    for i, (probs, labels) in enumerate(zip(risk_scores_list, labels_list)):
        # Get calibration curve values
        frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=10, strategy="uniform")

        # Determine curve position relative to identity
        diffs = np.array(frac_pos) - np.array(mean_pred)
        # if np.all(diffs > -0.15) and len(diffs) > 1:
        if np.sum(diffs) >= 0.15 * len(diffs) and len(diffs) > 1 and np.min(diffs) >= -0.1:
            color = "blue"  # always above
            n_blue += 1
            n_non_crosses += 1
            alpha = 0.9
        # elif np.all(diffs < 0.15) and len(diffs) > 1:
        elif np.sum(diffs) <= -0.15 * len(diffs) and len(diffs) > 1 and np.max(diffs) <= 0.1:
            color = "red"  # always below
            n_red += 1
            n_non_crosses += 1
            alpha = 0.9
        else:
            color = "grey"  # crosses
            n_crosses += 1
            alpha = 0.5

        # if color == 'grey':
        #     ax2.plot(mean_pred, frac_pos, color=color, alpha=0.5, marker="s")
        # else:
        ax1.plot(mean_pred, frac_pos, color=color, alpha=alpha, marker="s")

    # Add text in the top left corner
    # ax1.text(0.05, 0.85, f"Low Predictions: {n_blue}",
    #            transform=ax1.transAxes,
    #            fontsize=14,
    #            verticalalignment='top',
    #            bbox=dict(facecolor='white', alpha=0, edgecolor="white"),
    #            color="blue")
    # # Add text in the bottom right corner
    # ax1.text(0.57, 0.10, f"High Predictions: {n_red}",
    #            transform=ax1.transAxes,
    #            fontsize=14,
    #            verticalalignment='top',
    #            bbox=dict(facecolor='white', alpha=0, edgecolor="white"),
    #            color="red")

    # Add legend and title
    ax1.legend().set_visible(False)
    # ax2.legend().set_visible(False)
    lims = [0, 1]
    ax1.plot(lims, lims, linestyle="--", color="black", linewidth=2.5, zorder=5)
    # ax2.plot(lims, lims, linestyle='--', color='black', linewidth=2.5, zorder=5)
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    # ax1.set_title(f"Non-crossing Tasks: {n_non_crosses}")
    # ax2.set_xlabel("Mean Predicted Probability")
    # ax2.set_ylabel("Fraction of Positives")
    # ax2.set_title(f"Crossing Tasks: {n_crosses}")

    # Show or save the figures
    plt.figure(fig1.number)
    plt.tight_layout()
    plt.savefig("all_curves_llama.pdf", format="pdf", bbox_inches="tight")

    # plt.figure(fig2.number)
    # plt.tight_layout()
    # plt.savefig("crossing_calibration_curves_gpt.pdf", format="pdf", bbox_inches="tight")

    plt.clf()

    # write metrics to csv form
    # Define column names
    columns = ["dataset", "avg_risk", "auc", "std_risk", "label_imbalance", "ece", "xgb_auc"]

    # Create DataFrame
    df = pd.DataFrame(datapoints, columns=columns)

    # Write to CSV
    df.to_csv("remake_zero_shot_with_masking/llama.csv", index=False)

    bin_labels = ["dataset", "std_risk"] + [f"{bins[i]:.3f}-{bins[i + 1]:.3f}" for i in range(len(bins) - 1)]

    # Convert list of histograms into a DataFrame
    df = pd.DataFrame(binned_labels_list, columns=bin_labels)
    df.to_csv("remake_zero_shot_with_masking/llama_binned.csv", index=False)

    quantile_labels = ["dataset", "std_risk"] + [f"decile_{i}" for i in range(201)]
    df = pd.DataFrame(quantile_labels_list, columns=quantile_labels)
    df.to_csv("remake_zero_shot_with_masking/llama_deciles.csv", index=False)
