import json
import os
import pdb
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


risk_scores_dir = Path("mistral_classification")
confidence_scores_dir = Path("results")
masking_dir = Path("mistral_masking")
risk_score_results = {}
confidence_results = {}
mask_results = {}
auc_results = {}
score_results = {}
probs_results = {}

if __name__ == "__main__":
    ### RISK SCORES ###
    for f in os.listdir(risk_scores_dir):
        if "csv" in f:
            continue
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
            if results_file == "imgs":
                shutil.copy(
                    results_dir / results_file / "calibration_curve.pdf",
                    f"calibration_plots_llama/{f.split('_')[0]}.pdf",
                )

        avg_risk = np.mean(np.abs(risk_score_df["risk_score"] - np.round(risk_score_df["risk_score"])))
        std_risk = np.std(risk_score_df["risk_score"])
        label_imbalance = max(np.mean(risk_score_df["label"]), 1 - np.mean(risk_score_df["label"]))
        auc = metrics_json["roc_auc"]
        ece = metrics_json["ece"]

        # add average risk, metric to datapoints
        risk_score_results[f.split("_")[0]] = [avg_risk, auc, std_risk, label_imbalance, ece]

    # Convert to DataFrame
    risk_scores_df = pd.DataFrame.from_dict(risk_score_results, orient="index")
    risk_scores_df.columns = ["average_risk_score", "auc", "std_risk_scores", "label_imbalance", "ece"]
    risk_scores_df = risk_scores_df.reset_index().rename(columns={"index": "dataset_name"})

    ### CONFIDENCE ###
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
    confidence_scores_df = pd.DataFrame.from_dict(confidence_results, orient="index")
    confidence_scores_df.columns = ["confidence_score", "std_confidence_score"]
    confidence_scores_df = confidence_scores_df.reset_index().rename(columns={"index": "dataset_name"})

    ### MASKING ###

    for f in os.listdir(masking_dir):
        if "csv" in f:
            continue
        assert len(os.listdir(masking_dir / f)) == 1
        timestamp = os.listdir(masking_dir / f)[0]
        openai_dir = Path(f"{masking_dir}/{f}/{timestamp}")
        masking_pickle = openai_dir / "all_results.pickle"
        with open(masking_pickle, "rb") as handle:
            masking_results = pickle.load(handle)

        mask_results[f.split("_")[0]] = [
            np.mean([masking_results[x]["roc_auc"] for x in masking_results]),
            np.std([masking_results[x]["roc_auc"] for x in masking_results]),
        ]
    assert len(mask_results) == len(risk_score_results)
    masking_df = pd.DataFrame.from_dict(mask_results, orient="index")
    masking_df.columns = ["masking", "std_masking"]
    masking_df = masking_df.reset_index().rename(columns={"index": "dataset_name"})

    full_df = pd.merge(risk_scores_df, confidence_scores_df, on="dataset_name", how="inner")
    full_df = pd.merge(full_df, masking_df, on="dataset_name", how="inner")

    for no_data_experiment in ["auc", "probs", "score"]:
        csv = pd.read_csv(f"results/{no_data_experiment}_logprob.csv")
        csv = csv.rename(columns={"rating": no_data_experiment})
        full_df = pd.merge(full_df, csv, on="dataset_name", how="inner")

    full_df = full_df.loc[:, ~full_df.columns.str.contains("^Unnamed")]
    full_df.to_csv("all_results.csv")
