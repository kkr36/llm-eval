import json
import os
import pdb
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm


plt.rc("axes", titlesize=20)
plt.rc("axes", labelsize=20)
plt.rc("font", size=15)

confidence_scores_dir = Path("xgb_pickles")
risk_score_results = {}
masking_lists = {}

if __name__ == "__main__":
    xgb_df = pd.read_csv("xgb.csv")
    zero_shot_xgb_results = {
        dataset: xgb for dataset, xgb in zip(xgb_df["dataset_name"].tolist(), xgb_df["xgb_auc"].tolist())
    }
    keys = list(zero_shot_xgb_results.keys())
    keys.sort()

    for llm in tqdm(["GPT", "Llama", "GPT_4o", "Mistral"][-2:]):
        # get all risk score info
        metrics_df = pd.read_csv(f"{llm.lower()}.csv")
        datasets, aucs = metrics_df["dataset_name"].tolist(), metrics_df["auc"].tolist()
        risk_score_results = {dataset: auc for (dataset, auc) in zip(datasets, aucs)}

        # get all confidence score info
        if llm == "Llama" or llm == "Mistral":
            for f in os.listdir(confidence_scores_dir):
                masking_pickle = confidence_scores_dir / f
                with open(masking_pickle, "rb") as handle:
                    xgb_results = pickle.load(handle)
                tasks = list(xgb_results.keys())
                tasks.sort()
                xgb_raw = [xgb_results[x]["xgb_auc"] for x in tasks]

                masking_pickle_llama = Path(f"xgb_pickles_{llm.lower()}") / f
                with open(masking_pickle_llama, "rb") as handle:
                    masking_results = pickle.load(handle)
                # pdb.set_trace()
                auc_raw = [masking_results[x]["roc_auc"] for x in tasks]
                auc_norm = [masking_results[x]["roc_auc"] / xgb_results[x]["xgb_auc"] for x in tasks]
                masking_lists[f.split(".")[0]] = [auc_raw, xgb_raw, auc_norm]
                # pdb.set_trace()
        elif llm == "GPT":
            for f in os.listdir(confidence_scores_dir):
                masking_pickle = confidence_scores_dir / f
                with open(masking_pickle, "rb") as handle:
                    masking_results = pickle.load(handle)

                auc_raw = [masking_results[x]["roc_auc"] for x in masking_results]
                xgb_raw = [masking_results[x]["xgb_auc"] for x in masking_results]
                auc_norm = [masking_results[x]["roc_auc"] / masking_results[x]["xgb_auc"] for x in masking_results]

                masking_lists[f.split(".")[0]] = [auc_raw, xgb_raw, auc_norm]
        else:
            for f in os.listdir("xgb_pickles_4o"):
                masking_pickle = confidence_scores_dir / f
                with open(masking_pickle, "rb") as handle:
                    masking_results = pickle.load(handle)

                auc_raw = [masking_results[x]["roc_auc"] for x in masking_results]
                xgb_raw = [masking_results[x]["xgb_auc"] for x in masking_results]
                auc_norm = [masking_results[x]["roc_auc"] / masking_results[x]["xgb_auc"] for x in masking_results]

                masking_lists[f.split(".")[0]] = [auc_raw, xgb_raw, auc_norm]

        def scatter_with_line(X, y, xlab, ylab, path):
            model = LinearRegression()
            X, y = np.array(X).reshape(-1, 1), np.array(y)
            model.fit(X, y)
            # Predict
            y_pred = model.predict(X)

            # Calculate R²
            r_squared = r2_score(y, y_pred)

            print(f"R-squared: {r_squared}")

            # plot datapoints
            plt.scatter(X, y, c="blue", label="Data points")
            plt.plot(X, y_pred, color="red", label=f"Best fit line\n$R^2$ = {r_squared:.3f}")
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.savefig(path)
            plt.clf()

        for i, list_name in enumerate(["AUC", "Raw XGB AUC", "Normalized AUC"]):
            # calculate icc for normalized auc
            # if list_name == "Normalized AUC":
            #     icc_results = pg.intraclass_corr(data=[masking_lists[key][i] for key in keys], targets='Subject', raters='Rater', ratings='Rating')

            # remake box plots
            fig, ax = plt.subplots(figsize=(10, 15))  # Adjust height for label clarity
            ax.boxplot([masking_lists[key][i] for key in keys], vert=False, patch_artist=True)
            ax.set_yticks(range(1, len(masking_lists) + 1))
            ax.set_yticklabels(list(keys))
            ax.set_xlabel(list_name)
            # ax.set_title(f"Variation over Columns in Masking Task, {llm}")
            plt.tight_layout()
            plt.savefig(f"redone_masking/{llm}/{list_name}.pdf", format="pdf", bbox_inches="tight")
            plt.clf()
            plt.figure()

            downstream_aucs = [risk_score_results[key] for key in keys]
            normalized_aucs = [risk_score_results[key] / zero_shot_xgb_results[key] for key in keys]
            mean_val = [np.mean(masking_lists[key][i]) for key in keys]
            std_val = [np.std(masking_lists[key][i]) for key in keys]

            # remake scatter plots, dividing only the x axis (masking auc) by xgb vals
            scatter_with_line(
                mean_val,
                downstream_aucs,
                f"Mean value of {list_name} over masking cols",
                "Downstream AUC",
                f"redone_masking/{llm}/{list_name}_mean_raw.png",
            )
            scatter_with_line(
                std_val,
                downstream_aucs,
                f"STD Value of {list_name} over masking cols",
                "Downstream AUC",
                f"redone_masking/{llm}/{list_name}_std_raw.png",
            )

            # remake scatter plots, dividing both x and y axes by xgb vals
            scatter_with_line(
                mean_val,
                normalized_aucs,
                f"Mean Value of {list_name} over masking cols",
                "Normalized Downstream AUC",
                f"redone_masking/{llm}/{list_name}_mean_norm.png",
            )
            scatter_with_line(
                std_val,
                normalized_aucs,
                f"STD Value of {list_name} over masking cols",
                "Normalized Downstream AUC",
                f"redone_masking/{llm}/{list_name}_mean_norm.png",
            )
