import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


plt.rc("axes", titlesize=20)
plt.rc("axes", labelsize=20)
plt.rc("font", size=15)

if __name__ == "__main__":
    llama = pd.read_csv("llama.csv")
    gpt = pd.read_csv("gpt.csv")
    gpt_4o = pd.read_csv("gpt_4o.csv")
    mistral = pd.read_csv("mistral.csv")

    # assert(set(llama['dataset_name'].tolist()) == set(gpt['dataset_name'].tolist()))

    dfs = [(llama, "Llama"), (gpt, "GPT"), (gpt_4o, "GPT 4o"), (mistral, "Mistral")]
    for i, (df1, name1) in enumerate(dfs):
        for df2, name2 in dfs[i + 1 :]:
            folder = f"agreement_plots_category/{name1}_{name2}"
            if not os.path.exists(folder):
                os.makedirs(folder)

            merged_df = pd.merge(df1, df2, on="dataset_name", how="inner", suffixes=(f"_{name1}", f"_{name2}"))
            cat_col = pd.read_csv("dataset_labels.csv")
            merged_df = pd.merge(merged_df, cat_col, on="dataset_name", how="inner")

            # Map categories to colors
            categories = cat_col["label"].unique()
            color_map = {category: color for category, color in zip(categories, plt.cm.Set1.colors)}

            for col in df1.columns:
                if col == "dataset_name":
                    continue
                if col == "label_imbalance":
                    continue
                if col == "label":
                    continue
                if "Unnamed" in col:
                    continue
                llama_col, gpt_col = merged_df[f"{col}_{name1}"].tolist(), merged_df[f"{col}_{name2}"].tolist()

                model = LinearRegression()
                X, y = np.array(llama_col).reshape(-1, 1), np.array(gpt_col)
                model.fit(X, y)
                # Predict
                y_pred = model.predict(X)

                # Calculate R²
                r_squared = r2_score(y, y_pred)

                plt.scatter(llama_col, gpt_col, label="Data points")
                # Plot
                # for category in categories:
                #     subset = merged_df[merged_df['label'] == category]
                #     plt.scatter(subset[f"{col}_llama"], subset[f"{col}_gpt"], label=category, color=color_map[category])
                # plt.legend(title='Category')
                # plt.colorbar(label='Imbalance', orientation='horizontal')
                plt.plot(X, y_pred, color="red", label=f"Best fit line\n$R^2$ = {r_squared:.3f}")
                plt.plot(
                    [i / 10 for i in range(4, 11)],
                    [i / 10 for i in range(4, 11)],
                    linestyle=":",
                    color="black",
                    label="y = x",
                )
                plt.text(0.9, 0, rf"$R^2=${r_squared:.3f}", ha="right", va="bottom", transform=plt.gca().transAxes)

                # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                # plt.xlim((.4,1))
                # plt.ylim((.4,1))
                plt.xlabel(f"{name1} {col.upper()}")
                plt.ylabel(f"{name2} {col.upper()}")
                plt.tight_layout()
                plt.savefig(f"{folder}/{col}.pdf", format="pdf", bbox_inches="tight")
                plt.clf()

                if col == "auc":
                    xgb_df = pd.read_csv("xgb.csv")
                    joined_xgb_df = pd.merge(merged_df, xgb_df, on="dataset_name", how="inner")
                    assert len(joined_xgb_df) == len(merged_df)

                    # divide both xgb, llama vals by xgb, correlate, plot
                    joined_xgb_df[f"{name1}_auc_normalized"] = joined_xgb_df[f"auc_{name1}"] / joined_xgb_df["xgb_auc"]
                    joined_xgb_df[f"{name2}_auc_normalized"] = joined_xgb_df[f"auc_{name2}"] / joined_xgb_df["xgb_auc"]

                    model = LinearRegression()
                    X, y = (
                        np.array(joined_xgb_df[f"{name1}_auc_normalized"].tolist()).reshape(-1, 1),
                        np.array(joined_xgb_df[f"{name2}_auc_normalized"].tolist()),
                    )
                    model.fit(X, y)
                    # Predict
                    y_pred = model.predict(X)

                    # Calculate R²
                    r_squared = r2_score(y, y_pred)

                    plt.scatter(
                        joined_xgb_df[f"{name1}_auc_normalized"],
                        joined_xgb_df[f"{name2}_auc_normalized"],
                        label="Data points",
                    )
                    # Plot
                    # for category in categories:
                    #     subset = joined_xgb_df[joined_xgb_df['label'] == category]
                    #     plt.scatter(subset[f"llama_auc_normalized"], subset[f"gpt_auc_normalized"], label=category, color=color_map[category])
                    plt.legend(title="Category")
                    plt.plot(X, y_pred, color="red", label=rf"Best fit line\n$R^2$ = {r_squared:.3f}")
                    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                    plt.xlabel(f"{name1} {col.upper()}")
                    plt.ylabel(f"{name2} {col.upper()}")
                    plt.tight_layout()
                    plt.savefig(f"{folder}/auc_normalized.pdf", format="pdf", bbox_inches="tight")
                    plt.clf()
