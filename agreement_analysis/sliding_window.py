import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


plt.rc("axes", titlesize=20)
plt.rc("axes", labelsize=20)
plt.rc("font", size=14)

cols_of_interest = [
    "average_risk_score",
    "std_risk_scores",
    "confidence_score",
    "std_confidence_score",
    "masking",
    "std_masking",
    "predicted_auc",
    "probs",
    "score",
]

if __name__ == "__main__":
    for llm in ["gpt", "llama"]:
        metrics_df = pd.read_csv(f"{llm}.csv")
        for col in cols_of_interest:
            datapoints = []

            metrics = metrics_df[col]
            min_metric = min(metrics)
            max_metric = max(metrics)

            # hypothesis: as we increase lower bound, we should get better aucs
            for lower_bound in np.arange(min_metric, max_metric + 0.001, 0.005):
                subset = metrics_df[metrics_df[col] >= lower_bound]
                aucs = np.mean(subset["auc"])
                if lower_bound >= 0.4 and col == "std_risk_scores":
                    import pdb

                    pdb.set_trace()
                if lower_bound == min_metric and col == "std_risk_scores":
                    import pdb

                    pdb.set_trace()

                datapoints.append([lower_bound, aucs, len(subset)])

            x, y1, y2 = [x[0] for x in datapoints], [x[1] for x in datapoints], [x[2] for x in datapoints]

            fig, ax1 = plt.subplots()

            # Plot y1 on the primary y-axis
            color = "tab:blue"
            ax1.set_xlabel("lower bound (larger = only look at extremes)")
            ax1.set_ylabel("Average AUC", color=color)
            ax1.plot(x, y1, color=color)
            ax1.tick_params(axis="y", labelcolor=color)

            # Create a secondary y-axis
            ax2 = ax1.twinx()

            # Plot y2 on the secondary y-axis
            color = "tab:red"
            ax2.set_ylabel("number of datapoints", color=color)
            ax2.plot(x, y2, color=color)
            ax2.tick_params(axis="y", labelcolor=color)
            plt.savefig(f"sliding_window/{llm}/{col}.pdf", format="pdf", bbox_inches="tight")

            plt.clf()

            # plt.plot([x[0] for x in datapoints], [x[1] for x in datapoints])
            # plt.xlabel("lower bound (larger = only look at extremes)")
            # plt.ylabel("average auc over datasets")
            # plt.savefig(f"sliding_window/{llm}/{col}.png")
            # plt.clf()
