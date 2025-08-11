import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d

plt.rc("axes",titlesize=20)
plt.rc("axes",labelsize=20)
plt.rc("font",size=14)

exp_name_map = {
    "average_risk_score": "Avg. Distance From Predicted Label",
    "std_risk_scores": "Std. Predicted Probabilities",
    "confidence_score": "Avg. Verbalized Confidence Score",
    "std_confidence_score": "Std. Verbalized Confidence",
    "masking": "Avg. AUC over Masked Features",
    "std_masking": "Std. AUC over Masked Features",
    "predicted_auc": "LLM-Forecasted AUC",
    "probs": "Confidence in Prediction Task (0-1)",
    "score": "Confidence in Prediction Task (1-5)",
    "ece": "ECE"
}

def bootstrap_loess_confidence_interval(x, y, frac, grid, n_boot=100, alpha=0.05):
    boot_preds = np.zeros((n_boot, len(grid)))
    for i in range(n_boot):
        indices = np.random.choice(np.arange(len(x)), size=len(x), replace=True)
        x_boot = x[indices]
        y_boot = y[indices]
        smoothed = lowess(y_boot, x_boot, frac=frac, return_sorted=True)
        f_interp = interp1d(smoothed[:, 0], smoothed[:, 1], bounds_error=False, fill_value="extrapolate")
        boot_preds[i, :] = f_interp(grid)
    lower_bound = np.percentile(boot_preds, 100 * alpha / 2, axis=0)
    upper_bound = np.percentile(boot_preds, 100 * (1 - alpha / 2), axis=0)
    return lower_bound, upper_bound

if __name__ == "__main__":
    label_csv = pd.read_csv("dataset_labels.csv")
    categories = label_csv['label'].unique()
    color_map = {category: color for category, color in zip(categories, plt.cm.Set1.colors)}
    experiment_cols = ["average_risk_score", "std_risk_scores", "confidence_score", "std_confidence_score", "masking", "std_masking", "predicted_auc", "probs", "score", "ece"][:-1]
    gpt_stats = []
    llama_stats = []

    for llm_path in ["gpt", "llama"]:
        llm = llm_path.upper()

        llm_csv = pd.read_csv(f"{llm_path}.csv")
        merged_df = pd.merge(llm_csv, label_csv, on='dataset_name', how='inner')

        for col in experiment_cols:
            # if (col=="masking" and llm_path=="llama") or (col=="std_masking" and llm_path=="llama"): continue
            x = merged_df[col].values
            y = merged_df['auc'].values

            grid = np.linspace(x.min(), x.max(), 100)
            # Decay chosen as 1/10th of the predictor range; adjust as needed.
            decay = (x.max() - x.min()) / 10
            # smoothed_gpt = exponential_weighted_smoother(x, y, grid, decay)
            smoothed_gpt = lowess(y, x, frac=0.4, return_sorted=True)
            plt.scatter(x, y, c='C0', s=6, alpha=0.5)
            plt.plot(smoothed_gpt[:, 0], smoothed_gpt[:, 1], color='C0', label=f'{llm}')

            x_grid = np.linspace(x.min(), x.max(), 100)
            # smoothed_gpt = lowess(y, x, frac=0.4, return_sorted=True)
            # fitted_gpt = exponential_weighted_smoother(x, y, grid, decay)

            f_interp = interp1d(smoothed_gpt[:, 0], smoothed_gpt[:, 1], bounds_error=False, fill_value="extrapolate")
            gpt_fitted = f_interp(x_grid)
            ci_lower, ci_upper = bootstrap_loess_confidence_interval(x, y,
                                                                    frac=0.4, grid=x_grid, n_boot=1000, alpha=0.1)
            plt.fill_between(x_grid, ci_lower, ci_upper, color='C0', alpha=0.2, label=f'{llm} 95% CI')

            import scipy.stats
            scipy.stats.spearmanr(x, y)
            plt.xlabel(f'{exp_name_map[col]}')
            plt.ylabel('Downstream AUC')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5,-.24))
            plt.savefig(f"results_section_4/{llm}_{col}.pdf", format="pdf", bbox_inches="tight")
            plt.clf()


            thresholds = np.sort(np.unique(x))

            # compute the mean of data.auc for all points with std_risk_scores >= each threshold
            mean_auc = [y[x >= th].mean() for th in thresholds]

            # plot the computed means as a line
            plt.plot(thresholds, mean_auc, color='red', label='Mean AUC for thresholds')
            plt.xlabel(f'{exp_name_map[col]}')
            plt.ylabel('Downstream AUC')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5,-.24))
            plt.savefig(f"results_section_4/{llm}_{col}_sliding_window.pdf", format="pdf", bbox_inches="tight")
            plt.clf()
        
    # data = [
    #     ['llama'] + llama_stats,
    #     ['gpt'] + gpt_stats
    # ]

    # df = pd.DataFrame(data, columns=['llm'] + experiment_cols)

    # # Save to CSV
    # df.to_csv('r2_table.csv', index=False)
        