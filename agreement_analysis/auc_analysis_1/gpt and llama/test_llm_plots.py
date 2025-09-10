import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess


# Activate conversion
pandas2ri.activate()

llama_new = pd.read_csv("llama.csv")
gpt_new = pd.read_csv("gpt.csv")

plt.rc("axes", titlesize=26)
plt.rc("axes", labelsize=26)
plt.rc("font", size=20)


# llama_new = pd.read_csv('llama_new.csv')
# gpt_new = pd.read_csv('gpt_new.csv')

# metric = 'std_risk_scores'

# plt.scatter(data.std_risk_scores, data.auc)

# # create a sorted array of thresholds from the x-axis values
# thresholds = np.sort(data['std_risk_scores'].unique())

# # compute the mean of data.auc for all points with std_risk_scores >= each threshold
# mean_auc = [data.loc[data['std_risk_scores'] >= th, 'auc'].mean() for th in thresholds]


# # plot the computed means as a line
# plt.plot(thresholds, mean_auc, color='red', label='Mean AUC for thresholds')
# plt.xlabel('Standard Risk Scores Threshold')
# plt.ylabel('Mean AUC')
# plt.legend()
# plt.show()
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


# # For GPT
# x_grid = np.linspace(data_gpt[metric].min(), data_gpt[metric].max(), 100)
# smoothed_gpt = lowess(data_gpt['auc'], data_gpt[metric], frac=0.4, return_sorted=True)
# f_interp = interp1d(smoothed_gpt[:, 0], smoothed_gpt[:, 1], bounds_error=False, fill_value="extrapolate")
# gpt_fitted = f_interp(x_grid)
# ci_lower, ci_upper = bootstrap_loess_confidence_interval(data_gpt[metric].values, data_gpt['auc'].values,
#                                                          frac=0.4, grid=x_grid, n_boot=1000, alpha=0.1)
# plt.fill_between(x_grid, ci_lower, ci_upper, color='C0', alpha=0.2, label='GPT 95% CI')

# # For Llama
# x_grid_llama = np.linspace(data_llama[metric].min(), data_llama[metric].max(), 100)
# smoothed_llama = lowess(data_llama['auc'], data_llama[metric], frac=0.4, return_sorted=True)
# f_interp_llama = interp1d(smoothed_llama[:, 0], smoothed_llama[:, 1], bounds_error=False, fill_value="extrapolate")
# llama_fitted = f_interp_llama(x_grid_llama)
# ci_lower_llama, ci_upper_llama = bootstrap_loess_confidence_interval(data_llama[metric].values, data_llama['auc'].values,
#                                                                      frac=0.4, grid=x_grid_llama, n_boot=1000, alpha=0.1)
# plt.fill_between(x_grid_llama, ci_lower_llama, ci_upper_llama, color='C1', alpha=0.2, label='Llama 95% CI')

# def exponential_weighted_smoother(x, y, grid, decay):
#     smoothed = np.empty_like(grid)
#     for i, g in enumerate(grid):
#         weights = np.exp(-np.abs(x - g) / decay)
#         smoothed[i] = np.sum(weights * y) / np.sum(weights)
#     return smoothed

# for metric in data_gpt.columns:
#     if metric in ['auc', 'Unnamed: 0', 'dataset_name']:
#         continue

#     # Process GPT data
#     x = data_gpt[metric].values
#     y = data_gpt['auc'].values
#     grid = np.linspace(x.min(), x.max(), 100)
#     # Decay chosen as 1/10th of the predictor range; adjust as needed.
#     decay = (x.max() - x.min()) / 10
#     fitted_gpt = exponential_weighted_smoother(x, y, grid, decay)
#     plt.scatter(x, y, c='C0', s=6, alpha=0.5)
#     plt.plot(grid, fitted_gpt, color='C0', label='GPT')

#     # Process Llama data
#     x_llama = data_llama[metric].values
#     y_llama = data_llama['auc'].values
#     grid_llama = np.linspace(x_llama.min(), x_llama.max(), 100)
#     decay_llama = (x_llama.max() - x_llama.min()) / 10
#     fitted_llama = exponential_weighted_smoother(x_llama, y_llama, grid_llama, decay_llama)
#     plt.scatter(x_llama, y_llama, c='C1', s=6, alpha=0.5)
#     plt.plot(grid_llama, fitted_llama, color='C1', label='Llama')

#     plt.legend()
#     plt.title(metric)
#     plt.show()
# for metric in data_gpt.columns:
#     if metric in ['auc','Unnamed: 0', 'dataset_name' ]:
#         continue
#     plt.scatter(data_gpt[metric], data_gpt.auc, c='C0', s = 6, alpha=0.5)
#     smoothed = lowess(data_gpt['auc'], data_gpt[metric], frac=0.5)
#     plt.plot(smoothed[:, 0], smoothed[:, 1], color='C0', label='GPT')

#     plt.scatter(data_llama[metric], data_llama.auc, c='C1', s = 6, alpha=0.5)
#     smoothed = lowess(data_llama['auc'], data_llama[metric], frac=0.5)
#     plt.plot(smoothed[:, 0], smoothed[:, 1], color='C1', label='Llama')
#     plt.legend()
#     plt.title(metric)
#     plt.show()

metric = "std_risk"
x1 = gpt_new[metric].values
y1 = gpt_new["auc"].values
x2 = llama_new[metric].values
y2 = llama_new["auc"].values

for x, y, llm in [(x1, y1, "GPT"), (x2, y2, "Llama")]:
    # Create pandas DataFrame
    df = pd.DataFrame({"x": x, "y": y})

    # Push DataFrame to R
    ro.globalenv["df"] = pandas2ri.py2rpy(df)

    # R script to generate and save the plot
    r_script = f"""
    library(ggplot2)


    p <- ggplot(df, aes(x = x, y = y)) +
    geom_point(color = "blue", alpha = 0.6) +
    geom_smooth(method = "loess", se = TRUE, color = "red") +
    theme_minimal() +
    theme(
        axis.text = element_text(size = 20),        # Tick label size
        axis.title = element_text(size = 28)       # Axis title size
    ) +
    ylim(.1,1.05) +
    labs(x = "Std. Risk Scores", y = "AUC")

    pdf("../results_section_2/{llm}_std_risk_score.pdf", width = 7, height = 5)
    print(p)
    dev.off()

    """

    # Run the R code
    ro.r(r_script)

    # grid = np.linspace(x.min(), x.max(), 100)
    # # Decay chosen as 1/10th of the predictor range; adjust as needed.
    # decay = (x.max() - x.min()) / 10
    # # smoothed_gpt = exponential_weighted_smoother(x, y, grid, decay)
    # smoothed_gpt = lowess(y, x, frac=0.4, return_sorted=True)
    # plt.scatter(x, y, alpha=0.7)
    # plt.plot(smoothed_gpt[:, 0], smoothed_gpt[:, 1], color='black', label=f'Loess Smoothed')
    # plt.ylim((.1,1.05))

    # x_grid = np.linspace(x.min(), x.max(), 100)
    # # smoothed_gpt = lowess(y, x, frac=0.4, return_sorted=True)
    # # fitted_gpt = exponential_weighted_smoother(x, y, grid, decay)

    # f_interp = interp1d(smoothed_gpt[:, 0], smoothed_gpt[:, 1], bounds_error=False, fill_value="extrapolate")
    # gpt_fitted = f_interp(x_grid)
    # ci_lower, ci_upper = bootstrap_loess_confidence_interval(x, y,
    #                                                         frac=0.4, grid=x_grid, n_boot=1000, alpha=0.1)
    # plt.fill_between(x_grid, ci_lower, ci_upper, color='grey', alpha=0.2, label=f'Loess 95% CI')

    # import scipy.stats
    # scipy.stats.spearmanr(x, y)
    # plt.xlabel(f'Std. Risk Scores')
    # plt.ylabel('Downstream AUC')
    # # plt.legend(loc='upper center', bbox_to_anchor=(0.5,-.24))
    # # plt.legend()
    # plt.savefig(f"../results_section_2/{llm}_std_risk_score.pdf", format="pdf", bbox_inches="tight")
    # plt.clf()

    thresholds = np.sort(np.unique(x))

    # compute the mean of data.auc for all points with std_risk_scores >= each threshold
    mean_auc = [y[x >= th].mean() for th in thresholds]
    # import pdb; pdb.set_trace()

    # plot the computed means as a line
    plt.plot(thresholds, mean_auc, color="red", label="Mean AUC for thresholds")
    plt.xlabel("Std. Risk Scores ≥ Threshold")
    plt.ylabel("Mean AUC")
    # plt.legend()
    plt.ylim(0.5, 1)
    plt.savefig(f"../results_section_2/{llm}_std_risk_score_sliding_window.pdf", format="pdf", bbox_inches="tight")
    plt.clf()
