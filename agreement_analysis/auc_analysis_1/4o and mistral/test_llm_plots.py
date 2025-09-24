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

mistral_new = pd.read_csv("mistral.csv")
gpt_new = pd.read_csv("gpt.csv")

plt.rc("axes", titlesize=26)
plt.rc("axes", labelsize=26)
plt.rc("font", size=20)


# mistral_new = pd.read_csv('mistral_new.csv')
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

# # For mistral
# x_grid_mistral = np.linspace(data_mistral[metric].min(), data_mistral[metric].max(), 100)
# smoothed_mistral = lowess(data_mistral['auc'], data_mistral[metric], frac=0.4, return_sorted=True)
# f_interp_mistral = interp1d(smoothed_mistral[:, 0], smoothed_mistral[:, 1], bounds_error=False, fill_value="extrapolate")
# mistral_fitted = f_interp_mistral(x_grid_mistral)
# ci_lower_mistral, ci_upper_mistral = bootstrap_loess_confidence_interval(data_mistral[metric].values, data_mistral['auc'].values,
#                                                                      frac=0.4, grid=x_grid_mistral, n_boot=1000, alpha=0.1)
# plt.fill_between(x_grid_mistral, ci_lower_mistral, ci_upper_mistral, color='C1', alpha=0.2, label='mistral 95% CI')

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

#     # Process mistral data
#     x_mistral = data_mistral[metric].values
#     y_mistral = data_mistral['auc'].values
#     grid_mistral = np.linspace(x_mistral.min(), x_mistral.max(), 100)
#     decay_mistral = (x_mistral.max() - x_mistral.min()) / 10
#     fitted_mistral = exponential_weighted_smoother(x_mistral, y_mistral, grid_mistral, decay_mistral)
#     plt.scatter(x_mistral, y_mistral, c='C1', s=6, alpha=0.5)
#     plt.plot(grid_mistral, fitted_mistral, color='C1', label='mistral')

#     plt.legend()
#     plt.title(metric)
#     plt.show()
# for metric in data_gpt.columns:
#     if metric in ['auc','Unnamed: 0', 'dataset_name' ]:
#         continue
#     plt.scatter(data_gpt[metric], data_gpt.auc, c='C0', s = 6, alpha=0.5)
#     smoothed = lowess(data_gpt['auc'], data_gpt[metric], frac=0.5)
#     plt.plot(smoothed[:, 0], smoothed[:, 1], color='C0', label='GPT')

#     plt.scatter(data_mistral[metric], data_mistral.auc, c='C1', s = 6, alpha=0.5)
#     smoothed = lowess(data_mistral['auc'], data_mistral[metric], frac=0.5)
#     plt.plot(smoothed[:, 0], smoothed[:, 1], color='C1', label='mistral')
#     plt.legend()
#     plt.title(metric)
#     plt.show()

metric = "std_risk"
x1 = gpt_new[metric].values
y1 = gpt_new["auc"].values
x2 = mistral_new[metric].values
y2 = mistral_new["auc"].values

for x, y, llm in [(x1, y1, "GPT"), (x2, y2, "mistral")]:
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

    # pdf("../results_section_2/{llm}_std_risk_score.pdf", width = 7, height = 5)
    pdf("{llm}_std_risk_score.pdf", width = 7, height = 5)

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
    plt.savefig(f"{llm}_std_risk_score_sliding_window.pdf", format="pdf", bbox_inches="tight")
    # plt.savefig(f"../results_section_2/{llm}_std_risk_score_sliding_window.pdf", format="pdf", bbox_inches="tight")
    plt.clf()
