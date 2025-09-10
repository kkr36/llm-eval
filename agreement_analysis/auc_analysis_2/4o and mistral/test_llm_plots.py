import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from scipy.interpolate import interp1d
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import GroupKFold, KFold
from statsmodels.nonparametric.smoothers_lowess import lowess
from xgboost import XGBRegressor


# Activate conversion
pandas2ri.activate()


plt.rc("axes", titlesize=26)
plt.rc("axes", labelsize=26)
plt.rc("font", size=20)

# data_mistral = pd.read_csv('mistral.csv')
# data_gpt= pd.read_csv('gpt.csv')

# mistral_new = pd.read_csv('mistral_new.csv')
# gpt_new = pd.read_csv('gpt_new.csv')

# metric = 'std_risk_scores'

# plt.scatter(data.std_risk_scores, data.auc)

# # create a sorted array of thresholds from the x-axis values
# thresholds = np.sort(data['std_risk_scores'].unique())

# # compute the mean of data.auc for all points with std_risk_scores >= each threshold
# mean_auc = [data.loc[data['std_risk_scores'] >= th, 'auc'].mean() for th in thresholds]


# plot the computed means as a line
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


# For GPT
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

# # def exponential_weighted_smoother(x, y, grid, decay):
# #     smoothed = np.empty_like(grid)
# #     for i, g in enumerate(grid):
# #         weights = np.exp(-np.abs(x - g) / decay)
# #         smoothed[i] = np.sum(weights * y) / np.sum(weights)
# #     return smoothed

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

# metric= 'std_risk'
# x = gpt_new[metric].values
# y = gpt_new['auc'].values
# x = mistral_new[metric].values
# y = mistral_new['auc'].values

# grid = np.linspace(x.min(), x.max(), 100)
# # Decay chosen as 1/10th of the predictor range; adjust as needed.
# decay = (x.max() - x.min()) / 10
# # smoothed_gpt = exponential_weighted_smoother(x, y, grid, decay)
# smoothed_gpt = lowess(y, x, frac=0.4, return_sorted=True)
# plt.scatter(x, y, c='C0', s=6, alpha=0.5)
# plt.plot(smoothed_gpt[:, 0], smoothed_gpt[:, 1], color='C0', label='GPT')

# x_grid = np.linspace(x.min(), x.max(), 100)
# # smoothed_gpt = lowess(y, x, frac=0.4, return_sorted=True)
# # fitted_gpt = exponential_weighted_smoother(x, y, grid, decay)

# f_interp = interp1d(smoothed_gpt[:, 0], smoothed_gpt[:, 1], bounds_error=False, fill_value="extrapolate")
# gpt_fitted = f_interp(x_grid)
# ci_lower, ci_upper = bootstrap_loess_confidence_interval(x, y,
#                                                          frac=0.4, grid=x_grid, n_boot=1000, alpha=0.1)
# plt.fill_between(x_grid, ci_lower, ci_upper, color='C0', alpha=0.2, label='GPT 95% CI')

# import scipy.stats
# scipy.stats.spearmanr(x, y)


# thresholds = np.sort(np.unique(x))

# # compute the mean of data.auc for all points with std_risk_scores >= each threshold
# mean_auc = [y[x >= th].mean() for th in thresholds]

# # plot the computed means as a line
# plt.plot(thresholds, mean_auc, color='red', label='Mean AUC for thresholds')
# plt.xlabel('Standard Risk Scores Threshold')
# plt.ylabel('Mean AUC')
# plt.legend()
# plt.show()


gpt_features = pd.read_csv("gpt_deciles.csv")
mistral_features = pd.read_csv("mistral_deciles.csv")
gpt = pd.read_csv("gpt.csv")
mistral = pd.read_csv("mistral.csv")
for df in [gpt_features, mistral_features, gpt, mistral]:
    df["dataset"] = df["dataset"].astype(str).apply(lambda x: x.split("_")[0])

gpt_features.drop(columns=["dataset"], inplace=True)
mistral_features.drop(columns=["dataset"], inplace=True)

import scipy.stats


# def compute_entropy(row):
#     nonzero = row[row != 0]
#     if nonzero.sum() == 0:
#         return 0.0
#     p = nonzero / nonzero.sum()
#     return scipy.stats.entropy(p, base=2)

# # Add std_risk column from the corresponding CSV files.

# # Compute entropy over non-zero columns for each row.
# # gpt_features['entropy'] = gpt_features.apply(compute_entropy, axis=1)
# # mistral_features['entropy'] = mistral_features.apply(compute_entropy, axis=1)

# # gpt_features['std_risk'] = gpt['std_risk'].values
# # mistral_features['std_risk'] = mistral['std_risk'].values


# use_gpt = False
# if use_gpt:
#     X = gpt_features
#     y = gpt['auc'].values
# else:
#     X = mistral_features
#     y = mistral['auc'].values

# # Initialize out-of-sample predictions array
# oof_preds = np.zeros(len(X))

# # Set up 5-fold cross-validation
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # Loop over each fold
# for train_idx, val_idx in kf.split(X):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train = y[train_idx]

#     # Fit an XGBoost model on 4 folds
#     model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
#     # model = LassoCV(alphas=np.logspace(-4, 0, 50), cv=5, random_state=42)
#     model.fit(X_train, y_train)

#     # Store predictions on the held-out fold
#     oof_preds[val_idx] = model.predict(X_val)

# # Create a scatter plot comparing predictions and observed values
# plt.scatter(oof_preds, y, alpha=0.7)
# plt.xlabel('Out-of-sample Predictions')
# plt.ylabel('Actual y')
# plt.title('Cross-Fitted Predictions vs Actual y')
# np.corrcoef(oof_preds, y)[0,1]**2
# # Compute lowess smoothed curve and bootstrap CI on the (oof_preds, y) scatter plot
# x_grid_pred = np.linspace(oof_preds.min(), oof_preds.max(), 100)
# smoothed_curve = lowess(y, oof_preds, frac=0.4, return_sorted=True)
# interp_func = interp1d(smoothed_curve[:, 0], smoothed_curve[:, 1],
#                        bounds_error=False, fill_value="extrapolate")
# fitted_line = interp_func(x_grid_pred)

# # Compute confidence intervals using bootstrapping
# ci_lower_pred, ci_upper_pred = bootstrap_loess_confidence_interval(
#     oof_preds, y, frac=0.4, grid=x_grid_pred, n_boot=1000, alpha=0.05)

# plt.fill_between(x_grid_pred, ci_lower_pred, ci_upper_pred, color='gray', alpha=0.2, label='Loess 95% CI')
# plt.plot(x_grid_pred, fitted_line, color='black', label='Loess Smoothed')
# plt.legend()
# plt.savefig(f"1.pdf", format="pdf", bbox_inches="tight")
# plt.clf()

# np.corrcoef(oof_preds, y)[0,1]**2

# thresholds = np.sort(np.unique(oof_preds))

# # compute the mean of data.auc for all points with std_risk_scores >= each threshold
# mean_auc = [y[oof_preds >= th].mean() for th in thresholds]

# # plot the computed means as a line
# plt.plot(thresholds, mean_auc, color='red', label='Mean AUC for thresholds')
# plt.xlabel('Standard Risk Scores Threshold')
# plt.ylabel('Mean AUC')
# plt.legend()
# plt.ylim(0.5, 1)
# plt.savefig(f"2.pdf", format="pdf", bbox_inches="tight")
# plt.clf()


# Group-level cross-validation using the 'dataset' column from the gpt dataframe
# Toggle between GPT and mistral datasets:
# Set use_gpt = True to use GPT data; set to False to use mistral data.
use_gpt = False

if use_gpt:
    groups = gpt["dataset"]
    features = gpt_features.copy()
    target = gpt["auc"].values
else:
    groups = mistral["dataset"]
    features = mistral_features.copy()
    target = mistral["auc"].values
n_splits = min(5, groups.nunique())

group_kf = GroupKFold(n_splits=n_splits)

# Initialize out-of-sample predictions array for the target variable
oof_preds = np.zeros(len(target))

for train_idx, test_idx in group_kf.split(features, target, groups):
    X_train = features.iloc[train_idx]
    X_test = features.iloc[test_idx]
    y_train = target[train_idx]
    y_test = target[test_idx]

    # Train model on training groups
    model = XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test groups
    oof_preds[test_idx] = model.predict(X_test)

# Indices of 5 largest values
largest_indices = np.argsort(oof_preds)[-10:][::-1]  # descending order

# Indices of 5 smallest values
smallest_indices = np.argsort(oof_preds)[:10]

# import pdb; pdb.set_trace()

decile_cols = [col for col in features.columns if col.startswith("decile_")]
deciles = features[decile_cols]

for ind in smallest_indices:
    min_hist = deciles.iloc[ind].to_numpy()
    sorted_data = np.sort(min_hist)
    print(np.std(sorted_data))
    # plt.hist(sorted_data, bins=40); plt.show(); plt.clf()
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, color="red", alpha=0.6)
print("_________________________________")
for ind in largest_indices:
    min_hist = deciles.iloc[ind].to_numpy()
    sorted_data = np.sort(min_hist)
    print(np.std(sorted_data))
    # plt.hist(sorted_data, bins=40); plt.show(); plt.clf()
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, color="blue", alpha=0.6)

plt.xlabel("Probability")
plt.ylabel("CDF")
# plt.grid(True)
plt.tight_layout()
# plt.savefig("../results_section_2/cdfs_mistral.pdf", format='pdf', bbox_inches='tight')
plt.savefig("cdfs_mistral.pdf", format="pdf", bbox_inches="tight")
plt.clf()


# Scatter plot of out-of-sample predictions versus actual target values
# plt.scatter(oof_preds, target, alpha=0.7)
# plt.xlabel('Predicted AUC')
# plt.ylabel('Actual AUC')
# # plt.title('Dataset-level Cross-Fitted Predictions vs Actual AUC')

# # Compute lowess smoothed curve & confidence intervals
# x_grid = np.linspace(oof_preds.min(), oof_preds.max(), 100)
# smoothed = lowess(target, oof_preds, frac=0.4, return_sorted=True)
# interp_func = interp1d(smoothed[:, 0], smoothed[:, 1], bounds_error=False, fill_value="extrapolate")
# fitted_line = interp_func(x_grid)

# ci_lower, ci_upper = bootstrap_loess_confidence_interval(
#     oof_preds, target, frac=0.4, grid=x_grid, n_boot=1000, alpha=0.05)

# plt.fill_between(x_grid, ci_lower, ci_upper, color='gray', alpha=0.2, label='Loess 95% CI')
# plt.xlim((.1,1.05)); plt.ylim((.1,1.05))
# plt.plot(x_grid, fitted_line, color='black', label='Loess Smoothed')
# # plt.legend()
# plt.savefig(f"../results_section_2/grouped_kfold_mistral.pdf", format="pdf", bbox_inches="tight")
# plt.clf()

# Create pandas DataFrame
df = pd.DataFrame({"x": oof_preds, "y": target})

# Push DataFrame to R
ro.globalenv["df"] = pandas2ri.py2rpy(df)

# R script to generate and save the plot
r_script = """
library(ggplot2)


p <- ggplot(df, aes(x = x, y = y)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_smooth(method = "loess", se = TRUE, color = "red") +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 20),        # Tick label size
    axis.title = element_text(size = 28)       # Axis title size
  ) +
  xlim(0.2,1.05)+
  ylim(0.15,1.05)+
  labs(x = "Predicted AUC", y = "Actual AUC")

# pdf("../results_section_2/grouped_kfold_mistral.pdf", width = 7, height = 5)
pdf("grouped_kfold_mistral.pdf", width = 7, height = 5)

print(p)
dev.off()

"""

# Run the R code
ro.r(r_script)

print("R^2:", np.corrcoef(oof_preds, target)[0, 1] ** 2)

# Plot mean AUC over increasing prediction thresholds
thresholds = np.sort(np.unique(oof_preds))
mean_auc = [target[oof_preds >= th].mean() for th in thresholds if len(target[oof_preds >= th]) >= 5]
plt.plot(thresholds[: len(mean_auc)], mean_auc, color="red", label="Mean AUC for thresholds")
plt.xlabel("Avg OOS Pred ≥ Threshold")
plt.ylabel("Mean AUC")
# plt.legend()
plt.ylim(0.5, 1)
# plt.savefig(f"../results_section_2/threshold_mistral.pdf", format="pdf", bbox_inches="tight")
plt.savefig(f"threshold_mistral.pdf", format="pdf", bbox_inches="tight")

plt.clf()

group_kf = GroupKFold(n_splits=n_splits)


# Initialize out-of-sample predictions array for gpt
oof_preds_gpt = np.zeros(len(gpt))

for train_idx, test_idx in group_kf.split(gpt_features, gpt["auc"], groups):
    X_train = gpt_features.iloc[train_idx]
    X_test = gpt_features.iloc[test_idx]
    y_train = gpt["auc"].values[train_idx]
    y_test = gpt["auc"].values[test_idx]

    # Train model on training groups
    model = XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test groups
    oof_preds_gpt[test_idx] = model.predict(X_test)

# import pdb; pdb.set_trace()

# Indices of 5 largest values
largest_indices = np.argsort(oof_preds_gpt)[-10:][::-1]  # descending order

# Indices of 5 smallest values
smallest_indices = np.argsort(oof_preds_gpt)[:10]

decile_cols = [col for col in gpt_features.columns if col.startswith("decile_")]
deciles = gpt_features[decile_cols]

for ind in smallest_indices:
    min_hist = deciles.iloc[ind].to_numpy()
    sorted_data = np.sort(min_hist)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, color="red", alpha=0.6)

for ind in largest_indices:
    min_hist = deciles.iloc[ind].to_numpy()
    sorted_data = np.sort(min_hist)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, color="blue", alpha=0.6)

plt.xlabel("Probability")
plt.ylabel("CDF")
# plt.grid(True)
plt.tight_layout()
# plt.savefig("../results_section_2/cdfs_gpt.pdf", format='pdf', bbox_inches='tight')
plt.savefig("cdfs_gpt.pdf", format="pdf", bbox_inches="tight")

plt.clf()


# Scatter plot of out-of-sample predictions versus actual AUC from gpt
# plt.scatter(oof_preds_gpt, gpt['auc'], alpha=0.7)
# plt.xlabel('Predicted AUC')
# plt.ylabel('Actual AUC')
# # plt.title('Dataset-level Cross-Fitted Predictions vs Actual AUC')

# # Compute lowess smoothed curve & confidence intervals
# x_grid = np.linspace(oof_preds_gpt.min(), oof_preds_gpt.max(), 100)
# smoothed = lowess(gpt['auc'], oof_preds_gpt, frac=0.4, return_sorted=True)
# interp_func = interp1d(smoothed[:, 0], smoothed[:, 1], bounds_error=False, fill_value="extrapolate")
# fitted_line = interp_func(x_grid)

# ci_lower, ci_upper = bootstrap_loess_confidence_interval(
#     oof_preds_gpt, gpt['auc'].values, frac=0.4, grid=x_grid, n_boot=1000, alpha=0.05)

# plt.fill_between(x_grid, ci_lower, ci_upper, color='gray', alpha=0.2, label='Loess 95% CI')
# plt.plot(x_grid, fitted_line, color='black', label='Loess Smoothed')
# plt.xlim((.1,1.05)); plt.ylim((.1,1.05))
# # plt.legend()
# plt.savefig(f"../results_section_2/grouped_kfold_gpt.pdf", format="pdf", bbox_inches="tight")
# plt.clf()

# Create pandas DataFrame
df = pd.DataFrame({"x": oof_preds_gpt, "y": gpt["auc"].tolist()})

# Push DataFrame to R
ro.globalenv["df"] = pandas2ri.py2rpy(df)

# R script to generate and save the plot
r_script = """
library(ggplot2)


p <- ggplot(df, aes(x = x, y = y)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_smooth(method = "loess", se = TRUE, color = "red") +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 20),        # Tick label size
    axis.title = element_text(size = 28)       # Axis title size
  ) +
  xlim(0.2,1.05)+
  ylim(0.15,1.05)+
  labs(x = "Predicted AUC", y = "Actual AUC")

# pdf("../results_section_2/grouped_kfold_gpt.pdf", width = 7, height = 5)
pdf("grouped_kfold_gpt.pdf", width = 7, height = 5)

print(p)
dev.off()

"""

# Run the R code
ro.r(r_script)

print("R^2:", np.corrcoef(oof_preds_gpt, gpt["auc"])[0, 1] ** 2)

# Plot mean AUC over increasing prediction thresholds
thresholds = np.sort(np.unique(oof_preds_gpt))
mean_auc = [gpt["auc"][oof_preds_gpt >= th].mean() for th in thresholds if len(gpt["auc"][oof_preds_gpt >= th]) >= 5]
plt.plot(thresholds[: len(mean_auc)], mean_auc, color="red", label="Mean AUC for thresholds")
plt.xlabel("Avg OOS Pred ≥ Threshold")
plt.ylabel("Mean AUC")
# plt.legend()
plt.ylim(0.5, 1)
# plt.savefig(f"../results_section_2/threshold_gpt.pdf", format="pdf", bbox_inches="tight")
plt.savefig(f"threshold_gpt.pdf", format="pdf", bbox_inches="tight")

plt.clf()


import statsmodels.api as sm


for frame, name in [(gpt, "gpt"), (mistral, "mistral")]:
    frame["normalized_auc"] = frame["auc"] / frame["xgb_auc"]
    # Create indicator variables for the 'dataset' column without dropping any categories
    dataset_indicators = pd.get_dummies(frame["dataset"], drop_first=False)
    # Fit linear regression of auc on the indicators without an intercept
    model = sm.OLS(frame["auc"], dataset_indicators).fit()
    print(f"R^2 for regression without intercept {name}:", model.rsquared)

    # Calculate overall variance of the 'auc' column (sample variance)
    overall_var = frame["auc"].var()

    # Calculate variance within each dataset group and then the average variance
    avg_within_var = frame.groupby("dataset")["auc"].var().mean()

    # Compute the ratio of average within-dataset variance to overall variance
    ratio = avg_within_var / overall_var

    print(f"Overall variance of auc {name}:", overall_var)
    print(f"Average within-dataset variance {name}:", avg_within_var)
    print(f"Ratio (within-dataset / overall {name}):", ratio)
