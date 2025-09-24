# Predicting Language Models’ Success at Zero-Shot Probabilistic Prediction

[![arXiv](https://img.shields.io/badge/arXiv-2509.15356-b31b1b.svg)](https://arxiv.org/abs/2509.15356)

This repository collects the scripts and configs used to reproduce the experiments from the paper [Predicting Language Models’ Success at Zero-Shot Probabilistic Prediction](https://arxiv.org/abs/2509.15356) across
GPT-4o, GPT-4o-mini, Llama, and Mistral. Each model family lives in its own top-level directory (`4o/`,
`4o-mini/`, `llama/`, `mistral/`), and shared analysis lives under `agreement_analysis/`.

## Environment Setup

The project relies on [uv](https://docs.astral.sh/uv/) for dependency management. From the repository root:

```bash
uv venv
source .venv/bin/activate
uv pip install requirements.txt
uv pip install -e llama
```

The editable install exposes the a fork of the [`folktexts`](https://github.com/socialfoundations/folktexts) package and the command-line entry points used throughout the
experiments. The helper scripts expect to be run from the repository root unless stated otherwise.

## End-to-End Workflow

The pipeline is the same for each model directory (`4o/`, `4o-mini/`, `llama/`, `mistral/`). The example below
uses the Llama folder, but substitute the model name as needed.

### 1. Prepare configs and small test runs

1. Open `llama/experiments/` (or the matching model directory) and reduce the default dataset subsample size in
   experiment definitions from 1000 rows to 100–200 rows when doing quick smoke tests.
2. In `llama/make_configs.py`, limit the dataset list before generating configs (e.g. slice with `[:5]`) so you
   can iterate quickly.
3. Generate configs for each experiment type (classification, masking, confidence, no-data) by running
   `config_maker.py`. This creates a subdirectory in `configs/` for every dataset/experiment combination.

### 2. Launch batched experiments

1. Run `main_batch_runner.py` on each config folder to execute all configs it contains. The script dispatches
   `main.py` with the appropriate arguments for every config file.
2. After each run, rename the resulting folder inside `results/` before launching the next experiment so results
   stay separated (e.g. `results/gpt_masking`, `results/gpt_classification`).

### 3. Consolidate results for figure generation

1. Update the folder paths at the top of `copy_all_zero_shot.py` to point to the renamed classification and
   masking result folders, then run the script to collect zero-shot results.
2. Adjust the paths in `get_all_results.py`, uncomment any needed lines, and run it to produce `all_results.csv`
   summarizing masking, classification, confidence, AUC prediction, decimal scoring, and integer scoring runs.
3. Execute `remake_zero_shot.py` to recreate the figures and CSVs used downstream. This produces additional files
   that feed into the agreement analysis step.

### 4. Run agreement analysis scripts

1. Copy the renamed `all_results.csv` into `agreement_analysis/`.
2. Move the CSV outputs from `remake_zero_shot.py` into `agreement_analysis/bryan_analysis/` and
   `agreement_analysis/bryan_analysis_2/`.
3. From within `agreement_analysis/`, execute the following scripts to regenerate the paper figures:
   - `agreement.py`
   - `experiments_with_auc.py`
   - `fig_1.py`
   - `redo_masking.py`
   - `zero_shot_xgb.py`

## Key Directories and Scripts

- `configs/`: generated config files grouped by experiment run (e.g. `configs/gpt_masking/`).
- `data_encs/`: dataset variable definitions.
- `experiments/`: experiment definitions (masking, classification, confidence, etc.).
- `folktexts/`: fork of the FolkTexts evaluation package used by the runners.
- `results/`: default output location for experiment runs.
- `remake_zero_shot_with_masking/`: plots and CSVs emitted by `remake_zero_shot_with_masking.py`.
- `task_prompts/`: prompt templates for classification and confidence experiments.
- `xgb_pickles/`: shared feature column selections and XGBoost outputs (only present in GPT-4o-mini).

### Script Quick Reference

- `main.py`: run a single experiment for a specific model/dataset/metric using one config file.
- `main_batch_runner.py`: iterate over all configs in a folder and invoke `main.py` for each.
- `copy_all_zero_shot.py`: gather zero-shot and masking results into `all_zero_shot/` (folder name varies per model).
- `get_all_results.py`: merge outputs from all experiment types into a single CSV.
- `remake_zero_shot_with_masking.py`: generate risk-score distributions, AUC/ECE histograms, calibration curves, and
  supporting CSVs for downstream analysis.
- `xgb_runner.py`, `xgb_runner_masking.py`: scripts to run XGBoost baselines.

## Final Steps

After reproducing the baseline pipeline, consult `agreement_analysis/` for additional plots and regression analyses,
and adjust config generation to scale beyond the quick-test subsets.

## Citation

```bib
@inproceedings{ren2025predicting,
  title        = {Predicting Language Models' Success at Zero-Shot Probabilistic Prediction},
  author       = {Kevin Ren and Santiago Cortes-Gomez and Carlos Miguel Patiño and Ananya Joshi and Ruiqi Lyu and Jingjing Tang and Alistair Turcan and Khurram Yamin and Steven Wu and Bryan Wilder},
  booktitle    = {The 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)},
  year         = {2025},
  url          = {https://arxiv.org/abs/2509.15356}
}
```
