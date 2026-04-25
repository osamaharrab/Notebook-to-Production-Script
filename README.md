# From Notebook to Production — Churn Model Comparison CLI

A production CLI tool that compares churn classification models, selects an operating threshold from out-of-fold training predictions, and generates reproducible evaluation artifacts.

## Overview

- Compares six model configurations with stratified cross-validation
- Selects the best model using cross-validated PR-AUC
- Chooses an operating threshold from training out-of-fold predictions
- Evaluates the selected operating point on a held-out test set
- Saves plots, reports, metadata, and model artifacts to an output directory

The current default run selects **RF_default** as the best model and chooses **0.25** as the operating threshold using the `best_f1` strategy on out-of-fold training predictions.

## Models Compared

| Name          | Algorithm              | Key Settings                          |
|---------------|------------------------|---------------------------------------|
| Dummy         | DummyClassifier        | strategy=most_frequent                |
| LR_default    | LogisticRegression     | max_iter=1000, solver=lbfgs           |
| LR_balanced   | LogisticRegression     | max_iter=1000, class_weight=balanced  |
| DT_depth5     | DecisionTreeClassifier | max_depth=5                           |
| RF_default    | RandomForestClassifier | n_estimators=100                      |
| RF_balanced   | RandomForestClassifier | n_estimators=100, class_weight=balanced |

Logistic regression pipelines use median imputation and standard scaling. Tree-based pipelines use median imputation without scaling.

## Project Structure

```
.
├── model_comparison.py
├── setup.sh
├── README.md
├── DECISION_MEMO.md
├── requirements.txt
├── config.json
├── data/
│   └── telecom_churn.csv
└── output/
    ├── comparison_table.csv
    ├── pr_curves.png
    ├── calibration.png
    ├── best_model.joblib
    ├── experiment_log.csv
    ├── threshold_tuning.csv
    ├── threshold_sweep.png
    ├── threshold_recommendation.md
    ├── operating_metrics.json
    ├── calibration_metrics.csv
    ├── test_predictions.csv
    ├── error_analysis.md
    ├── tree_vs_linear_disagreement.md
    ├── data_quality_report.json
    ├── run_metadata.json
    └── runs/
        └── <timestamp>/
```

## Setup

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python model_comparison.py [options]
```

### Arguments

| Argument         | Default                | Description                                           |
|------------------|------------------------|-------------------------------------------------------|
| `--data-path`    | `data/telecom_churn.csv` | Path to the input CSV dataset                       |
| `--output-dir`   | `./output`             | Directory where all artifacts will be saved           |
| `--n-folds`      | `5`                    | Number of stratified cross-validation folds           |
| `--random-seed`  | `42`                   | Random seed for reproducibility                       |
| `--dry-run`      | off                    | Validate the dataset and configuration without training |
| `--config-path`  | `config.json`          | Path to a JSON config file                            |
| `--log-level`    | `INFO`                 | Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL  |

### Example Commands

**Dry run:**

```bash
python model_comparison.py --dry-run
```

**Standard run:**

```bash
python model_comparison.py
```

**Custom CV and logging:**

```bash
python model_comparison.py --n-folds 7 --random-seed 123 --log-level DEBUG
```

### Configuration File

`config.json` controls model selection and threshold behavior (CLI arguments override):

```json
{
  "selection_metric": "pr_auc_mean",
  "threshold_selection_strategy": "best_f1",
  "recall_target": 0.8,
  "max_alert_rate": null,
  "calibration_bins": 10,
  "error_analysis_top_n": 10,
  "run_snapshot": true
}
```

## Output Artifacts

| File                              | Description                                                              |
|-----------------------------------|--------------------------------------------------------------------------|
| `comparison_table.csv`            | Cross-validated metrics for all models                                   |
| `pr_curves.png`                   | Precision-recall curves for the top models                               |
| `calibration.png`                 | Calibration curves for the top models                                    |
| `best_model.joblib`               | Serialized best model fitted on the full training set                    |
| `experiment_log.csv`              | Timestamped summary metrics                                              |
| `threshold_tuning.csv`            | Threshold sweep table using training out-of-fold predictions             |
| `threshold_sweep.png`             | Precision, recall, F1, and alerts per 1,000 by threshold                 |
| `threshold_recommendation.md`     | Selected operating threshold and alternatives                            |
| `operating_metrics.json`          | Final test-set metrics at the selected threshold (and at 0.50)           |
| `calibration_metrics.csv`         | Brier score, ECE, and test PR-AUC for each model                        |
| `test_predictions.csv`            | Row-level predicted probabilities and class decisions                    |
| `error_analysis.md`               | Highest-confidence false positives and lowest-confidence false negatives |
| `tree_vs_linear_disagreement.md`  | Where tree and linear models differ structurally                         |
| `data_quality_report.json`        | Dataset checks and suspicious data patterns                              |
| `run_metadata.json`               | Configuration, selected model, threshold, and evaluation summary        |

## Current Result Summary

**At selected threshold 0.25:**

| Metric             | 0.25 (selected) | 0.50 (default) |
|--------------------|------------------|-----------------|
| Accuracy           | 0.7656           | 0.8456          |
| Precision          | 0.3609           | 0.5833          |
| Recall             | 0.5646           | 0.1905          |
| F1                 | 0.4403           | 0.2872          |
| Predicted positives| 230 / 900        | 48 / 900        |
| Alert rate         | 25.6%            | 5.3%            |

At 0.50 the model catches only ~19% of churners; at 0.25 it catches ~56%. The lower threshold is more appropriate when catching churners matters more than minimizing false positives.

## Threshold Selection Logic

Threshold selection uses training out-of-fold predictions only — the held-out test set is reserved for honest evaluation.

Supported strategies: `best_f1`, `target_recall_with_best_precision`, `max_recall_at_or_below_alert_rate`.

## Data Quality

- 4500 rows, 14 columns, no missing values, no duplicates
- 278 suspicious rows where `tenure > 3` and `total_charges == 0` — may be a data-entry issue

## Decision Rationale

See **DECISION_MEMO.md** for the full deployment recommendation.
