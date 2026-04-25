# Petra Telecom Churn Model Comparison CLI

A production-style command-line tool for comparing churn classification models, selecting an operating threshold from out-of-fold training predictions, and generating reproducible evaluation artifacts for business decision-making.

## Overview

This project evaluates multiple classification pipelines for Petra Telecom churn prediction and turns the workflow into a reusable CLI application instead of a notebook-only analysis. The script:

- compares six model configurations with stratified cross-validation
- selects the best model using cross-validated PR-AUC by default
- chooses an operating threshold from training out-of-fold predictions
- evaluates the selected operating point on a held-out test set
- saves plots, reports, logs, metadata, and model artifacts to an output directory

The current default run selects **RF_default** as the best model and chooses **0.25** as the operating threshold using the `best_f1` strategy on out-of-fold training predictions.

## Features

- Command-line interface with clear arguments and help text
- Dry-run mode to validate data and configuration without training models
- Cross-validation model comparison across six configurations
- Threshold selection without test leakage using training out-of-fold probabilities
- Calibration analysis with calibration curves, Brier score, and ECE
- Error analysis at the selected operating threshold
- Tree-vs-linear disagreement analysis for interpretability
- Data quality report with integrity checks and suspicious-pattern detection
- Run metadata and snapshot outputs for reproducibility

## Models Compared

The default configuration compares six pipelines:

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

### Recommended setup

Use the provided setup script to create the virtual environment and install dependencies automatically.

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

The setup script will:

- create `.venv` if it does not already exist
- activate the virtual environment
- upgrade pip, setuptools, and wheel
- install packages from `requirements.txt`
- optionally install `requirements-dev.txt` if it exists
- optionally run `test_environment.py` if it exists

### Manual setup

If you prefer to set up the environment manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Requirements

`requirements.txt`:

```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
joblib>=1.3
scikit-learn>=1.3
```

## Usage

After setup, activate the environment and run the CLI with a dataset path:

```bash
source .venv/bin/activate
python model_comparison.py --data-path data/telecom_churn.csv [options]
```

### Arguments

| Argument         | Required | Default    | Description                                                        |
|------------------|----------|------------|--------------------------------------------------------------------|
| `--data-path`    | Yes      | —          | Path to the input CSV dataset                                      |
| `--output-dir`   | No       | `./output` | Directory where all artifacts will be saved                        |
| `--n-folds`      | No       | `5`        | Number of stratified cross-validation folds                        |
| `--random-seed`  | No       | `42`       | Random seed for reproducibility                                    |
| `--dry-run`      | No       | off        | Validate the dataset and configuration without training            |
| `--config-path`  | No       | config.json| Optional path to a JSON config file                                |
| `--log-level`    | No       | INFO       | Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL               |

### Example Commands

#### 1) First-time project setup

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

#### 2) Dry run

```bash
source .venv/bin/activate
python model_comparison.py --data-path data/telecom_churn.csv --dry-run
```

Use this first to confirm that:

- the dataset exists
- required columns are present
- class distribution is valid
- cross-validation settings are feasible
- output paths and configuration are correct

#### 3) Standard run

```bash
source .venv/bin/activate
python model_comparison.py --data-path data/telecom_churn.csv --output-dir output
```

#### 4) Custom CV and logging level

```bash
source .venv/bin/activate
python model_comparison.py \
  --data-path data/telecom_churn.csv \
  --output-dir output_debug \
  --n-folds 7 \
  --random-seed 123 \
  --log-level DEBUG
```

### Optional Configuration File

You can place a `config.json` file next to the script or pass a custom path with `--config-path`. CLI arguments override config values.

Example:

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

A successful run generates the following artifacts:

### Core artifacts

| File                    | Description                                                     |
|-------------------------|-----------------------------------------------------------------|
| `comparison_table.csv`  | Cross-validated metrics for all models                          |
| `pr_curves.png`         | Precision-recall curves for the top models                      |
| `calibration.png`       | Calibration curves for the top models                           |
| `best_model.joblib`     | Serialized best model fitted on the full training set           |
| `experiment_log.csv`    | Flat experiment log with timestamped summary metrics            |

### Threshold and deployment artifacts

| File                        | Description                                                                  |
|-----------------------------|------------------------------------------------------------------------------|
| `threshold_tuning.csv`      | Threshold sweep table using training out-of-fold predictions                 |
| `threshold_sweep.png`       | Plot of precision, recall, F1, and alerts per 1,000 by threshold             |
| `threshold_recommendation.md` | Selected operating threshold and alternative candidates                     |
| `operating_metrics.json`    | Final test-set metrics at the selected operating threshold                   |

### Diagnostics and interpretability artifacts

| File                                | Description                                                              |
|-------------------------------------|--------------------------------------------------------------------------|
| `calibration_metrics.csv`           | Brier score, ECE, and test PR-AUC for each model                        |
| `test_predictions.csv`              | Row-level predicted probabilities and class decisions                    |
| `error_analysis.md`                 | Highest-confidence false positives and lowest-confidence false negatives |
| `tree_vs_linear_disagreement.md`    | An example showing how tree and linear models differ structurally        |
| `data_quality_report.json`          | Dataset checks and suspicious data patterns                             |
| `run_metadata.json`                 | Saved configuration, selected model, threshold, and evaluation summary  |

### Snapshot artifacts

| File                            | Description                                                   |
|---------------------------------|---------------------------------------------------------------|
| `output/runs/<timestamp>/`      | A snapshot copy of the major outputs for reproducibility      |

## Decision Summary

For the final model recommendation and business-facing deployment rationale, see `DECISION_MEMO.md`.

## Current Result Summary

From the current run:

- **Best model:** RF_default
- **Model selection metric:** pr_auc_mean
- **Selected threshold:** 0.25
- **Threshold strategy:** best_f1

**Test-set operating metrics at selected threshold 0.25:**

| Metric             | Value  |
|--------------------|--------|
| Accuracy           | 0.7656 |
| Precision          | 0.3609 |
| Recall             | 0.5646 |
| F1                 | 0.4403 |
| PR-AUC             | 0.4286 |
| Brier score        | 0.1154 |
| Predicted positives| 230 / 900 |
| Alert rate         | 0.2556 |

**Confusion matrix at threshold 0.25:**

|       | Pred Neg | Pred Pos |
|-------|----------|----------|
| **Actual Neg** | 606 (TN) | 147 (FP) |
| **Actual Pos** | 64 (FN)  | 83 (TP)  |

**Test-set metrics at default threshold 0.50 (for comparison):**

| Metric             | Value  |
|--------------------|--------|
| Accuracy           | 0.8456 |
| Precision          | 0.5833 |
| Recall             | 0.1905 |
| F1                 | 0.2872 |
| Predicted positives| 48 / 900 |
| Alert rate         | 0.0533 |

**Confusion matrix at threshold 0.50:**

|       | Pred Neg | Pred Pos |
|-------|----------|----------|
| **Actual Neg** | 733 (TN) | 20 (FP)  |
| **Actual Pos** | 119 (FN) | 28 (TP)  |

### Interpretation

The selected threshold (0.25) produces a much more recall-oriented operating point than the default 0.50 threshold. At 0.50, the model only catches ~19% of churners; at 0.25, it catches ~56%. This makes the model more useful for churn intervention, but it also increases false positives and outreach volume (alert rate rises from ~5% to ~26%). In other words, the current setup is appropriate when catching more true churners is more valuable than minimizing unnecessary retention offers.

## Data Quality Notes

The latest data quality report shows:

- 4500 rows and 14 columns
- no missing values in the required modeling columns
- no duplicate rows
- no negative numeric values in the required features
- 278 suspicious rows where `tenure > 3` and `total_charges == 0`

That last pattern should be reviewed because it may represent a data-entry issue, a business edge case, or zero-filled missing values.

## Threshold Selection Logic

Threshold selection is intentionally done on training out-of-fold predictions, not on the held-out test set. This keeps the final test set reserved for honest evaluation.

Supported threshold strategies:

- **best_f1** — choose the threshold that maximizes out-of-fold F1
- **target_recall_with_best_precision** — among thresholds meeting the recall target, choose the one with the best precision
- **max_recall_at_or_below_alert_rate** — among thresholds under the alert-rate cap, choose the one with the highest recall

## Why PR-AUC Is the Default Selection Metric

The churn dataset is imbalanced, so accuracy alone is not a reliable ranking criterion. PR-AUC is more informative because it measures how well a model ranks the minority class across possible thresholds.

## Troubleshooting

### File not found

If you see a file-not-found error, confirm the path passed to `--data-path`.

```bash
source .venv/bin/activate
python model_comparison.py --data-path data/telecom_churn.csv --dry-run
```

### Missing dependencies

If you see an error such as `ModuleNotFoundError`, run setup again:

```bash
./setup.sh
source .venv/bin/activate
```

Or install dependencies manually:

```bash
python -m pip install -r requirements.txt
```

### Validation failed

If validation fails, check:

- required columns exist
- `churned` is binary
- CV folds are not larger than the minority-class count
- numeric features do not contain invalid text values

### Output looks too conservative or too aggressive

Review:

- `threshold_recommendation.md`
- `threshold_tuning.csv`
- `operating_metrics.json`

If outreach capacity is limited, use a more restrictive threshold strategy or introduce `max_alert_rate` in `config.json`.

## Reproducibility

Each run saves:

- the selected best model
- the selected operating threshold
- the configuration used
- the data quality report
- the operating metrics
- a timestamped snapshot of outputs

This makes it easier to compare runs and trace how a deployment recommendation was produced.
