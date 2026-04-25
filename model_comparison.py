#!/usr/bin/env python3
"""Petra Telecom Churn Model Comparison CLI.

A production-style command-line tool for comparing churn classification
models, selecting an operating threshold from out-of-fold training
predictions, and generating reproducible evaluation artifacts.
"""

import argparse
import copy
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.calibration import calibration_curve
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "customer_id", "gender", "senior_citizen", "tenure", "monthly_charges",
    "total_charges", "contract_type", "internet_service", "num_support_calls",
    "payment_method", "has_partner", "has_dependents", "churned", "contract_months",
]

NUMERIC_FEATURES = [
    "senior_citizen", "tenure", "monthly_charges", "total_charges",
    "num_support_calls", "has_partner", "has_dependents", "contract_months",
]

CATEGORICAL_FEATURES = ["gender", "contract_type", "internet_service", "payment_method"]

TARGET_COL = "churned"
DROP_COLS = ["customer_id"]

SCORING_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

DEFAULT_CONFIG = {
    "selection_metric": "pr_auc_mean",
    "threshold_selection_strategy": "best_f1",
    "recall_target": 0.8,
    "max_alert_rate": None,
    "calibration_bins": 10,
    "error_analysis_top_n": 10,
    "run_snapshot": True,
}

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


def _build_models(random_seed: int) -> dict:
    """Return the dictionary of models to compare."""
    return {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "LR_default": LogisticRegression(
            max_iter=1000, random_state=random_seed, solver="lbfgs",
        ),
        "LR_balanced": LogisticRegression(
            max_iter=1000, random_state=random_seed, solver="lbfgs",
            class_weight="balanced",
        ),
        "DT_depth5": DecisionTreeClassifier(
            max_depth=5, random_state=random_seed,
        ),
        "RF_default": RandomForestClassifier(
            n_estimators=100, random_state=random_seed, n_jobs=-1,
        ),
        "RF_balanced": RandomForestClassifier(
            n_estimators=100, random_state=random_seed, n_jobs=-1,
            class_weight="balanced",
        ),
    }


def _build_pipeline(name: str, model, random_seed: int) -> Pipeline:
    """Build a preprocessing + model pipeline.

    Logistic regression pipelines use median imputation + standard scaling.
    Tree-based pipelines use median imputation only (no scaling).
    """
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    is_linear = name.startswith("LR") or name == "Dummy"
    num_transformer = Pipeline([
        ("imputer", num_imputer),
        ("scaler", StandardScaler() if is_linear else "passthrough"),
    ])
    cat_transformer = Pipeline([
        ("imputer", cat_imputer),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="infrequent_if_exist")),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, NUMERIC_FEATURES),
        ("cat", cat_transformer, CATEGORICAL_FEATURES),
    ])

    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


# ---------------------------------------------------------------------------
# Data I/O and validation
# ---------------------------------------------------------------------------


def load_data(data_path: str) -> pd.DataFrame:
    """Load CSV dataset; exit on failure."""
    path = Path(data_path)
    if not path.exists():
        logger.error("Data file not found: %s", path.resolve())
        sys.exit(1)
    try:
        df = pd.read_csv(path)
        logger.info("Loaded data from %s (%d rows, %d columns)", path.resolve(), len(df), len(df.columns))
        return df
    except Exception as exc:
        logger.error("Failed to read data file %s: %s", path, exc)
        sys.exit(1)


def validate_data(df: pd.DataFrame, n_folds: int) -> None:
    """Validate DataFrame structure; exit on failure."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        logger.error("Missing required columns: %s", sorted(missing))
        sys.exit(1)

    if TARGET_COL not in df.columns:
        logger.error("Target column '%s' not found in dataset", TARGET_COL)
        sys.exit(1)

    if df[TARGET_COL].isnull().any():
        logger.warning("Target column has %d null values", df[TARGET_COL].isnull().sum())

    minority = df[TARGET_COL].value_counts().min()
    if n_folds > minority:
        logger.error("n_folds (%d) exceeds minority class count (%d) — reduce --n-folds", n_folds, minority)
        sys.exit(1)

    dist = df[TARGET_COL].value_counts().to_dict()
    logger.info("Validation passed — shape: %s, target distribution: %s", df.shape, dist)

    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        logger.warning("Columns with missing values:\n%s", cols_with_nulls.to_string())
    else:
        logger.info("No missing values detected")


def generate_data_quality_report(df: pd.DataFrame, output_dir: Path) -> dict:
    """Generate and save a data quality report."""
    report = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": {col: int(df[col].isnull().sum()) for col in df.columns if df[col].isnull().sum() > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": df[TARGET_COL].value_counts().to_dict(),
        "suspicious_patterns": [],
    }

    # Check for negative numeric values
    for col in NUMERIC_FEATURES:
        if df[col].dtype in [np.float64, np.int64]:
            neg_count = int((df[col] < 0).sum())
            if neg_count > 0:
                report["suspicious_patterns"].append(
                    {"pattern": f"negative_values_in_{col}", "count": neg_count}
                )

    # Check for tenure > 3 and total_charges == 0
    if "tenure" in df.columns and "total_charges" in df.columns:
        suspicious = int(((df["tenure"] > 3) & (df["total_charges"] == 0)).sum())
        if suspicious > 0:
            report["suspicious_patterns"].append(
                {"pattern": "tenure_gt_3_and_total_charges_zero", "count": suspicious}
            )

    report_path = output_dir / "data_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved data quality report → %s", report_path)
    return report


# ---------------------------------------------------------------------------
# Cross-validation model comparison
# ---------------------------------------------------------------------------


def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: dict,
    n_folds: int,
    random_seed: int,
) -> tuple:
    """Run stratified CV for all models, returning metrics and OOF predictions."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    results = {}
    oof_probas = {}

    for name, model in models.items():
        pipe = _build_pipeline(name, model, random_seed)
        logger.info("Training %s with %d-fold CV (seed=%d)…", name, n_folds, random_seed)

        # Get OOF predicted probabilities for the positive class
        oof_proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        oof_probas[name] = oof_proba

        # Compute per-fold metrics
        fold_metrics = {m: [] for m in SCORING_METRICS}
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            pipe_clone = _build_pipeline(name, copy.deepcopy(model), random_seed)
            pipe_clone.fit(X_tr, y_tr)
            y_pred = pipe_clone.predict(X_val)
            y_proba = pipe_clone.predict_proba(X_val)[:, 1]

            fold_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
            fold_metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
            fold_metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
            fold_metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
            fold_metrics["roc_auc"].append(roc_auc_score(y_val, y_proba))
            fold_metrics["pr_auc"].append(average_precision_score(y_val, y_proba))

        results[name] = {}
        for metric, values in fold_metrics.items():
            results[name][metric] = {"mean": round(np.mean(values), 4), "std": round(np.std(values), 4)}
            logger.info("  %s: %.4f ± %.4f", metric, np.mean(values), np.std(values))

    logger.info("All models cross-validated successfully")
    return results, oof_probas


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def select_best_model(results: dict, selection_metric: str) -> str:
    """Select best model based on mean of selection_metric."""
    metric_key = selection_metric.replace("_mean", "")
    best = max(results, key=lambda m: results[m].get(metric_key, results[m].get("pr_auc", {}))["mean"])
    logger.info("Selected best model: %s (%s = %.4f)", best, selection_metric, results[best][metric_key]["mean"])
    return best


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


def select_threshold(
    oof_proba: np.ndarray,
    y_train: pd.Series,
    strategy: str,
    recall_target: float = 0.8,
    max_alert_rate: float | None = None,
) -> tuple:
    """Select operating threshold from OOF predictions.

    Returns (threshold, sweep_df).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_proba)
    # precision_recall_curve returns extra point; trim to align
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    alert_rates = np.array([
        (oof_proba >= t).sum() / len(oof_proba) for t in thresholds
    ])

    sweep = pd.DataFrame({
        "threshold": thresholds,
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
        "alert_rate": alert_rates,
        "alerts_per_1000": (alert_rates * 1000).astype(int),
    })

    if strategy == "best_f1":
        idx = sweep["f1"].idxmax()
    elif strategy == "target_recall_with_best_precision":
        candidates = sweep[sweep["recall"] >= recall_target]
        if candidates.empty:
            logger.warning("No threshold meets recall_target=%.2f — falling back to best_f1", recall_target)
            idx = sweep["f1"].idxmax()
        else:
            idx = candidates["precision"].idxmax()
    elif strategy == "max_recall_at_or_below_alert_rate":
        if max_alert_rate is None:
            logger.warning("max_alert_rate not set — falling back to best_f1")
            idx = sweep["f1"].idxmax()
        else:
            candidates = sweep[sweep["alert_rate"] <= max_alert_rate]
            if candidates.empty:
                logger.warning("No threshold meets max_alert_rate=%.4f — falling back to best_f1", max_alert_rate)
                idx = sweep["f1"].idxmax()
            else:
                idx = candidates["recall"].idxmax()
    else:
        logger.warning("Unknown strategy '%s' — defaulting to best_f1", strategy)
        idx = sweep["f1"].idxmax()

    chosen_threshold = round(float(sweep.loc[idx, "threshold"]), 4)
    logger.info(
        "Selected threshold: %.4f (strategy=%s, precision=%.4f, recall=%.4f, f1=%.4f)",
        chosen_threshold, strategy,
        sweep.loc[idx, "precision"], sweep.loc[idx, "recall"], sweep.loc[idx, "f1"],
    )
    return chosen_threshold, sweep


# ---------------------------------------------------------------------------
# Final test-set evaluation
# ---------------------------------------------------------------------------


def evaluate_on_test(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict:
    """Evaluate fitted pipeline on test set at the given threshold."""
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        "brier_score": round(brier_score_loss(y_test, y_proba), 4),
        "predicted_positives": int(y_pred.sum()),
        "total_rows": len(y_test),
        "alert_rate": round(y_pred.sum() / len(y_test), 4),
        "confusion_matrix": {
            "tn": int(confusion_matrix(y_test, y_pred, labels=[0, 1])[0, 0]),
            "fp": int(confusion_matrix(y_test, y_pred, labels=[0, 1])[0, 1]),
            "fn": int(confusion_matrix(y_test, y_pred, labels=[0, 1])[1, 0]),
            "tp": int(confusion_matrix(y_test, y_pred, labels=[0, 1])[1, 1]),
        },
    }
    logger.info("Test-set metrics at threshold %.2f: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
                threshold, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"])

    test_preds = pd.DataFrame({
        "y_true": y_test.values,
        "y_proba": y_proba,
        "y_pred": y_pred,
    })

    # Also compute metrics at the default 0.50 threshold for comparison
    y_pred_50 = (y_proba >= 0.50).astype(int)
    metrics_at_050 = {
        "threshold": 0.50,
        "accuracy": round(accuracy_score(y_test, y_pred_50), 4),
        "precision": round(precision_score(y_test, y_pred_50, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred_50, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred_50, zero_division=0), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        "brier_score": round(brier_score_loss(y_test, y_proba), 4),
        "predicted_positives": int(y_pred_50.sum()),
        "total_rows": len(y_test),
        "alert_rate": round(y_pred_50.sum() / len(y_test), 4),
        "confusion_matrix": {
            "tn": int(confusion_matrix(y_test, y_pred_50, labels=[0, 1])[0, 0]),
            "fp": int(confusion_matrix(y_test, y_pred_50, labels=[0, 1])[0, 1]),
            "fn": int(confusion_matrix(y_test, y_pred_50, labels=[0, 1])[1, 0]),
            "tp": int(confusion_matrix(y_test, y_pred_50, labels=[0, 1])[1, 1]),
        },
    }

    metrics["default_050_metrics"] = metrics_at_050
    logger.info("Test-set metrics at default threshold 0.50: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
                metrics_at_050["accuracy"], metrics_at_050["precision"], metrics_at_050["recall"], metrics_at_050["f1"])

    return metrics, test_preds


# ---------------------------------------------------------------------------
# Calibration analysis
# ---------------------------------------------------------------------------


def calibration_analysis(
    pipes: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_bins: int,
    output_dir: Path,
) -> dict:
    """Compute calibration curves, Brier score, ECE for top models."""
    # Only analyze non-Dummy models
    top_models = {k: v for k, v in pipes.items() if k != "Dummy"}
    cal_metrics = {}

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, pipe in top_models.items():
        y_proba = pipe.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=n_bins)
        brier = brier_score_loss(y_test, y_proba)
        pr_ap = average_precision_score(y_test, y_proba)

        # ECE
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
            n_bin = mask.sum()
            if n_bin > 0:
                ece += abs(y_test[mask].mean() - y_proba[mask].mean()) * n_bin
        ece /= len(y_test)

        cal_metrics[name] = {
            "brier_score": round(brier, 4),
            "ece": round(ece, 4),
            "pr_auc": round(pr_ap, 4),
        }
        ax.plot(prob_pred, prob_true, marker="o", label=f"{name} (Brier={brier:.4f})")

    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    cal_path = output_dir / "calibration.png"
    fig.savefig(cal_path, dpi=150)
    plt.close(fig)
    logger.info("Saved calibration plot → %s", cal_path)

    cal_df = pd.DataFrame([
        {"model": name, "brier_score": v["brier_score"], "ece": v["ece"], "pr_auc": v["pr_auc"]}
        for name, v in cal_metrics.items()
    ])
    cal_df.to_csv(output_dir / "calibration_metrics.csv", index=False)
    logger.info("Saved calibration metrics → %s", output_dir / "calibration_metrics.csv")

    return cal_metrics


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------


def error_analysis(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    top_n: int,
    output_dir: Path,
) -> None:
    """Generate error analysis markdown."""
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    df_analysis = X_test.copy()
    df_analysis["y_true"] = y_test.values
    df_analysis["y_proba"] = y_proba
    df_analysis["y_pred"] = y_pred

    # False positives: predicted positive but actually negative
    fp = df_analysis[(df_analysis["y_true"] == 0) & (df_analysis["y_pred"] == 1)]
    fp_top = fp.nlargest(top_n, "y_proba")

    # False negatives: predicted negative but actually positive
    fn = df_analysis[(df_analysis["y_true"] == 1) & (df_analysis["y_pred"] == 0)]
    fn_top = fn.nsmallest(top_n, "y_proba")

    lines = [
        f"# Error Analysis (threshold={threshold})",
        "",
        f"## Top {top_n} Highest-Confidence False Positives",
        "",
        "These customers were predicted to churn but did not.",
        "",
    ]
    if not fp_top.empty:
        lines.append(fp_top[["y_proba", "y_true", "y_pred"] + NUMERIC_FEATURES[:5]].to_string())
    else:
        lines.append("No false positives at this threshold.")

    lines.extend([
        "",
        f"## Top {top_n} Lowest-Confidence False Negatives",
        "",
        "These customers churned but were predicted not to.",
        "",
    ])
    if not fn_top.empty:
        lines.append(fn_top[["y_proba", "y_true", "y_pred"] + NUMERIC_FEATURES[:5]].to_string())
    else:
        lines.append("No false negatives at this threshold.")

    (output_dir / "error_analysis.md").write_text("\n".join(lines))
    logger.info("Saved error analysis → %s", output_dir / "error_analysis.md")


# ---------------------------------------------------------------------------
# Tree vs linear disagreement
# ---------------------------------------------------------------------------


def tree_vs_linear_disagreement(
    pipes: dict,
    X_test: pd.DataFrame,
    threshold: float,
    output_dir: Path,
) -> None:
    """Analyze where tree and linear models disagree."""
    tree_models = [n for n in pipes if n.startswith(("RF", "DT"))]
    linear_models = [n for n in pipes if n.startswith("LR")]

    if not tree_models or not linear_models:
        logger.info("Skipping tree-vs-linear disagreement — need both types")
        return

    tree_name = tree_models[0]
    linear_name = linear_models[0]
    tree_proba = pipes[tree_name].predict_proba(X_test)[:, 1]
    linear_proba = pipes[linear_name].predict_proba(X_test)[:, 1]

    tree_pred = (tree_proba >= threshold).astype(int)
    linear_pred = (linear_proba >= threshold).astype(int)

    disagree_mask = tree_pred != linear_pred
    n_disagree = int(disagree_mask.sum())

    lines = [
        "# Tree vs Linear Model Disagreement",
        "",
        f"Comparing **{tree_name}** (tree) vs **{linear_name}** (linear) at threshold {threshold}.",
        "",
        f"Total disagreements: {n_disagree} / {len(X_test)} ({n_disagree/len(X_test):.1%})",
        "",
        "## Why They Disagree",
        "",
        "Tree models capture non-linear interactions (e.g., short tenure + high charges → churn).",
        "Linear models assign independent weight to each feature and may miss interaction effects.",
        "",
        "## Examples of Disagreement",
        "",
    ]

    if n_disagree > 0:
        disagree_df = X_test[disagree_mask].copy()
        disagree_df["tree_proba"] = tree_proba[disagree_mask]
        disagree_df["linear_proba"] = linear_proba[disagree_mask]
        disagree_df["tree_pred"] = tree_pred[disagree_mask]
        disagree_df["linear_pred"] = linear_pred[disagree_mask]
        cols_show = ["tree_proba", "linear_proba", "tree_pred", "linear_pred"] + NUMERIC_FEATURES[:5]
        lines.append(disagree_df[cols_show].head(10).to_string())
    else:
        lines.append("No disagreements at this threshold.")

    (output_dir / "tree_vs_linear_disagreement.md").write_text("\n".join(lines))
    logger.info("Saved tree-vs-linear disagreement → %s", output_dir / "tree_vs_linear_disagreement.md")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_pr_curves(pipes: dict, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> None:
    """Plot PR curves for top (non-Dummy) models."""
    top_models = {k: v for k, v in pipes.items() if k != "Dummy"}
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, pipe in top_models.items():
        y_proba = pipe.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    fig.tight_layout()
    pr_path = output_dir / "pr_curves.png"
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    logger.info("Saved PR curves → %s", pr_path)


def plot_threshold_sweep(sweep_df: pd.DataFrame, threshold: float, output_dir: Path) -> None:
    """Plot threshold sweep with precision, recall, F1, and alerts per 1000."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", color="#4c72b0")
    ax1.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", color="#55a868")
    ax1.plot(sweep_df["threshold"], sweep_df["f1"], label="F1", color="#c44e52")
    ax1.axvline(threshold, color="black", linestyle="--", alpha=0.7, label=f"Selected ({threshold})")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Score")
    ax1.legend(loc="upper left")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(sweep_df["threshold"], sweep_df["alerts_per_1000"], label="Alerts/1000", color="#8172b2", linestyle=":")
    ax2.set_ylabel("Alerts per 1,000")
    ax2.legend(loc="upper right")

    fig.suptitle("Threshold Sweep — Out-of-Fold Predictions")
    fig.tight_layout()
    sweep_path = output_dir / "threshold_sweep.png"
    fig.savefig(sweep_path, dpi=150)
    plt.close(fig)
    logger.info("Saved threshold sweep plot → %s", sweep_path)


# ---------------------------------------------------------------------------
# Saving artifacts
# ---------------------------------------------------------------------------


def save_comparison_table(results: dict, output_dir: Path) -> None:
    """Save cross-validated metrics as comparison_table.csv."""
    rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        for metric, vals in metrics.items():
            row[f"{metric}_mean"] = vals["mean"]
            row[f"{metric}_std"] = vals["std"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "comparison_table.csv", index=False)
    logger.info("Saved comparison table → %s", output_dir / "comparison_table.csv")


def save_experiment_log(results: dict, best_model: str, output_dir: Path, random_seed: int, n_folds: int) -> None:
    """Save a flat experiment log."""
    rows = []
    ts = datetime.now().isoformat()
    for model_name, metrics in results.items():
        row = {"timestamp": ts, "model": model_name, "random_seed": random_seed, "n_folds": n_folds}
        for metric, vals in metrics.items():
            row[f"{metric}_mean"] = vals["mean"]
            row[f"{metric}_std"] = vals["std"]
        row["is_best"] = model_name == best_model
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "experiment_log.csv", index=False)
    logger.info("Saved experiment log → %s", output_dir / "experiment_log.csv")


def save_threshold_recommendation(
    threshold: float,
    strategy: str,
    sweep_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save threshold_recommendation.md with selected and alternative thresholds."""
    # Find alternatives
    alt_thresholds = []
    # best_f1 candidate
    best_f1_idx = sweep_df["f1"].idxmax()
    alt_thresholds.append(("best_f1", round(float(sweep_df.loc[best_f1_idx, "threshold"]), 4)))
    # 0.50 default
    if 0.50 in sweep_df["threshold"].values:
        row_50 = sweep_df[sweep_df["threshold"] == 0.50].iloc[0]
        alt_thresholds.append(("default_0.50", 0.50))
    # high precision
    high_prec_idx = sweep_df[sweep_df["precision"] >= 0.6]["f1"].idxmax() if (sweep_df["precision"] >= 0.6).any() else best_f1_idx
    alt_thresholds.append(("high_precision", round(float(sweep_df.loc[high_prec_idx, "threshold"]), 4)))

    lines = [
        "# Threshold Recommendation",
        "",
        f"**Selected threshold:** {threshold}",
        f"**Strategy:** {strategy}",
        "",
        "## Selected Operating Point",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    sel_row = sweep_df.loc[sweep_df["threshold"].sub(threshold).abs().idxmin()]
    for col in ["threshold", "precision", "recall", "f1", "alert_rate", "alerts_per_1000"]:
        lines.append(f"| {col} | {sel_row[col]} |")

    lines.extend([
        "",
        "## Alternative Thresholds",
        "",
        "| Strategy | Threshold |",
        "|----------|-----------|",
    ])
    for strat, thr in alt_thresholds:
        lines.append(f"| {strat} | {thr} |")

    (output_dir / "threshold_recommendation.md").write_text("\n".join(lines))
    logger.info("Saved threshold recommendation → %s", output_dir / "threshold_recommendation.md")


def save_run_metadata(
    config: dict,
    best_model: str,
    threshold: float,
    operating_metrics: dict,
    random_seed: int,
    n_folds: int,
    output_dir: Path,
) -> None:
    """Save run_metadata.json."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "random_seed": random_seed,
        "n_folds": n_folds,
        "config": config,
        "selected_model": best_model,
        "selected_threshold": threshold,
        "operating_metrics": operating_metrics,
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved run metadata → %s", output_dir / "run_metadata.json")


def save_snapshot(output_dir: Path) -> None:
    """Copy major outputs to a timestamped snapshot directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_dir = output_dir / "runs" / ts
    snap_dir.mkdir(parents=True, exist_ok=True)

    key_files = [
        "comparison_table.csv", "pr_curves.png", "calibration.png",
        "best_model.joblib", "experiment_log.csv",
        "threshold_tuning.csv", "threshold_sweep.png",
        "threshold_recommendation.md", "operating_metrics.json",
        "run_metadata.json",
    ]
    for fname in key_files:
        src = output_dir / fname
        if src.exists():
            import shutil
            shutil.copy2(src, snap_dir / fname)

    logger.info("Saved run snapshot → %s", snap_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_config(config_path: str | None) -> dict:
    """Load optional JSON config; CLI args override."""
    config = dict(DEFAULT_CONFIG)
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                file_config = json.load(f)
            config.update(file_config)
            logger.info("Loaded config from %s", path.resolve())
        else:
            logger.warning("Config file not found: %s — using defaults", path)
    return config


def parse_args(argv: list | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Petra Telecom Churn Model Comparison CLI — compare models, "
                    "select thresholds, and generate evaluation artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", default="data/telecom_churn.csv", help="Path to the input CSV dataset")
    parser.add_argument("--output-dir", default="./output", help="Directory where all artifacts will be saved")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of stratified cross-validation folds")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true", help="Validate the dataset and configuration without training")
    parser.add_argument("--config-path", default="config.json", help="Optional path to a JSON config file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    return parser.parse_args(argv)


def main(argv: list | None = None) -> None:
    """Entry point for the model comparison pipeline."""
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config_path if not args.dry_run else None)

    logger.info("Pipeline configuration:")
    logger.info("  data-path   : %s", args.data_path)
    logger.info("  output-dir  : %s", args.output_dir)
    logger.info("  n-folds     : %d", args.n_folds)
    logger.info("  random-seed : %d", args.random_seed)
    logger.info("  dry-run     : %s", args.dry_run)
    logger.info("  log-level   : %s", args.log_level)
    logger.info("  config-path : %s", args.config_path)

    df = load_data(args.data_path)
    validate_data(df, args.n_folds)

    if args.dry_run:
        models = _build_models(args.random_seed)
        logger.info("— DRY RUN — No models will be trained")
        logger.info("Models that would be compared: %s", list(models.keys()))
        logger.info("Selection metric: %s", config["selection_metric"])
        logger.info("Threshold strategy: %s", config["threshold_selection_strategy"])
        logger.info("Scoring metrics: %s", SCORING_METRICS)
        logger.info("CV strategy: StratifiedKFold (%d folds, seed=%d)", args.n_folds, args.random_seed)
        logger.info("Output directory: %s", Path(args.output_dir).resolve())
        logger.info("Dry run complete — configuration is valid")
        sys.exit(0)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Data quality report ---
    generate_data_quality_report(df, output_dir)

    # --- Train / test split ---
    X = df.drop(columns=[TARGET_COL] + DROP_COLS)
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_seed, stratify=y,
    )
    logger.info("Train/test split: %d train, %d test", len(X_train), len(X_test))

    # --- Cross-validation model comparison ---
    models = _build_models(args.random_seed)
    results, oof_probas = cross_validate_models(X_train, y_train, models, args.n_folds, args.random_seed)

    # --- Model selection ---
    best_model_name = select_best_model(results, config["selection_metric"])

    # --- Threshold selection from OOF predictions ---
    threshold, sweep_df = select_threshold(
        oof_probas[best_model_name], y_train,
        strategy=config["threshold_selection_strategy"],
        recall_target=config.get("recall_target", 0.8),
        max_alert_rate=config.get("max_alert_rate"),
    )
    sweep_df.to_csv(output_dir / "threshold_tuning.csv", index=False)
    logger.info("Saved threshold tuning → %s", output_dir / "threshold_tuning.csv")
    plot_threshold_sweep(sweep_df, threshold, output_dir)

    # --- Fit best model on full training set ---
    best_pipe = _build_pipeline(best_model_name, copy.deepcopy(models[best_model_name]), args.random_seed)
    best_pipe.fit(X_train, y_train)
    joblib.dump(best_pipe, output_dir / "best_model.joblib")
    logger.info("Saved best model → %s", output_dir / "best_model.joblib")

    # --- Fit all models on full training set for diagnostics ---
    all_pipes = {}
    for name, model in models.items():
        pipe = _build_pipeline(name, copy.deepcopy(model), args.random_seed)
        pipe.fit(X_train, y_train)
        all_pipes[name] = pipe

    # --- Test-set evaluation ---
    operating_metrics, test_preds = evaluate_on_test(best_pipe, X_test, y_test, threshold)
    with open(output_dir / "operating_metrics.json", "w") as f:
        json.dump(operating_metrics, f, indent=2)
    logger.info("Saved operating metrics → %s", output_dir / "operating_metrics.json")
    test_preds.to_csv(output_dir / "test_predictions.csv", index=False)
    logger.info("Saved test predictions → %s", output_dir / "test_predictions.csv")

    # --- Core artifacts ---
    save_comparison_table(results, output_dir)
    save_experiment_log(results, best_model_name, output_dir, args.random_seed, args.n_folds)
    plot_pr_curves(all_pipes, X_test, y_test, output_dir)

    # --- Calibration analysis ---
    calibration_analysis(all_pipes, X_test, y_test, config.get("calibration_bins", 10), output_dir)

    # --- Error analysis ---
    error_analysis(best_pipe, X_test, y_test, threshold, config.get("error_analysis_top_n", 10), output_dir)

    # --- Tree vs linear disagreement ---
    tree_vs_linear_disagreement(all_pipes, X_test, threshold, output_dir)

    # --- Threshold recommendation ---
    save_threshold_recommendation(threshold, config["threshold_selection_strategy"], sweep_df, output_dir)

    # --- Run metadata ---
    save_run_metadata(config, best_model_name, threshold, operating_metrics, args.random_seed, args.n_folds, output_dir)

    # --- Snapshot ---
    if config.get("run_snapshot", True):
        save_snapshot(output_dir)

    logger.info("Pipeline finished successfully — best model: %s, threshold: %.4f", best_model_name, threshold)


if __name__ == "__main__":
    main()
