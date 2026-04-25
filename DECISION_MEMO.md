# Decision Memo — Petra Telecom Churn Model Deployment

## Recommendation

**Deploy RF_default with a 0.25 operating threshold.**

## Model Selection

Six models were compared using stratified 5-fold cross-validation on the training set. PR-AUC was used as the selection metric because the churn dataset is imbalanced (approximately 84% non-churn, 16% churn), making accuracy an unreliable ranking criterion.

| Model        | PR-AUC (mean) | F1 (mean) | Recall (mean) |
|--------------|---------------|-----------|---------------|
| RF_default   | 0.4988        | 0.3709    | 0.2546        |
| RF_balanced  | 0.4841        | 0.3136    | 0.2020        |
| LR_default   | 0.4633        | 0.2988    | 0.1935        |
| LR_balanced  | 0.4617        | 0.4376    | 0.6944        |
| DT_depth5    | 0.4608        | 0.3804    | 0.2699        |
| Dummy        | 0.1636        | 0.0000    | 0.0000        |

RF_default was selected because it achieves the highest PR-AUC, meaning it ranks churners best across all thresholds. The recall gap at the default 0.50 threshold is addressed by threshold selection.

## Threshold Selection

The default 0.50 threshold is inappropriate for churn intervention because it produces very low recall — most actual churners are missed.

Using the `best_f1` strategy on out-of-fold training predictions, a threshold of **0.25** was selected. This shifts the operating point toward higher recall while accepting more false positives.

### Comparison of operating points on the test set

| Metric             | Threshold 0.25 | Threshold 0.50 |
|--------------------|-----------------|-----------------|
| Accuracy           | 0.7656          | 0.8456          |
| Precision          | 0.3609          | 0.5833          |
| Recall             | 0.5646          | 0.1905          |
| F1                 | 0.4403          | 0.2872          |
| Predicted positives| 230 / 900       | 48 / 900        |
| Alert rate         | 25.6%           | 5.3%            |

## Business Rationale

For churn intervention, the cost of a false positive (an unnecessary retention offer) is typically much lower than the cost of a false negative (a churned customer who was not offered retention). A recall-oriented threshold is therefore appropriate.

At threshold 0.50, the model only catches ~19% of churners (28 out of 147). At threshold 0.25, it catches ~56% (83 out of 147), with a manageable alert rate of ~26%.

If outreach capacity is limited, the threshold can be raised or the `max_alert_rate` strategy can be used in `config.json` to cap the alert rate.

## Data Quality Caveat

The data quality report identified 278 rows where `tenure > 3` but `total_charges == 0`. These may represent data-entry issues or business edge cases. This should be investigated before production deployment but does not invalidate the current model comparison.

## Calibration

Calibration analysis shows the selected model's predicted probabilities are reasonably well-calibrated, though some over-confidence is typical for tree ensembles. If well-calibrated probabilities are needed for cost modeling, consider applying Platt scaling or isotonic regression as a post-processing step.

## Next Steps

1. Validate the threshold with business stakeholders against outreach budget
2. Investigate the `tenure > 3, total_charges == 0` data pattern
3. Monitor model drift and retrain on updated data periodically
4. Consider calibration post-processing if probability accuracy matters for cost analysis
