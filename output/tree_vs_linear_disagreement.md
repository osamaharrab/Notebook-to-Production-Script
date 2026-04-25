# Tree vs Linear Model Disagreement

Comparing **DT_depth5** (tree) vs **LR_default** (linear) at threshold 0.25.

Total disagreements: 131 / 900 (14.6%)

## Why They Disagree

Tree models capture non-linear interactions (e.g., short tenure + high charges → churn).
Linear models assign independent weight to each feature and may miss interaction effects.

## Examples of Disagreement

      tree_proba  linear_proba  tree_pred  linear_pred  senior_citizen  tenure  monthly_charges  total_charges  num_support_calls
394     0.027842      0.371166          0            1               0      10            43.58           0.00                  3
1213    0.268562      0.103571          1            0               0      66            94.17        9243.23                  2
4003    0.304636      0.166789          1            0               0       1            56.02         150.66                  2
3741    0.392857      0.037723          1            0               0      12            24.20         263.58                  0
695     0.214286      0.368184          0            1               0      60            41.06        2128.96                  4
3909    0.304636      0.108340          1            0               1       5            68.07           0.00                  2
4453    0.268562      0.133288          1            0               0      54            48.44        2494.55                  2
4417    0.190476      0.485988          0            1               0       8           102.43         719.28                  4
609     0.268562      0.248388          1            0               0      21            74.48         225.64                  2
3444    0.268562      0.179396          1            0               0      72            42.21        3514.22                  3