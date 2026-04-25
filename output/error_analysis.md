# Error Analysis (threshold=0.25)

## Top 10 Highest-Confidence False Positives

These customers were predicted to churn but did not.

      y_proba  y_true  y_pred  senior_citizen  tenure  monthly_charges  total_charges  num_support_calls
3799     0.80       0       1               0       0            72.61          20.41                  5
2850     0.75       0       1               0       8            46.19          22.29                  4
3700     0.72       0       1               0       2            32.06           0.00                  4
1823     0.71       0       1               0      72            68.93           0.00                  4
2837     0.70       0       1               0       7           112.79        1075.34                  2
1566     0.70       0       1               0      28            20.00        1485.27                  3
3498     0.65       0       1               1      43            20.36        1129.31                  3
1225     0.57       0       1               0       1            20.00           0.00                  1
2572     0.57       0       1               0       4           122.15         547.56                  2
1948     0.56       0       1               0       5            61.35         197.27                  4

## Top 10 Lowest-Confidence False Negatives

These customers churned but were predicted not to.

      y_proba  y_true  y_pred  senior_citizen  tenure  monthly_charges  total_charges  num_support_calls
202      0.01       1       0               0      59            53.30        1823.88                  1
2332     0.01       1       0               0       6            67.19         371.08                  0
1803     0.02       1       0               0      13            47.00         209.26                  1
1414     0.02       1       0               0      20            94.32        2914.10                  2
4076     0.02       1       0               0      13            71.46         804.55                  2
1659     0.03       1       0               0      37            61.51         916.37                  2
4315     0.03       1       0               0       5            32.65         114.88                  1
3633     0.04       1       0               1      22            97.44        3480.51                  1
3394     0.05       1       0               0       5            68.03         283.70                  3
1391     0.06       1       0               0       9            52.13         317.54                  0