# Synthetic to Real-Target 500-Cap Summary

Protocol: full synthetic Scenario 1 pretraining, then 250 synthetic + 250 real Scenario 1 replay rows, plus the listed number of real target rows. Results are averaged over 3 seeds.

| Target | Model | 0 target samples (%) | 200 target samples (%) | Gain (percentage points) |
|---|---|---:|---:|---:|
| Scenario 2: McAllister night | MLP | 57.82 | 64.20 | +6.39 |
| Scenario 2: McAllister night | Random Forest | 36.25 | 79.66 | +43.42 |
| Scenario 2: McAllister night | k-NN | 35.41 | 78.21 | +42.80 |
| Scenario 2: McAllister night | Fourier + k-NN | 34.23 | 77.03 | +42.80 |
| Scenario 3: Rural Road day | MLP | 5.59 | 22.93 | +17.34 |
| Scenario 3: Rural Road day | Random Forest | 0.78 | 72.26 | +71.48 |
| Scenario 3: Rural Road day | k-NN | 8.39 | 66.22 | +57.83 |
| Scenario 3: Rural Road day | Fourier + k-NN | 10.40 | 63.20 | +52.80 |
| Scenario 7: wide 4-lane site day | MLP | 1.36 | 18.71 | +17.35 |
| Scenario 7: wide 4-lane site day | Random Forest | 1.36 | 74.46 | +73.10 |
| Scenario 7: wide 4-lane site day | k-NN | 8.38 | 66.08 | +57.70 |
| Scenario 7: wide 4-lane site day | Fourier + k-NN | 19.69 | 65.89 | +46.20 |
| Scenario 33straight: College Avenue night, straight segment | MLP | 0.19 | 4.92 | +4.72 |
| Scenario 33straight: College Avenue night, straight segment | Random Forest | 0.71 | 90.03 | +89.32 |
| Scenario 33straight: College Avenue night, straight segment | k-NN | 0.71 | 84.85 | +84.14 |
| Scenario 33straight: College Avenue night, straight segment | Fourier + k-NN | 6.21 | 84.92 | +78.71 |
| Scenario 33: College Avenue night, full road | MLP | 11.50 | 29.69 | +18.19 |
| Scenario 33: College Avenue night, full road | Random Forest | 6.57 | 85.61 | +79.05 |
| Scenario 33: College Avenue night, full road | k-NN | 6.57 | 79.63 | +73.07 |
| Scenario 33: College Avenue night, full road | Fourier + k-NN | 5.69 | 78.00 | +72.31 |
