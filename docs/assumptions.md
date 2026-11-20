# Assumptions, Limitations, Risks and Model Card

**Author:** Muhammad Huzaifa | **Roll Number:** TRN-2026-001512 | **Date:** November 20, 2026

---

## Part A — Data Cleaning & Validation Assumptions

| Assumption | Decision | Reason |
|---|---|---|
| Blank `New_cases` / `New_deaths` = zero activity | Imputed with `0.0` | WHO surveillance notes confirm blanks represent zero reported infections/deaths for that weekly window |
| Negative counts = retrospective adjustments | Clipped to `0.0` with `np.maximum` | Negative numbers reflect retrospective administrative deductions by health ministries; mortality cannot physically be negative |
| Blank `WHO_region` = overseas territory | Labeled `'OTHER'` | These are valid autonomous territories (e.g., Gibraltar, Falkland Islands) rather than measurement errors |
| Namibia country code `NA` | Loaded with `keep_default_na=False` | Preserves Namibia's valid ISO 3166-1 alpha-2 code `NA` from being parsed as NaN |
| Uniqueness constraint | Enforced on `(Date_reported, Country)` | Each nation reports exactly one surveillance log per 7-day period |

### Row-Level Quality Gate Strategy
The quality gate evaluates validation at the **row level**:
- An individual record passes if its reporting date is valid, all surveillance metrics (`New_cases`, `New_deaths`, `Cumulative_cases`, `Cumulative_deaths`) are non-null and non-negative, categorical fields are populated, and the composite key is unique.
- The pipeline requires at least **95%** of all rows to pass simultaneously before proceeding to analytics and reporting. On the cleaned WHO dataset, the row validation pass rate is **100.0%**.

---

## Part B — Model Card

### Prediction Problem Definition
- **Target:** `New_deaths` — count of newly confirmed COVID-19 deaths reported during a 7-day surveillance week per country.
- **Problem Type:** Tabular Regression.
- **Intended Users:** Public health epidemiologists, resource coordinators, and hospital administrative teams estimating weekly critical-care resource requirements.

### Features Included
- `New_cases`: Number of newly confirmed infections in the reporting week (primary disease volume driver).
- `Cumulative_cases`: Cumulative confirmed cases to date (surrogate for total population infection history).
- `Month`: Calendar month (1–12), capturing seasonal respiratory transmission waves.
- `Day_of_week`: Day of the week for the report date (captures weekly reporting schedule variance).
- `WHO_region`: Geographic region (AFRO, AMRO, EMRO, EURO, SEARO, WPRO, OTHER).

### Target Leakage Prevention
To ensure strict leakage-free modeling, the following columns were explicitly identified and excluded from the feature matrix:
1. `Cumulative_deaths`: Directly contains the target variable: Cumulative_deaths(t) = Cumulative_deaths(t-1) + New_deaths(t). Including it would allow trivial reconstruction of the target.
2. `Mortality_rate_percent`: Calculated using Cumulative_deaths / Cumulative_cases, encoding future mortality knowledge.
3. `Log_new_deaths`: Direct mathematical transformation of the target.

### Split Justification
- **Method:** Time-based train/test split at cutoff `2023-01-01`.
- **Training Set:** 2020-01-05 to 2022-12-25 (37,440 historical records).
- **Test Set:** 2023-01-01 to 2026-11-01 (48,240 future records).
- **Cross-Validation:** 5-Fold `TimeSeriesSplit` on training data.
- **Why Time-Based:** Outbreak surveillance is non-stationary and strictly chronological. A random split allows future data to inform past predictions (temporal lookahead leakage), producing falsely optimistic performance estimates. A forward-facing temporal split genuinely tests how well historical models forecast future waves.

---

## Honest Analysis of Model Comparison & R2 Metrics

### Summary Results Table

| Model | CV MAE | CV RMSE | CV R2 | Test MAE | Test RMSE | Test R2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Mean Dummy) | 277.62 | 1,029.15 | -0.1970 | 180.77 | 614.99 | -0.4379 |
| Linear Regression | 258.97 | 1,010.51 | -0.1580 | 305.12 | 1,595.68 | -8.6248 |
| Ridge Regression (L2) | 258.91 | 1,010.45 | -0.1578 | 305.07 | 1,595.40 | -8.6215 |
| **Random Forest Regressor** | **150.15** | **835.62** | **0.2078** | **19.61** | **483.47** | **0.1092** |

### Critical Discussion of Negative R2 and Low Forest R2

1. **Why Linear Regression and Ridge Produced Negative R2 on the Test Set:**
   - In scikit-learn, the coefficient of determination is defined as:
     R2 = 1 - (MSE / Var(y))
     Whenever a model's mean squared error exceeds the variance of the target in the test set, R2 becomes negative. A negative R2 indicates the model's predictions are quantitatively worse than simply predicting the test set's mean value.
   - **Epidemiological Cause (The 2020-2022 vs 2023-2026 Decoupling):**
     During the 2020-2022 training period, population immunity was low, clinical severity was high, and positive cases reliably dictated hospitalizations and mortality. Linear Regression learned steep positive slope coefficients for New_cases.
     In the 2023-2026 test period, widespread vaccination and the global dominance of milder Omicron sub-variants fundamentally altered the virus's Case Fatality Rate (CFR). Major waves generated tens of thousands of weekly cases with very few deaths.
     When the linear model encountered high case numbers in 2023+, its large unconstrained coefficients predicted astronomical mortality counts (thousands of deaths) that never occurred. These massive squared residual penalties drove the test R2 down to **-8.62**.
   - **Why Ridge Did Not Prevent This:** Ridge adds an L2 weight penalty (alpha = 10.0), slightly shrinking coefficient magnitudes, but it remains a linear model. Regularization alone cannot adapt to a structural breakdown in the underlying physical relationship between features and target.

2. **Why Random Forest Achieved a Low R2 (~0.11) Despite Exceptional MAE (19.61):**
   - **Vastly Superior MAE:** Random Forest achieved a Test MAE of **19.61 deaths/week** — more than 15 times better than the linear models and 9 times better than the dummy baseline.
   - **Why MAE is Strong:** Because decision trees partition feature space into bounded leaf segments, Random Forest never extrapolates unconstrained linear trends. When case counts surged in 2023, predictions remained capped within historical leaf bounds, avoiding catastrophic blowups.
   - **Why R2 Remains Modest (~0.11):** R2 measures the proportion of total target variance explained. The post-2023 mortality signal is dominated by nation-specific retrospective backlog releases, hospital reporting gaps, and localized ICU capacity rather than raw infection totals. Without retraining on post-2023 surveillance data or adding dynamic lag features (deaths lag cases by 14-21 days) and vaccination status, no model trained strictly on 2020-2022 data can explain the remaining variance.

---

## Limitations & Intended Use Boundaries

### Identified Limitations
1. **Lack of Dynamic Lag Variables:** Biologically, death occurs weeks after initial diagnosis. A static snapshot model cannot capture the 2- to 3-week lag between surge peaks and mortality peaks.
2. **Administrative Backlog Noise:** Several countries accumulated death records over multiple months and dumped thousands of backlog cases into single weekly surveillance reports. These cannot be predicted from same-week case counts alone.
3. **Under-Reporting Disparity:** Surveillance testing capacity varies widely between high-income and low-income nations, introducing systemic geographical noise.

### Intended Use Boundaries
- **Intended Use:** High-level exploratory analysis, teaching tabular regression pipelines, and broad regional planning estimates.
- **Prohibited Use:** Real-time clinical triage, patient-level prognostic forecasting, or binding legal policy without formal epidemiological review.
