# WHO COVID-19 Global Data — Reproducible Pipelines & Tabular ML

**Author:** Muhammad Huzaifa | **Roll Number:** TRN-2026-001512  
**Assignment:** Reproducible Data Pipelines, Tabular ML & Systems Exploration  
**Date:** November 20, 2026  

---

## Dataset Overview

| Attribute | Detail |
|---|---|
| Name | WHO COVID-19 Global Weekly Surveillance Data |
| Source URL | https://covid19.who.int/data |
| License | Creative Commons Attribution-NonCommercial-ShareAlike 3.0 IGO (CC BY-NC-SA 3.0 IGO) |
| Access Date | November 2026 |
| Total Records | 85,680 weekly surveillance rows |
| Total Columns | 8 attributes |
| Date Range | 2020-01-05 to 2026-11-01 (6 full epidemiological years) |

---

## Repository Layout

```
pseb_assignment_2/
├── README.md                           Project overview and setup instructions
├── requirements.txt                    Pinned Python package dependencies
├── predict.py                          Part B bonus: CLI inference script
├── data/
│   ├── raw/
│   │   └── WHO-COVID-19-global-data.csv   Original raw data (never modified)
│   ├── cleaned/
│   │   └── cleaned_covid_data.csv         Pipeline cleaned output
│   ├── sample_unseen.csv               Unseen test rows (input run 1)
│   └── sample_unseen_2.csv             Unseen test rows (input run 2)
├── src/
│   ├── __init__.py
│   ├── cleaning.py                     Modular cleaning & outlier functions (with docstrings)
│   ├── validation.py                   Row-level validation checks & quality gate (with docstrings)
│   ├── features.py                     Temporal feature extraction & leakage exclusion
│   ├── pipeline.py                     Part A: end-to-end data processing & SQLite pipeline
│   └── train.py                        Part B: TimeSeriesSplit CV, model training & evaluation
├── tests/
│   └── test_pipeline.py                7 pytest unit tests for cleaning & validation
├── outputs/
│   ├── figures/
│   │   ├── figure1_cases_over_time_by_region.png
│   │   ├── figure2_top10_countries_cumulative_cases.png
│   │   ├── figure3_cases_vs_deaths_scatter.png
│   │   └── figure4_residual_plot.png
│   ├── reports/
│   │   ├── metrics.json                Row-level quality gate metrics
│   │   ├── data_quality_report.txt     Before vs after cleaning summary
│   │   ├── regional_summary.csv        Pandas groupby regional table
│   │   ├── monthly_summary.csv         Pandas pivot table by month & region
│   │   ├── sqlite_query_results.csv    SQLite JOIN query output
│   │   ├── model_comparison.csv        5-fold TimeSeriesSplit cross-validation table
│   │   └── error_analysis.txt          Detailed error breakdown & R2 discussion
│   ├── predictions.csv                 Predictions on sample_unseen.csv
│   └── predictions_2.csv               Predictions on sample_unseen_2.csv
├── models/
│   └── covid_death_predictor.joblib    Fitted Random Forest Pipeline object
├── sql/
│   └── regional_summary.sql            SQLite analytical join query
├── notebooks/
│   └── exploration.ipynb               Exploratory data analysis notebook
└── docs/
    ├── schema.md                       Raw data schema, types, allowed ranges & nulls
    └── assumptions.md                  Data assumptions, limitations, and ML model card
```

---

## Setup & Execution (Fresh Clone Run)

Anyone cloning this repository can run the entire project directly with these steps:

```bash
# 1. Clone repository
git clone https://github.com/MuhammadHuzaifa65/pseb_assignment_2.git
cd pseb_assignment_2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run tests (All 7 passed)
python -m pytest tests/ -v

# 4. Run Part A: Data Pipeline (Cleaned data, SQLite join, charts, reports)
python src/pipeline.py

# 5. Run Part B: Tabular ML (Train models, evaluate, error analysis)
python src/train.py

# 6. Run Inference on unseen records
python predict.py --input data/sample_unseen.csv --output outputs/predictions.csv
```

---

## Part A — Data Quality & Insights Pipeline

### Cleaning & Quality Gate
- **Row-Level Quality Gate:** Checks that every record simultaneously satisfies date validity, non-negative counts, populated categorical fields, and key uniqueness. On the cleaned data, **100.0% of rows pass validation** (threshold $\ge 95\%$).
- **Missing Value Imputation:** 32,389 blank `New_cases` and 47,799 blank `New_deaths` imputed with `0.0` (zero cases/deaths reported that week).
- **Outlier Handling:** Retrospective negative administrative revisions (18 negative cases, 13 negative deaths) clipped to `0.0` via `np.maximum`.
- **Category Standardization:** Overseas territories with blank regions labeled `'OTHER'`; Namibia's ISO code `'NA'` preserved using `keep_default_na=False`.
- **SQLite Join:** Cleaned table joined with `who_regions` dimension table via `sql/regional_summary.sql` (Europe leads in total cases at 281M; Americas leads in total deaths at 3.07M; Africa has highest case fatality ratio at 1.93%).

### Visualizations
1. `outputs/figures/figure1_cases_over_time_by_region.png`: Monthly trend of new infections across all 6 WHO regions (shows Delta and Omicron wave dynamics).
2. `outputs/figures/figure2_top10_countries_cumulative_cases.png`: Horizontal bar chart of top 10 nations by total cumulative cases (USA, China, India, France, Germany).
3. `outputs/figures/figure3_cases_vs_deaths_scatter.png`: Log-transformed scatter plot illustrating positive correlation between weekly cases and deaths.

---

## Part B — Tabular ML Baseline & Evaluation

### Problem Framing & Leakage-Free Design
- **Target:** `New_deaths` (weekly newly confirmed COVID-19 deaths) — Continuous Regression.
- **Leakage Prevention:** `Cumulative_deaths`, `Mortality_rate_percent`, and `Log_new_deaths` are strictly excluded from features because cumulative deaths directly contain the target.
- **Split Justification:** Time-based split at `2023-01-01` (Training: 2020–2022, 37,440 rows; Test: 2023–2026, 48,240 rows). Prevents lookahead bias in chronological epidemic data.
- **Cross-Validation:** 5-Fold `TimeSeriesSplit` on training data, evaluating models sequentially on forward-expanding temporal windows.

### Model Comparison Table

| Model | CV MAE | CV RMSE | CV $R^2$ | Test MAE | Test RMSE | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Mean Dummy) | 277.62 | 1,029.15 | -0.1970 | 180.77 | 614.99 | -0.4379 |
| Linear Regression | 258.97 | 1,010.51 | -0.1580 | 305.12 | 1,595.68 | -8.6248 |
| Ridge Regression ($L_2$) | 258.91 | 1,010.45 | -0.1578 | 305.07 | 1,595.40 | -8.6215 |
| **Random Forest Regressor** | **150.15** | **835.62** | **0.2078** | **19.61** | **483.47** | **0.1092** |

### Honest Discussion of $R^2$ Metrics & Distribution Shift

1. **Why Linear Regression & Ridge Produced Negative Test $R^2$ (-8.62):**
   - $R^2 = 1 - \frac{\text{MSE}}{\text{Var}(y)}$. When a model's squared error exceeds the target's variance, $R^2$ becomes negative, meaning the model performs worse than predicting the test mean.
   - **Epidemiological Shift:** In 2020–2022, high cases meant high mortality; linear models learned steep positive slopes. In 2023–2026, widespread vaccination and Omicron decoupled cases from deaths. Unconstrained linear slopes predicted thousands of deaths during high-case weeks where very few actually died, generating massive squared error penalties.
   - Regularization in Ridge ($L_2, \alpha=10.0$) shrank weights slightly but could not adapt to this fundamental structural break.

2. **Why Random Forest Achieved an Exceptional MAE (19.61) but Modest $R^2$ (~0.11):**
   - **15x Better MAE:** Random Forest achieved a Test MAE of 19.61 deaths/week because decision trees bound predictions within leaf intervals, preventing runaway astronomical errors.
   - **Why $R^2$ is Modest:** $R^2$ measures proportion of variance explained. The post-2023 mortality signal is dominated by national backlog batch dumps and reporting anomalies rather than same-week infection volume. Without post-2023 retraining or dynamic 2–3 week lag features, a static pre-2023 model cannot capture the remaining variance.

---

## Part B Bonus — Command-Line Prediction Script

The standalone `predict.py` script loads the saved pipeline, validates input schema, outputs predictions to CSV, and explains feature importance:

```bash
# Test on sample 1
python predict.py --input data/sample_unseen.csv --output outputs/predictions.csv

# Test on sample 2
python predict.py --input data/sample_unseen_2.csv --output outputs/predictions_2.csv
```

**Feature Importance Insight:** `New_cases` accounted for **59.6%** of split decisions, followed by `Cumulative_cases` (15.2%) and `WHO_region_AMRO` (13.2%), which makes complete clinical sense as infection volume is the primary driver of subsequent mortality.

---
