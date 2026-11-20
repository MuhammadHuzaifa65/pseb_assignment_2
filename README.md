# WHO COVID-19 Global Data — Reproducible Pipelines & Tabular ML

**Author:** Muhammad Huzaifa | **Roll Number:** TRN-2026-001512
**Assignment:** Reproducible Data Pipelines, Tabular ML & Systems Exploration
**Date:** November 20, 2026

---

## Dataset

| Attribute | Detail |
|---|---|
| Name | WHO COVID-19 Global Weekly Surveillance Data |
| Source URL | https://covid19.who.int/data |
| License | Creative Commons Attribution-NonCommercial-ShareAlike 3.0 IGO |
| Access Date | November 2026 |
| Rows | 85,680 |
| Columns | 8 |
| Date Range | 2020-01-05 to 2026-11-01 |

---

## Repository Layout

```
assignment_2/
├── README.md
├── requirements.txt
├── predict.py                          Part B bonus: CLI inference script
├── data/
│   ├── raw/WHO-COVID-19-global-data.csv   original file, never edited
│   ├── cleaned/cleaned_covid_data.csv     pipeline output
│   └── sample_unseen.csv                  5 new rows for inference demo
├── src/
│   ├── cleaning.py                     type fixes, imputation, dedup, outliers
│   ├── validation.py                   range/type/null/uniqueness checks + quality gate
│   ├── features.py                     feature extraction and leakage-safe split
│   ├── pipeline.py                     Part A: data quality and insights pipeline
│   └── train.py                        Part B: model training and evaluation
├── tests/
│   └── test_pipeline.py                7 pytest unit tests
├── outputs/
│   ├── figures/
│   │   ├── figure1_cases_over_time_by_region.png
│   │   ├── figure2_top10_countries_cumulative_cases.png
│   │   ├── figure3_cases_vs_deaths_scatter.png
│   │   └── figure4_residual_plot.png
│   └── reports/
│       ├── metrics.json
│       ├── data_quality_report.txt
│       ├── regional_summary.csv
│       ├── monthly_summary.csv
│       ├── sqlite_query_results.csv
│       ├── model_comparison.csv
│       └── error_analysis.txt
├── models/
│   └── covid_death_predictor.joblib
├── sql/
│   └── regional_summary.sql
└── docs/
    ├── schema.md
    └── assumptions.md
```

---

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd assignment_2

# 2. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

```bash
# Part A — Data Quality & Insights Pipeline
python src/pipeline.py

# Part B — ML Training & Evaluation
python src/train.py

# Run all tests
pytest tests/ -v

# Part B Bonus — Predict on new unseen rows
python predict.py --input data/sample_unseen.csv --output outputs/predictions.csv
```

Each command is independent and can be run from the repository root.

---

## Part A — Key Findings

**Data quality after cleaning (85,680 rows):**

| Issue | Raw | After Cleaning |
|---|---|---|
| Missing `New_cases` | 32,389 (37.8%) | 0 — imputed with 0.0 |
| Missing `New_deaths` | 47,799 (55.8%) | 0 — imputed with 0.0 |
| Blank `WHO_region` | 6,426 (7.5%) | 0 — labeled OTHER |
| Negative case revisions | 18 | 0 — clipped to 0 |
| Negative death revisions | 13 | 0 — clipped to 0 |
| Duplicates | 0 | 0 |

**Quality Gate: PASSED (100% pass rate)**

**SQLite JOIN result:** European Region (EURO) had the highest total cases (281 million). Americas (AMRO) had the highest death toll (3.07 million). Africa (AFRO) had the highest case fatality ratio (1.93%).

**Visualizations:**
- `figure1` — Monthly new cases by WHO region shows the Delta and Omicron waves clearly.
- `figure2` — Top 10 countries: USA, China, India, France, Germany lead cumulative counts.
- `figure3` — Log-scale scatter shows strong linear correlation between weekly cases and deaths.

---

## Part B — Model Comparison

**Task:** Predict `New_deaths` (weekly COVID-19 deaths per country) — Regression
**Split:** Time-based at 2023-01-01 (train: 2020–2022; test: 2023–2026)

| Model | CV MAE | Test MAE | Test R² |
|---|---|---|---|
| Baseline (Mean Dummy) | 287.87 | 180.77 | -0.44 |
| Linear Regression | 233.84 | 305.12 | -8.62 |
| Ridge Regression | 233.76 | 305.07 | -8.62 |
| **Random Forest** | **106.96** | **19.61** | **0.11** |

**Winner:** Random Forest (Test MAE = 19.61 deaths/week).
**Top feature:** `New_cases` (59.6% of splits) — epidemiologically sensible.

**Metric choice:** MAE is used because it is interpretable in original units (deaths/week). We do not use accuracy because this is a regression problem.

---

## AI Tools Acknowledgment

An AI coding assistant was used to help structure code and documentation. All code was reviewed and understood by the student author.

**Dataset source and license:** See `docs/schema.md`.
**Modeling assumptions, limitations and risks:** See `docs/assumptions.md`.
