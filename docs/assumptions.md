# Assumptions, Limitations, Risks and Model Card

**Author:** Muhammad Huzaifa | **Roll Number:** TRN-2026-001512 | **Date:** November 20, 2026

---

## Part A — Cleaning Assumptions

| Assumption | Decision | Reason |
|---|---|---|
| Blank `New_cases` / `New_deaths` = zero activity | Filled with `0.0` | WHO data dictionary confirms blanks mean no reported cases |
| Negative case/death counts = retrospective corrections | Clipped to `0.0` with `np.maximum` | Negatives are adjustment artefacts, not real values |
| Blank `WHO_region` = overseas territory | Labeled `'OTHER'` | These are valid rows, just not assigned to a WHO bureau |
| Namibia `NA` code | Loaded with `keep_default_na=False` | Preserves `NA` as a string, not a null |
| Duplicate key = `(Date_reported, Country)` | Kept first occurrence, dropped rest | Each country should have exactly one weekly row |

## What We Did Not Fix

- Did not remove all-zero rows from Jan–Feb 2020: they are epidemiologically valid (pre-pandemic).
- Did not adjust cumulative counts after clipping negatives: would require complex recalculation and is out of scope.

## Part A Limitations

- Weekly aggregation hides intra-week surge dynamics.
- Under-reporting in low-resource health systems means actual numbers may be higher than recorded.
- Retrospective corrections remain in cumulative columns even after clipping weekly values.

---

## Part B — Model Card

### What is Predicted
**Target:** `New_deaths` — weekly newly confirmed COVID-19 deaths per country  
**Problem type:** Regression  
**Who uses it:** Public health analysts and hospital planners to estimate weekly mortality demand  

### Features Used

| Feature | Type | Why included |
|---|---|---|
| `New_cases` | Numeric | Primary driver of deaths |
| `Cumulative_cases` | Numeric | Proxy for overall pandemic maturity |
| `Month` | Numeric | Captures seasonal respiratory wave patterns |
| `Day_of_week` | Numeric | Captures reporting day variation |
| `WHO_region` | Categorical | Region-level baseline mortality differences |

### Columns Excluded to Prevent Leakage

| Column | Reason |
|---|---|
| `Cumulative_deaths` | Contains target: `Cumulative(t) = Cumulative(t-1) + New_deaths(t)` |
| `Mortality_rate_percent` | Derived from `Cumulative_deaths` |
| `Log_new_deaths` | Mathematical transform of the target |

### Split Justification
**Time-based split at 2023-01-01** — not random.  
A random split would let 2024 rows inform 2021 predictions, which is temporal leakage. The time-based cut honestly evaluates whether the model works on future data the pipeline has never seen.

### Model Performance

| Model | CV MAE | Test MAE | Test R² |
|---|---|---|---|
| Baseline (Mean Dummy) | 287.87 | 180.77 | -0.44 |
| Linear Regression | 233.84 | 305.12 | -8.62 |
| Ridge Regression | 233.76 | 305.07 | -8.62 |
| Random Forest | 106.96 | 19.61 | 0.11 |

**Metric choice:** MAE is interpretable in original units (deaths/week) and treats all errors equally. R² shows variance explained. We do not use accuracy because this is regression.

### Limitations and Risks

1. **Low R² (0.11):** The model explains only 11% of variance on the test set. The post-vaccine era (2023+) behaves very differently from the training era (2020–2022).
2. **No lag features:** Clinically, deaths lag behind cases by 1–3 weeks. Without lag features the model cannot capture this delay.
3. **Retrospective batch spikes:** Countries occasionally release months of backlogged deaths in one week, which the model cannot predict from same-week case counts.
4. **Not for clinical decisions:** This is a teaching baseline. Any policy use requires validation by qualified epidemiologists.

### Intended Use Boundaries
- **OK:** Academic exploration, teaching regression modeling, rough regional planning estimates.
- **NOT OK:** Individual patient triage, real-time clinical systems, legally binding policy decisions.

### AI Tools Acknowledgment
An AI coding assistant was used to help structure code and documentation. All code was reviewed and understood by the student author and can be explained line-by-line.
