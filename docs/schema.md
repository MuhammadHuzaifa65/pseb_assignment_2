# Raw Data Schema — WHO COVID-19 Global Data

**Author:** Muhammad Huzaifa | **Roll Number:** TRN-2026-001512 | **Date:** November 20, 2026

## 1. Overview

| Attribute | Value |
|---|---|
| Dataset Name | WHO COVID-19 Global Weekly Surveillance Data |
| Source URL | https://covid19.who.int/data |
| License | CC BY-NC-SA 3.0 IGO |
| Access Date | November 2026 |
| Total Rows | 85,680 |
| Total Columns | 8 |
| Date Range | 2020-01-05 to 2026-11-01 (weekly) |

---

## 2. Column Specifications

| Column | Type | Format / Units | Allowed Values | Missing (Raw) | Notes |
|---|---|---|---|---|---|
| `Date_reported` | String → Datetime | `YYYY-MM-DD` | Valid dates 2020–2026 | 0 | End of 7-day reporting period |
| `Country_code` | String | ISO 3166-1 alpha-2 | 2 uppercase chars | 357 | Namibia code `NA` is misread as null by default pandas |
| `Country` | String | Country name | Official WHO name | 0 | Includes overseas territories |
| `WHO_region` | String | Region code | AFRO, AMRO, EMRO, EURO, SEARO, WPRO, OTHER | 6,426 | Blank for some territories |
| `New_cases` | Float | Count | Integer >= 0 | 32,389 | Negatives = retrospective corrections |
| `Cumulative_cases` | Integer | Count | Integer >= 0 | 0 | Running total |
| `New_deaths` | Float | Count | Integer >= 0 | 47,799 | Negatives = retrospective corrections |
| `Cumulative_deaths` | Integer | Count | Integer >= 0 | 0 | Running total |

---

## 3. Data Quality Issues Found

1. **Large share of missing numerics:** 37.8% of `New_cases` and 55.8% of `New_deaths` are blank (early pandemic weeks with no reported activity).
2. **Negative values:** `New_cases` min = -65,079 and `New_deaths` min = -3,432 (WHO retrospective downward corrections).
3. **Blank WHO_region:** 6,426 rows from overseas territories lack a region code.
4. **Namibia country code:** ISO code `NA` is treated as null by pandas default null-detection.

---

## 4. Target Variable (Part B)

| Attribute | Value |
|---|---|
| Target column | `New_deaths` |
| Problem type | Regression |
| Use case | Predict weekly deaths per country to help health systems pre-position ICU capacity |
| Leakage risk | `Cumulative_deaths` must be excluded: `Cumulative_deaths(t) = Cumulative_deaths(t-1) + New_deaths(t)` |
