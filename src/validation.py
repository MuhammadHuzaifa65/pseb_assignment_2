# Name: Muhammad Huzaifa
# Roll Number: TRN-2026-001512
# Date: November 20, 2026

import json
import os
import pandas as pd
import numpy as np


def validate_types(df):
    """
    Validate that critical columns have the required data types.
    Ensures Date_reported is a datetime type and all case and death
    metrics are numeric dtypes.
    """
    if "Date_reported" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date_reported"]):
        return False, "Error: Date_reported must be a datetime type."

    for col in ["New_cases", "Cumulative_cases", "New_deaths", "Cumulative_deaths"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Error: {col} must be numeric."

    return True, "Types check passed."


def validate_ranges(df):
    """
    Validate that all surveillance counts fall within allowable ranges.
    Ensures that New_cases, Cumulative_cases, New_deaths, and Cumulative_deaths
    contain no negative values.
    """
    for col in ["New_cases", "Cumulative_cases", "New_deaths", "Cumulative_deaths"]:
        if col in df.columns:
            bad_count = (df[col] < 0).sum()
            if bad_count > 0:
                return False, f"Error: Found {bad_count} negative values in {col}."

    return True, "Range check passed."


def validate_uniqueness(df):
    """
    Validate uniqueness of the primary surveillance key.
    Checks that each (Date_reported, Country) pair appears at most once.
    """
    if "Date_reported" in df.columns and "Country" in df.columns:
        dupes = df.duplicated(subset=["Date_reported", "Country"]).sum()
        if dupes > 0:
            return False, f"Error: Found {dupes} duplicate rows for (Date_reported, Country)."

    return True, "Uniqueness check passed."


def validate_nulls(df):
    """
    Validate that mandatory surveillance columns contain no missing values.
    Checks Date_reported, Country, WHO_region, New_cases, and New_deaths.
    """
    important_cols = ["Date_reported", "Country", "WHO_region", "New_cases", "New_deaths"]
    for col in important_cols:
        if col in df.columns:
            nulls = df[col].isnull().sum()
            if nulls > 0:
                return False, f"Error: {col} contains {nulls} null values."

    return True, "Null check passed."


def run_data_quality_gate(df, min_pass_rate=0.95, metrics_path="outputs/reports/metrics.json"):
    """
    Run the automated data quality gate on the DataFrame.
    Evaluates row-level validation by checking the percentage of individual rows
    that satisfy all validity criteria (non-negative counts, valid dates, non-null
    mandatory fields, and key uniqueness). Halts with an error if the row pass rate
    falls below min_pass_rate, and writes summary metrics to a JSON file.
    """
    total_rows = len(df)

    if total_rows == 0:
        raise ValueError("Quality Gate FAILED: input DataFrame is empty!")

    # Check row-level conditions
    # 1. Date is valid (not null/NaT)
    if "Date_reported" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["Date_reported"]):
            c_date = df["Date_reported"].notnull()
        else:
            c_date = pd.to_datetime(df["Date_reported"], errors="coerce").notnull()
    else:
        c_date = pd.Series([False] * total_rows, index=df.index)

    # 2. Case and death counts are non-null and >= 0
    c_cases = df["New_cases"].notnull() & (df["New_cases"] >= 0) if "New_cases" in df.columns else True
    c_deaths = df["New_deaths"].notnull() & (df["New_deaths"] >= 0) if "New_deaths" in df.columns else True
    c_cum_cases = df["Cumulative_cases"].notnull() & (df["Cumulative_cases"] >= 0) if "Cumulative_cases" in df.columns else True
    c_cum_deaths = df["Cumulative_deaths"].notnull() & (df["Cumulative_deaths"] >= 0) if "Cumulative_deaths" in df.columns else True

    # 3. Categorical fields are non-null and non-empty
    c_country = df["Country"].notnull() & (df["Country"].astype(str).str.strip() != "") if "Country" in df.columns else True
    c_region = df["WHO_region"].notnull() & (df["WHO_region"].astype(str).str.strip() != "") if "WHO_region" in df.columns else True

    # 4. Row uniqueness
    c_unique = ~df.duplicated(subset=["Date_reported", "Country"]) if {"Date_reported", "Country"}.issubset(df.columns) else True

    # Combine row conditions
    valid_rows_mask = c_date & c_cases & c_deaths & c_cum_cases & c_cum_deaths & c_country & c_region & c_unique

    valid_count = int(valid_rows_mask.sum())
    invalid_count = total_rows - valid_count
    row_pass_rate = valid_count / total_rows

    # Also record individual check statuses
    checks = {
        "types_check": validate_types(df),
        "ranges_check": validate_ranges(df),
        "uniqueness_check": validate_uniqueness(df),
        "nulls_check": validate_nulls(df),
    }

    metrics = {
        "total_rows": total_rows,
        "valid_rows": valid_count,
        "invalid_rows": invalid_count,
        "pass_rate": round(row_pass_rate, 4),
        "row_validation_pass_rate": round(row_pass_rate, 4),
        "threshold": min_pass_rate,
        "status": "PASSED" if row_pass_rate >= min_pass_rate else "FAILED",
        "gate_status": "PASSED" if row_pass_rate >= min_pass_rate else "FAILED",
        "dataset_level_checks": {k: {"passed": v[0], "message": v[1]} for k, v in checks.items()}
    }

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Halt pipeline if row pass rate falls below threshold
    if row_pass_rate < min_pass_rate:
        raise ValueError(
            f"Quality Gate FAILED: only {row_pass_rate:.1%} of rows passed validation "
            f"({valid_count}/{total_rows} valid rows). Minimum required is {min_pass_rate:.1%}!"
        )

    return metrics
