# Name: Muhammad Huzaifa
# Roll Number: TRN-2026-001512
# Date: November 20, 2026

import json
import os
import pandas as pd


def validate_types(df):
    # Check that the date column is actually a date, not plain text
    if not pd.api.types.is_datetime64_any_dtype(df["Date_reported"]):
        return False, "Error: Date_reported should be a date type."

    # Check that case and death columns are numbers
    for col in ["New_cases", "Cumulative_cases", "New_deaths", "Cumulative_deaths"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Error: {col} should be a number."

    return True, "Types check passed."


def validate_ranges(df):
    # Cases and deaths cannot be negative
    for col in ["New_cases", "Cumulative_cases", "New_deaths", "Cumulative_deaths"]:
        if col in df.columns:
            bad_count = (df[col] < 0).sum()
            if bad_count > 0:
                return False, f"Error: Found {bad_count} negative values in {col}."

    return True, "Range check passed."


def validate_uniqueness(df):
    # Each country should appear only once per date
    if "Date_reported" in df.columns and "Country" in df.columns:
        dupes = df.duplicated(subset=["Date_reported", "Country"]).sum()
        if dupes > 0:
            return False, f"Error: Found {dupes} duplicate rows."

    return True, "Uniqueness check passed."


def validate_nulls(df):
    # These important columns must not have any missing values after cleaning
    important_cols = ["Date_reported", "Country", "WHO_region", "New_cases", "New_deaths"]
    for col in important_cols:
        if col in df.columns:
            nulls = df[col].isnull().sum()
            if nulls > 0:
                return False, f"Error: {col} has {nulls} null values."

    return True, "Null check passed."


def run_data_quality_gate(df, min_pass_rate=0.95, metrics_path="outputs/reports/metrics.json"):
    # Run all 4 checks and count how many pass
    checks = {
        "types_check": validate_types(df),
        "ranges_check": validate_ranges(df),
        "uniqueness_check": validate_uniqueness(df),
        "nulls_check": validate_nulls(df),
    }

    passed = sum(1 for ok, _ in checks.values() if ok)
    total = len(checks)
    rate = passed / total

    # Save the results to a JSON file every time the pipeline runs
    metrics = {
        "total_rows": int(len(df)),
        "total_checks": total,
        "passed_checks": passed,
        "pass_rate": round(rate, 4),
        "threshold": min_pass_rate,
        "status": "PASSED" if rate >= min_pass_rate else "FAILED",
        "details": {k: {"passed": v[0], "message": v[1]} for k, v in checks.items()}
    }

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Stop the pipeline if data quality is too low
    if rate < min_pass_rate:
        raise ValueError(f"Quality Gate FAILED: only {rate:.0%} checks passed (need {min_pass_rate:.0%})!")

    return metrics
