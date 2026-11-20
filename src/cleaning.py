# Name: Muhammad Huzaifa
# Roll Number: TRN-2026-001512
# Date: November 20, 2026

import pandas as pd
import numpy as np


def fix_data_types(df):
    # Make a copy so we don't change the original
    cleaned = df.copy()

    # Convert the date column from string to actual date format
    cleaned["Date_reported"] = pd.to_datetime(cleaned["Date_reported"], errors="coerce")

    # Clean up text columns by stripping extra spaces
    for col in ["Country_code", "Country", "WHO_region"]:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    # Convert case and death columns to numbers
    for col in ["New_cases", "Cumulative_cases", "New_deaths", "Cumulative_deaths"]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    return cleaned


def standardize_categories(df):
    # Fill blank WHO regions with OTHER
    cleaned = df.copy()
    cleaned["WHO_region"] = cleaned["WHO_region"].replace({"": "OTHER", "nan": "OTHER"})
    cleaned["WHO_region"] = cleaned["WHO_region"].fillna("OTHER")
    cleaned["WHO_region"] = cleaned["WHO_region"].str.upper()

    # Fill blank country codes with XX (unknown)
    cleaned["Country_code"] = cleaned["Country_code"].replace({"": "XX", "nan": "XX"})
    cleaned["Country_code"] = cleaned["Country_code"].fillna("XX")
    cleaned["Country_code"] = cleaned["Country_code"].str.upper()

    return cleaned


def handle_missing_values(df):
    # If New_cases or New_deaths is empty it means zero was reported that week
    cleaned = df.copy()
    for col in ["New_cases", "New_deaths", "Cumulative_cases", "Cumulative_deaths"]:
        cleaned[col] = cleaned[col].fillna(0.0)
    return cleaned


def handle_duplicates(df):
    # Remove rows where the same country appears twice on the same date
    cleaned = df.copy()
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["Date_reported", "Country"], keep="first")
    cleaned = cleaned.reset_index(drop=True)
    removed = before - len(cleaned)
    return cleaned, removed


def handle_outliers(df):
    # WHO sometimes sends negative numbers to correct old data
    # We clip them to 0 because a negative count makes no sense for analysis
    cleaned = df.copy()
    neg_cases = int((cleaned["New_cases"] < 0).sum())
    neg_deaths = int((cleaned["New_deaths"] < 0).sum())

    # np.maximum returns element-wise max between the value and 0
    cleaned["New_cases"] = np.maximum(cleaned["New_cases"], 0.0)
    cleaned["New_deaths"] = np.maximum(cleaned["New_deaths"], 0.0)

    counts = {"negative_new_cases": neg_cases, "negative_new_deaths": neg_deaths}
    return cleaned, counts


def clean_dataset(raw_df):
    # Run each cleaning step one by one
    info = {
        "raw_rows": len(raw_df),
        "raw_cols": len(raw_df.columns)
    }

    df = fix_data_types(raw_df)
    df = standardize_categories(df)
    df = handle_missing_values(df)
    df, dupes = handle_duplicates(df)
    df, outliers = handle_outliers(df)

    info["cleaned_rows"] = len(df)
    info["duplicates_removed"] = dupes
    info["outliers_handled"] = outliers

    return df, info
