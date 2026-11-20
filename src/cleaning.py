# Name: Muhammad Huzaifa
# Roll Number: TRN-2026-001512
# Date: November 20, 2026

import pandas as pd
import numpy as np


def fix_data_types(df):
    """
    Cast raw columns to their expected data types.
    Converts Date_reported to datetime, text columns to stripped strings,
    and surveillance counts to numeric values.
    """
    cleaned = df.copy()

    # Convert the date column from string to actual date format
    cleaned["Date_reported"] = pd.to_datetime(cleaned["Date_reported"], errors="coerce")

    # Clean up text columns by stripping extra spaces
    for col in ["Country_code", "Country", "WHO_region"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip()

    # Convert case and death columns to numbers
    for col in ["New_cases", "Cumulative_cases", "New_deaths", "Cumulative_deaths"]:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    return cleaned


def standardize_categories(df):
    """
    Standardize categorical codes across the dataset.
    Fills missing or empty WHO regions with 'OTHER' and blank country
    codes with 'XX' (unknown).
    """
    cleaned = df.copy()
    if "WHO_region" in cleaned.columns:
        cleaned["WHO_region"] = cleaned["WHO_region"].replace({"": "OTHER", "nan": "OTHER", "None": "OTHER"})
        cleaned["WHO_region"] = cleaned["WHO_region"].fillna("OTHER")
        cleaned["WHO_region"] = cleaned["WHO_region"].str.upper()

    if "Country_code" in cleaned.columns:
        cleaned["Country_code"] = cleaned["Country_code"].replace({"": "XX", "nan": "XX", "None": "XX"})
        cleaned["Country_code"] = cleaned["Country_code"].fillna("XX")
        cleaned["Country_code"] = cleaned["Country_code"].str.upper()

    if "Country" in cleaned.columns:
        cleaned["Country"] = cleaned["Country"].str.strip()

    return cleaned


def handle_missing_values(df):
    """
    Impute missing values in numeric surveillance columns.
    In the WHO surveillance format, empty cells represent zero cases or
    deaths reported during that week, so they are imputed with 0.0.
    """
    cleaned = df.copy()
    for col in ["New_cases", "New_deaths", "Cumulative_cases", "Cumulative_deaths"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna(0.0)
    return cleaned


def handle_duplicates(df):
    """
    Remove duplicate surveillance records based on (Date_reported, Country).
    Keeps the first occurrence and returns the deduplicated DataFrame and
    the number of removed rows.
    """
    cleaned = df.copy()
    before = len(cleaned)
    subset = ["Date_reported", "Country"] if {"Date_reported", "Country"}.issubset(cleaned.columns) else None
    cleaned = cleaned.drop_duplicates(subset=subset, keep="first")
    cleaned = cleaned.reset_index(drop=True)
    removed = before - len(cleaned)
    return cleaned, removed


def handle_outliers(df):
    """
    Handle negative values in New_cases and New_deaths.
    Negative numbers in WHO data reflect retrospective downward corrections
    from local health authorities. They are clipped to 0.0 using np.maximum.
    """
    cleaned = df.copy()
    counts = {}

    if "New_cases" in cleaned.columns:
        counts["negative_new_cases"] = int((cleaned["New_cases"] < 0).sum())
        cleaned["New_cases"] = np.maximum(cleaned["New_cases"], 0.0)

    if "New_deaths" in cleaned.columns:
        counts["negative_new_deaths"] = int((cleaned["New_deaths"] < 0).sum())
        cleaned["New_deaths"] = np.maximum(cleaned["New_deaths"], 0.0)

    return cleaned, counts


def clean_dataset(raw_df):
    """
    Run the complete sequential cleaning pipeline on raw input data.
    Executes type casting, category standardization, missing value imputation,
    duplicate removal, and outlier clipping. Returns cleaned data and metadata.
    """
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
