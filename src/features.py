# Name: Muhammad Huzaifa
# Roll Number: TRN-2026-001512
# Date: November 20, 2026

import pandas as pd

# These are the columns we will use as input features for the model
NUMERIC_FEATURES = ["New_cases", "Cumulative_cases", "Month", "Day_of_week"]
CATEGORICAL_FEATURES = ["WHO_region"]
TARGET = "New_deaths"

# These columns must NOT be used as features because they give away the answer
# Cumulative_deaths already includes New_deaths inside it, so using it would be cheating
LEAKAGE_COLUMNS = ["Cumulative_deaths", "Mortality_rate_percent", "Log_new_deaths"]


def extract_features(df):
    # Pull the month number and day of week from the date column
    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data["Date_reported"]):
        data["Date_reported"] = pd.to_datetime(data["Date_reported"])

    data["Month"] = data["Date_reported"].dt.month
    data["Day_of_week"] = data["Date_reported"].dt.dayofweek
    data["Year"] = data["Date_reported"].dt.year
    return data


def split_features_target(df):
    # Separate the input columns (X) from the column we want to predict (y)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y
