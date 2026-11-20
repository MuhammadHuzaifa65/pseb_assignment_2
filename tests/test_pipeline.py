# Name: Muhammad Huzaifa
# Roll Number: TRN-2026-001512
# Date: November 20, 2026

import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from src.cleaning import (
    fix_data_types, standardize_categories, handle_missing_values,
    handle_duplicates, handle_outliers, clean_dataset
)
from src.validation import (
    validate_types, validate_ranges, validate_uniqueness, validate_nulls,
    run_data_quality_gate
)


@pytest.fixture
def sample_raw():
    # Small made-up dataset that has the same problems as the real WHO data
    return pd.DataFrame({
        "Date_reported": ["2021-05-01", "2021-05-08", "2021-05-08", "2021-05-15"],
        "Country_code": ["PK", "NA", "NA", ""],
        "Country": ["Pakistan", "Namibia", "Namibia", "Unknown"],
        "WHO_region": ["EMRO", "AFRO", "AFRO", ""],
        "New_cases": ["1000", "-50", "-50", None],
        "Cumulative_cases": ["50000", "2000", "2000", "100"],
        "New_deaths": ["20", "-5", "-5", None],
        "Cumulative_deaths": ["1000", "50", "50", "2"],
    })


def test_fix_data_types(sample_raw):
    # After fixing types, Date_reported should be a date and cases should be a number
    df = fix_data_types(sample_raw)
    assert pd.api.types.is_datetime64_any_dtype(df["Date_reported"])
    assert pd.api.types.is_numeric_dtype(df["New_cases"])


def test_standardize_categories(sample_raw):
    df = fix_data_types(sample_raw)
    df = standardize_categories(df)
    # Empty region should become OTHER
    assert "OTHER" in df["WHO_region"].values
    # Namibia's ISO code NA must be kept
    assert "NA" in df["Country_code"].values
    # Blank country code becomes XX
    assert "XX" in df["Country_code"].values


def test_handle_missing_values(sample_raw):
    # After filling, there should be no missing values in numeric columns
    df = fix_data_types(sample_raw)
    df = handle_missing_values(df)
    assert df["New_cases"].isnull().sum() == 0
    assert df["New_deaths"].isnull().sum() == 0
    # The row that had None should now be 0.0
    assert df.loc[3, "New_cases"] == 0.0


def test_handle_duplicates(sample_raw):
    # Row 1 and row 2 are both Namibia on 2021-05-08, so one should be removed
    df = fix_data_types(sample_raw)
    cleaned, dupes = handle_duplicates(df)
    assert dupes == 1
    assert len(cleaned) == 3


def test_handle_outliers():
    # Negative values should be clipped to zero
    df = pd.DataFrame({
        "New_cases": [-100.0, 50.0, -10.0, 200.0],
        "New_deaths": [-5.0, 10.0, 0.0, -1.0],
    })
    cleaned, counts = handle_outliers(df)
    assert (cleaned["New_cases"] >= 0).all()
    assert (cleaned["New_deaths"] >= 0).all()
    assert counts["negative_new_cases"] == 2
    assert counts["negative_new_deaths"] == 2


def test_validation_functions():
    # Build a clean dataset that should pass all checks
    good = pd.DataFrame({
        "Date_reported": pd.to_datetime(["2021-01-01", "2021-01-08"]),
        "Country_code": ["PK", "US"],
        "Country": ["Pakistan", "United States"],
        "WHO_region": ["EMRO", "AMRO"],
        "New_cases": [500.0, 12000.0],
        "Cumulative_cases": [5000.0, 200000.0],
        "New_deaths": [10.0, 300.0],
        "Cumulative_deaths": [100.0, 5000.0],
    })

    assert validate_types(good)[0] is True
    assert validate_ranges(good)[0] is True
    assert validate_uniqueness(good)[0] is True
    assert validate_nulls(good)[0] is True

    # Now add a negative value - range check must catch it
    bad = good.copy()
    bad.loc[0, "New_cases"] = -50.0
    passed, msg = validate_ranges(bad)
    assert passed is False
    assert "negative" in msg


def test_quality_gate(sample_raw):
    # Clean data should pass the gate; corrupted data should fail loudly
    cleaned_df, _ = clean_dataset(sample_raw)

    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_file = os.path.join(tmpdir, "metrics.json")

        # Should pass on clean data
        result = run_data_quality_gate(cleaned_df, min_pass_rate=0.95, metrics_path=metrics_file)
        assert result["status"] == "PASSED"
        assert os.path.exists(metrics_file)

        # Corrupt the data and check that it raises an error
        bad = cleaned_df.copy()
        bad["New_cases"] = -999.0
        bad["New_deaths"] = -50.0
        bad["Date_reported"] = "not_a_date"

        with pytest.raises(ValueError) as exc:
            run_data_quality_gate(bad, min_pass_rate=0.95, metrics_path=metrics_file)
        assert "Quality Gate FAILED" in str(exc.value)
