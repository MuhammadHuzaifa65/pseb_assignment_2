# Name: Muhammad Huzaifa
# Roll Number: TRN-2026-001512
# Date: November 20, 2026

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd
import numpy as np

from src.features import extract_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def check_columns(df):
    # Make sure the input file has all the columns we need before predicting
    needed = ["Date_reported", "WHO_region", "New_cases", "Cumulative_cases"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing these columns: {missing}")


def print_feature_importance(pipeline):
    # Show which features the Random Forest used most
    try:
        model = pipeline.named_steps["model"]
        pre = pipeline.named_steps["preprocessor"]
        cat_names = list(
            pre.named_transformers_["cat"]
            .named_steps["encode"]
            .get_feature_names_out(CATEGORICAL_FEATURES)
        )
        all_names = NUMERIC_FEATURES + cat_names

        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame({
                "Feature": all_names,
                "Importance": np.round(model.feature_importances_, 4)
            }).sort_values("Importance", ascending=False)

            top = imp.iloc[0]
            print("\n--- Feature Importance ---")
            print(imp.to_string(index=False))
            print(f"\nMost important: '{top['Feature']}' ({top['Importance']:.1%})")
            print("This makes sense — more new cases means more deaths.")
    except Exception as e:
        print(f"Could not show feature importance: {e}")


def main():
    parser = argparse.ArgumentParser(description="Predict weekly COVID-19 deaths for new data.")
    parser.add_argument("--input", default="data/sample_unseen.csv", help="Path to input CSV")
    parser.add_argument("--output", default="outputs/predictions.csv", help="Where to save predictions")
    parser.add_argument("--model", default="models/covid_death_predictor.joblib", help="Path to saved model")
    args = parser.parse_args()

    print("=" * 55)
    print("  PREDICT: COVID-19 WEEKLY DEATH ESTIMATES")
    print("=" * 55)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found at '{args.model}'. Run src/train.py first.")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: '{args.input}'")

    # Load the new rows
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} row(s) from {args.input}")

    # Check required columns are present
    check_columns(df)

    # Extract month/day features and select the right columns
    df_feat = extract_features(df)
    X = df_feat[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    # Load the saved model and predict
    pipeline = joblib.load(args.model)
    raw_preds = pipeline.predict(X)

    # Clip to 0 since deaths cannot be negative
    preds = np.maximum(raw_preds, 0.0)

    # Add predictions to the original data and save
    output_df = df.copy()
    output_df["Predicted_New_deaths"] = np.round(preds, 1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df.to_csv(args.output, index=False)

    print(f"\nPredictions saved to: {args.output}")
    print(output_df[["Date_reported", "Country", "WHO_region", "New_cases", "Predicted_New_deaths"]].to_string(index=False))

    print_feature_importance(pipeline)
    print("\nDone.")


if __name__ == "__main__":
    main()
