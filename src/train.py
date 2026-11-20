# Name: Muhammad Huzaifa
# Roll Number: TRN-2026-001512
# Date: November 20, 2026

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

from src.features import extract_features, split_features_target, NUMERIC_FEATURES, CATEGORICAL_FEATURES

CLEAN_PATH = "data/cleaned/cleaned_covid_data.csv"
SPLIT_DATE = "2023-01-01"
MODEL_PATH = "models/covid_death_predictor.joblib"


def build_preprocessor():
    # Numeric columns: fill missing values with the median, then scale to similar range
    num_steps = Pipeline([
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    # Categorical columns: fill missing with "OTHER", then convert to 0/1 columns
    cat_steps = Pipeline([
        ("fill", SimpleImputer(strategy="constant", fill_value="OTHER")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Combine numeric and categorical transformations
    preprocessor = ColumnTransformer([
        ("num", num_steps, NUMERIC_FEATURES),
        ("cat", cat_steps, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def train_and_evaluate():
    print("=" * 55)
    print("  PART B: ML MODEL TRAINING AND EVALUATION")
    print("=" * 55)

    # --- Step 1: Load cleaned data ---
    print(f"\n[1] Loading cleaned data...")
    if not os.path.exists(CLEAN_PATH):
        raise FileNotFoundError(f"Run Part A first. Missing: {CLEAN_PATH}")

    df = pd.read_csv(CLEAN_PATH)
    df = extract_features(df)
    print(f"    Total rows: {len(df):,}")

    # --- Step 2: Train/test split ---
    # We split by DATE not randomly because this is time series data
    # Using a random split would let the model "see the future" which is cheating
    print(f"\n[2] Splitting into train (before {SPLIT_DATE}) and test (after)...")
    train_df = df[df["Date_reported"] < SPLIT_DATE].copy()
    test_df = df[df["Date_reported"] >= SPLIT_DATE].copy()

    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)
    print(f"    Training rows: {len(X_train):,}")
    print(f"    Test rows    : {len(X_test):,}")

    # --- Step 3: Define models ---
    # Baseline: always predict the average (simplest possible model)
    # Then we try two real models and compare
    models = {
        "Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
    }

    results = []
    trained_models = {}
    preprocessor = build_preprocessor()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[3] Training each model with 5-fold cross-validation...")
    for name, model in models.items():
        # Combine preprocessing and the model into one pipeline object
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

        # Cross-validate: train on 4 folds, test on the 5th, repeat 5 times
        cv_results = cross_validate(
            pipe, X_train, y_train, cv=kfold,
            scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]
        )

        # Train on all training data and test on the held-out test set
        pipe.fit(X_train, y_train)
        trained_models[name] = pipe

        # Predictions can't be negative (can't have negative deaths)
        y_pred = np.maximum(pipe.predict(X_test), 0.0)

        row = {
            "Model": name,
            "CV_MAE": round(-cv_results["test_neg_mean_absolute_error"].mean(), 2),
            "CV_RMSE": round(-cv_results["test_neg_root_mean_squared_error"].mean(), 2),
            "CV_R2": round(cv_results["test_r2"].mean(), 4),
            "Test_MAE": round(mean_absolute_error(y_test, y_pred), 2),
            "Test_RMSE": round(root_mean_squared_error(y_test, y_pred), 2),
            "Test_R2": round(r2_score(y_test, y_pred), 4),
        }
        results.append(row)
        print(f"    {name:25s} | CV MAE: {row['CV_MAE']:7.2f} | Test MAE: {row['Test_MAE']:7.2f} | R2: {row['Test_R2']:.4f}")

    # --- Step 4: Save comparison table as CSV ---
    comparison = pd.DataFrame(results)
    os.makedirs("outputs/reports", exist_ok=True)
    comparison.to_csv("outputs/reports/model_comparison.csv", index=False)
    print("\n    Saved model_comparison.csv.")

    # --- Step 5: Residual plot for the best model ---
    # Random Forest gave the best test MAE so we analyse its errors
    best_name = "Random Forest"
    best_pipe = trained_models[best_name]
    y_pred_best = np.maximum(best_pipe.predict(X_test), 0.0)
    residuals = y_test.values - y_pred_best

    os.makedirs("outputs/figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred_best, residuals, alpha=0.3, color="#16a085", edgecolors="none", s=20)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title("Residual Plot - Random Forest on Test Set", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Weekly Deaths", fontsize=11)
    ax.set_ylabel("Residuals (Actual - Predicted)", fontsize=11)
    ax.set_ylim(-3000, 3000)
    plt.tight_layout()
    plt.savefig("outputs/figures/figure4_residual_plot.png", dpi=200)
    plt.close()
    print("    Saved figure4_residual_plot.png.")

    # --- Step 6: Error analysis saved as plain text ---
    test_df = test_df.copy()
    test_df["Predicted"] = np.round(y_pred_best, 1)
    test_df["Error"] = np.abs(residuals)

    # Rows where the model was most wrong
    worst_rows = test_df.sort_values("Error", ascending=False).head(10)

    # Average error per region
    region_errors = test_df.groupby("WHO_region").agg(
        Rows=("New_deaths", "count"),
        Avg_Deaths=("New_deaths", "mean"),
        Avg_Error=("Error", "mean")
    ).reset_index()

    test_mae = mean_absolute_error(y_test, y_pred_best)
    test_r2 = r2_score(y_test, y_pred_best)

    error_lines = [
        f"ERROR ANALYSIS - {best_name}",
        "=" * 50,
        f"Test MAE : {test_mae:.2f} deaths/week",
        f"Test R2  : {test_r2:.4f}",
        "",
        "TOP 10 WORST PREDICTIONS",
        worst_rows[["Date_reported", "Country", "New_deaths", "Predicted", "Error"]].to_string(index=False),
        "",
        "WHY THE MODEL FAILS",
        "1. Some countries dump months of backlogged deaths in one week.",
        "   The model cannot predict a spike from same-week cases alone.",
        "2. Deaths usually peak 2-3 weeks after cases. Without lag features",
        "   the model underestimates deaths at the end of a wave.",
        "3. Small islands with 0 deaths give a small constant error.",
        "",
        "ERROR BY WHO REGION",
        region_errors.to_string(index=False),
    ]
    with open("outputs/reports/error_analysis.txt", "w") as f:
        f.write("\n".join(error_lines))
    print("    Saved error_analysis.txt.")

    # --- Step 7: Save the trained pipeline ---
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)
    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"    Saved pipeline to {MODEL_PATH} ({size_kb:.1f} KB).")

    print("\n" + "=" * 55)
    print("  PART B DONE")
    print("=" * 55)


if __name__ == "__main__":
    train_and_evaluate()
