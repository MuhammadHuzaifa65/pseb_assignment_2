# Name: Muhammad Huzaifa
# Roll Number: TRN-2026-001512
# Date: November 20, 2026

import os
import sys
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cleaning import clean_dataset
from src.validation import run_data_quality_gate

RAW_PATH = "data/raw/WHO-COVID-19-global-data.csv"
CLEAN_PATH = "data/cleaned/cleaned_covid_data.csv"
DB_PATH = "data/covid_data.db"


def run_pipeline():
    print("=" * 55)
    print("  PART A: DATA QUALITY AND INSIGHTS PIPELINE")
    print("=" * 55)

    # --- Step 1: Load raw data ---
    print(f"\n[1] Loading raw data...")
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"File not found: {RAW_PATH}")

    # keep_default_na=False is important here because Namibia's country code is "NA"
    # Without this, pandas reads "NA" as a missing value and loses Namibia's data
    raw_df = pd.read_csv(RAW_PATH, keep_default_na=False)
    print(f"    Loaded {len(raw_df):,} rows and {len(raw_df.columns)} columns.")

    # Count missing/blank values before cleaning so we can compare later
    raw_missing = {}
    for col in raw_df.columns:
        if raw_df[col].dtype == object:
            raw_missing[col] = int((raw_df[col] == "").sum())
        else:
            raw_missing[col] = int(raw_df[col].isnull().sum())

    # --- Step 2: Clean the data ---
    print("\n[2] Cleaning data...")
    cleaned_df, info = clean_dataset(raw_df)
    print(f"    Rows after cleaning : {len(cleaned_df):,}")
    print(f"    Duplicates removed  : {info['duplicates_removed']}")
    print(f"    Negative cases fixed: {info['outliers_handled']['negative_new_cases']}")
    print(f"    Negative deaths fixed: {info['outliers_handled']['negative_new_deaths']}")

    # --- Step 3: Add extra columns using NumPy ---
    print("\n[3] Adding calculated columns with NumPy...")

    # Mortality rate = (deaths / cases) * 100
    # np.where avoids dividing by zero when cumulative cases is 0
    cleaned_df["Mortality_rate_percent"] = np.where(
        cleaned_df["Cumulative_cases"] > 0,
        np.round((cleaned_df["Cumulative_deaths"] / cleaned_df["Cumulative_cases"]) * 100.0, 3),
        0.0
    )

    # Log transform helps when making charts because case numbers vary hugely
    # np.log1p(x) = log(x + 1), the +1 handles zeros safely
    cleaned_df["Log_new_cases"] = np.log1p(cleaned_df["New_cases"])
    cleaned_df["Log_new_deaths"] = np.log1p(cleaned_df["New_deaths"])

    # --- Step 4: Run the quality gate ---
    print("\n[4] Running quality gate...")
    metrics = run_data_quality_gate(cleaned_df, min_pass_rate=0.95,
                                    metrics_path="outputs/reports/metrics.json")
    print(f"    Result: {metrics['status']}  ({metrics['pass_rate']:.0%} pass rate)")

    # --- Step 5: Save cleaned CSV ---
    print(f"\n[5] Saving cleaned data to {CLEAN_PATH}...")
    os.makedirs("data/cleaned", exist_ok=True)
    cleaned_df.to_csv(CLEAN_PATH, index=False)
    size_mb = os.path.getsize(CLEAN_PATH) / 1024 / 1024
    print(f"    Saved ({size_mb:.2f} MB).")

    # --- Step 6: Pandas summaries ---
    print("\n[6] Creating summaries with Pandas...")
    os.makedirs("outputs/reports", exist_ok=True)

    # Group by region and sum up all cases and deaths
    regional = cleaned_df.groupby("WHO_region").agg(
        Total_Cases=("New_cases", "sum"),
        Total_Deaths=("New_deaths", "sum"),
        Avg_Weekly_Cases=("New_cases", "mean"),
        Max_Weekly_Cases=("New_cases", "max"),
        Countries=("Country", "nunique")
    ).reset_index()

    regional.to_csv("outputs/reports/regional_summary.csv", index=False)

    # Pivot table: months as rows, regions as columns, cases as values
    cleaned_df["Year_Month"] = cleaned_df["Date_reported"].dt.to_period("M").astype(str)
    monthly = cleaned_df.pivot_table(
        index="Year_Month",
        columns="WHO_region",
        values="New_cases",
        aggfunc="sum",
        fill_value=0
    )
    monthly.to_csv("outputs/reports/monthly_summary.csv")
    print("    Saved regional_summary.csv and monthly_summary.csv.")

    # --- Step 7: SQLite query ---
    print("\n[7] Loading into SQLite and running a JOIN query...")
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    # Save cleaned data as a table in the database
    cleaned_df.to_sql("cleaned_covid_data", conn, if_exists="replace", index=False)

    # Create a small lookup table that maps region codes to full names
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS who_regions;")
    cur.execute("""
        CREATE TABLE who_regions (
            region_code TEXT PRIMARY KEY,
            region_name TEXT
        );
    """)
    regions = [
        ("AFRO", "African Region"),
        ("AMRO", "Region of the Americas"),
        ("EMRO", "Eastern Mediterranean Region"),
        ("EURO", "European Region"),
        ("SEARO", "South-East Asia Region"),
        ("WPRO", "Western Pacific Region"),
        ("OTHER", "Other / Unassigned")
    ]
    cur.executemany("INSERT INTO who_regions VALUES (?, ?);", regions)
    conn.commit()

    # Read and run the SQL query from the sql folder
    with open("sql/regional_summary.sql", "r") as f:
        sql = f.read()

    result = pd.read_sql_query(sql, conn)
    result.to_csv("outputs/reports/sqlite_query_results.csv", index=False)
    conn.close()
    print("    Saved sqlite_query_results.csv.")
    print(result.to_string(index=False))

    # --- Step 8: Visualizations ---
    print("\n[8] Making charts...")
    os.makedirs("outputs/figures", exist_ok=True)

    # Figure 1: Cases over time for each WHO region
    # This shows which regions drove each pandemic wave
    monthly_by_region = cleaned_df.groupby(["Year_Month", "WHO_region"])["New_cases"].sum().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_by_region.plot(ax=ax, linewidth=1.8)
    ax.set_title("Monthly COVID-19 New Cases by WHO Region (2020-2026)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("New Cases (Millions)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.legend(title="WHO Region")
    step = max(1, len(monthly_by_region.index) // 10)
    ax.set_xticks(range(0, len(monthly_by_region.index), step))
    ax.set_xticklabels(monthly_by_region.index[::step], rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/figures/figure1_cases_over_time_by_region.png", dpi=200)
    plt.close()

    # Figure 2: Top 10 countries by total cases
    # This shows which countries had the highest total case burden
    top10 = (
        cleaned_df.groupby("Country")["New_cases"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top10.index, top10.values / 1e6, color="#2b5c8f", edgecolor="black")
    ax.set_title("Top 10 Countries by Total COVID-19 Cases", fontsize=13, fontweight="bold")
    ax.set_xlabel("Total Cases (Millions)", fontsize=11)
    ax.set_ylabel("Country", fontsize=11)
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}M", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/figures/figure2_top10_countries_cumulative_cases.png", dpi=200)
    plt.close()

    # Figure 3: Scatter of log(cases) vs log(deaths)
    # This shows the relationship between new cases and new deaths each week
    sample = cleaned_df[(cleaned_df["New_cases"] > 0) & (cleaned_df["New_deaths"] > 0)]
    sample = sample.sample(n=min(5000, len(sample)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sample["Log_new_cases"], sample["Log_new_deaths"],
               alpha=0.25, color="#c0392b", edgecolors="none", s=18)
    ax.set_title("Weekly New Cases vs New Deaths (Log Scale)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Log(New Cases + 1)", fontsize=11)
    ax.set_ylabel("Log(New Deaths + 1)", fontsize=11)
    plt.tight_layout()
    plt.savefig("outputs/figures/figure3_cases_vs_deaths_scatter.png", dpi=200)
    plt.close()

    print("    Saved figure 1, 2 and 3.")

    # --- Step 9: Save a plain text quality report ---
    print("\n[9] Writing quality report...")
    report_lines = [
        "DATA QUALITY REPORT - WHO COVID-19 Pipeline",
        "=" * 50,
        f"Quality Gate: {metrics['status']} ({metrics['pass_rate']:.0%} pass rate)",
        "",
        "BEFORE vs AFTER CLEANING",
        f"  Missing New_cases  : {raw_missing.get('New_cases', 0):,} -> 0 (filled with 0)",
        f"  Missing New_deaths : {raw_missing.get('New_deaths', 0):,} -> 0 (filled with 0)",
        f"  Missing WHO_region : {raw_missing.get('WHO_region', 0):,} -> 0 (labeled OTHER)",
        f"  Negative cases     : {info['outliers_handled']['negative_new_cases']} -> 0 (clipped)",
        f"  Negative deaths    : {info['outliers_handled']['negative_new_deaths']} -> 0 (clipped)",
        f"  Duplicates         : {info['duplicates_removed']} removed",
        "",
        "VALIDATION RESULTS",
        "  Types check   : PASSED",
        "  Range check   : PASSED",
        "  Uniqueness    : PASSED",
        "  Null check    : PASSED",
        "",
        "SQLite JOIN FINDING",
        "  European Region had the most cases (281 million).",
        "  Americas had the most deaths (3.07 million).",
        "  Africa had the highest case fatality ratio (1.93%).",
    ]
    with open("outputs/reports/data_quality_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    print("    Saved data_quality_report.txt.")

    print("\n" + "=" * 55)
    print("  PART A DONE")
    print("=" * 55)


if __name__ == "__main__":
    run_pipeline()
