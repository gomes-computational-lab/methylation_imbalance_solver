#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from pathlib import Path
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# ============================================================
# Output directory
# ============================================================

output_dir = SCRIPT_DIR / "output_files3/ROS/Meth"
output_dir.mkdir(parents=True, exist_ok=True)

print("Current working directory:", Path.cwd())
print("Output directory:", output_dir.resolve())


# ============================================================
# Load and clean dataset
# ============================================================

file = DATA_DIR / "BetaData_SimpleImpute_Zero.csv"

df = pd.read_csv(file)

# Remove accidental saved-index columns
df = df.drop(
    columns=[
        col for col in df.columns
        if str(col).startswith("Unnamed:")
        or str(col).lower() in {"index", "level_0"}
    ],
    errors="ignore"
)

target_names = {
    0: "normal",
    1: "tumor"
}

df["target"] = df["is_tumor"].map(target_names)

df = df.drop(
    columns=["is_tumor", "Donor_Sample"],
    errors="raise"
)

X_original = df.drop(columns="target")
y_original = df["target"]

print("Dataframe shape:", df.shape)
print("X_original shape:", X_original.shape)
print("First five predictors:", X_original.columns[:5].tolist())
print("Last five predictors:", X_original.columns[-5:].tolist())
print("Original target distribution:")
print(y_original.value_counts())


# ============================================================
# Run ROS, RF, and ANOVA across 10 runs
# ============================================================

for i in range(1, 11):

    print("\n==============================")
    print(f"Starting ROS run {i}")
    print("==============================")

    ros = RandomOverSampler(
        sampling_strategy=0.5,
        random_state=i
    )

    X_resampled, y_resampled = ros.fit_resample(
        X_original,
        y_original
    )

    X = pd.DataFrame(
        X_resampled,
        columns=X_original.columns
    )

    y = pd.Series(
        y_resampled,
        name="target"
    )

    # Reproducible shuffle
    combined = X.copy()
    combined["target"] = y.values

    combined = combined.sample(
        frac=1,
        random_state=i
    ).reset_index(drop=True)

    X = combined.drop(columns="target")
    y = combined["target"]

    print("Resampled target distribution:")
    print(y.value_counts())

    # ========================================================
    # Random Forest statistics
    # ========================================================

    print("Running Random Forest")

    rf_model = RandomForestClassifier(
        n_estimators=500,
        random_state=i,
        n_jobs=-1
    )

    rf_model.fit(X, y)

    rf_results = pd.DataFrame({
        "CpG_Marker": X.columns,
        "RF_Importance": rf_model.feature_importances_
    })

    rf_results["Method"] = "ROS"
    rf_results["Run"] = i

    rf_results = rf_results.sort_values(
        "RF_Importance",
        ascending=False
    )

    # Save all RF importance values
    rf_results.to_csv(
        output_dir / f"Meth_RF_Statistics_Run{i}.csv",
        index=False
    )

    # Preserve old selected-feature output
    rf_results.loc[
        rf_results["RF_Importance"] > 0,
        ["CpG_Marker"]
    ].to_csv(
        output_dir / f"Meth_Impt_Features{i}RF.csv",
        index=False,
        header=False
    )

    # ========================================================
    # ANOVA statistics
    # ========================================================

    print("Running ANOVA")

    fvals, pvals = f_classif(X, y)

    anova_results = pd.DataFrame({
        "CpG_Marker": X.columns,
        "ANOVA_F": fvals,
        "ANOVA_P": pvals
    })

    anova_results["Method"] = "ROS"
    anova_results["Run"] = i

    # Handle constant-feature warnings or invalid values
    anova_results["ANOVA_F"] = (
        anova_results["ANOVA_F"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    anova_results["ANOVA_P"] = (
        anova_results["ANOVA_P"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )

    anova_results = anova_results.sort_values(
        "ANOVA_F",
        ascending=False
    )

    # Save all ANOVA statistics
    anova_results.to_csv(
        output_dir / f"Meth_ANOVA_Statistics_Run{i}.csv",
        index=False
    )

    # Preserve old selected-feature output
    anova_results.loc[
        anova_results["ANOVA_P"] < 0.05,
        ["CpG_Marker"]
    ].to_csv(
        output_dir / f"Meth_Impt_Features{i}Anova.csv",
        index=False,
        header=False
    )

    print(f"Completed ROS run {i}")
