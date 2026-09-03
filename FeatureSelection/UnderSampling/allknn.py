#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from pathlib import Path
from imblearn.under_sampling import AllKNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# ============================================================
# Output directory
# ============================================================

output_dir = SCRIPT_DIR / "output_files3/AllKNN/Meth"
output_dir.mkdir(parents=True, exist_ok=True)

print("Current working directory:", Path.cwd())
print("Output directory:", output_dir.resolve())


# ============================================================
# Load and clean dataset
# ============================================================

file = DATA_DIR / "BetaData_SimpleImpute_Zero.csv"

meth_df = pd.read_csv(file)

# Remove accidental saved-index columns
meth_df = meth_df.drop(
    columns=[
        col for col in meth_df.columns
        if str(col).startswith("Unnamed:")
        or str(col).lower() in {"index", "level_0"}
    ],
    errors="ignore"
)

target_names = {
    0: "normal",
    1: "tumor"
}

meth_df["target"] = meth_df["is_tumor"].map(target_names)

meth_df = meth_df.drop(
    columns=["is_tumor", "Donor_Sample"],
    errors="raise"
)

X_original = meth_df.drop(columns="target")
y_original = meth_df["target"]

print("Dataframe shape:", meth_df.shape)
print("X_original shape:", X_original.shape)
print("First five predictors:", X_original.columns[:5].tolist())
print("Last five predictors:", X_original.columns[-5:].tolist())

print("\nOriginal target distribution:")
print(y_original.value_counts())


# ============================================================
# Run AllKNN, RF, and ANOVA across 10 runs
# ============================================================

for i in range(1, 11):

    print("\n==============================")
    print(f"Starting AllKNN run {i}")
    print("==============================")

    allknn = AllKNN(
        n_neighbors=3,
        kind_sel="all",
        allow_minority=False
    )

    X_resampled, y_resampled = allknn.fit_resample(
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

    # Reproducible shuffle for downstream model fitting
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

    normal_count = y.value_counts().get("normal", 0)
    tumor_count = y.value_counts().get("tumor", 0)

    # Save compact run summary
    run_summary = pd.DataFrame({
        "Method": ["AllKNN"],
        "Run": [i],
        "N_Neighbors": [3],
        "Kind_Sel": ["all"],
        "Normal_Count": [normal_count],
        "Tumor_Count": [tumor_count],
        "Total_Count": [len(y)]
    })

    run_summary.to_csv(
        output_dir / f"AllKNN_Run_Summary_{i}.csv",
        index=False
    )

    # Save retained original row indices for reproducibility
    if hasattr(allknn, "sample_indices_"):
        selected_indices = allknn.sample_indices_

        sample_manifest = pd.DataFrame({
            "Original_Row_Index": selected_indices,
            "Target": y_original.iloc[selected_indices].to_numpy(),
            "Method": "AllKNN",
            "Run": i
        })

        sample_manifest.to_csv(
            output_dir / f"AllKNN_Selected_Samples_Run{i}.csv",
            index=False
        )

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

    rf_results["Method"] = "AllKNN"
    rf_results["Run"] = i

    rf_results = rf_results.sort_values(
        "RF_Importance",
        ascending=False
    )

    # Save all continuous RF importance values
    rf_results.to_csv(
        output_dir / f"Meth_RF_Statistics_Run{i}.csv",
        index=False
    )

    # Preserve selected-feature output used by the old pipeline
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

    anova_results["Method"] = "AllKNN"
    anova_results["Run"] = i

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

    # Save all continuous ANOVA statistics
    anova_results.to_csv(
        output_dir / f"Meth_ANOVA_Statistics_Run{i}.csv",
        index=False
    )

    # Preserve selected-feature output used by the old pipeline
    anova_results.loc[
        anova_results["ANOVA_P"] < 0.05,
        ["CpG_Marker"]
    ].to_csv(
        output_dir / f"Meth_Impt_Features{i}Anova.csv",
        index=False,
        header=False
    )

    print(f"Completed AllKNN run {i}")
