#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif


# ============================================================
# Output directory
# ============================================================

output_dir = Path("output_files3/NoSmp/Meth")
output_dir.mkdir(parents=True, exist_ok=True)

print("Current working directory:", Path.cwd())
print("Output directory:", output_dir.resolve())


# ============================================================
# Load and clean dataset
# ============================================================

file = (
    "../../Final/Main Code/Preprocessing/"
    "Methylation_Imputation/BetaData_SimpleImpute_Zero.csv"
)

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

X = meth_df.drop(columns="target")
y = meth_df["target"]

print("Dataset shape:", meth_df.shape)
print("Predictor shape:", X.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nFirst five predictors:")
print(X.columns[:5].tolist())

print("\nLast five predictors:")
print(X.columns[-5:].tolist())


# ============================================================
# Random Forest: 10 runs
# ============================================================

for run in range(1, 11):

    print("\n==============================")
    print(f"Starting No Sampling RF run {run}")
    print("==============================")

    rf_model = RandomForestClassifier(
        n_estimators=500,
        random_state=run,
        n_jobs=-1
    )

    rf_model.fit(X, y)

    rf_results = pd.DataFrame({
        "CpG_Marker": X.columns,
        "RF_Importance": rf_model.feature_importances_,
        "Method": "NoSmp",
        "Run": run
    })

    rf_results = rf_results.sort_values(
        "RF_Importance",
        ascending=False
    )

    # Save all continuous RF importance values
    rf_results.to_csv(
        output_dir / f"Meth_RF_Statistics_Run{run}.csv",
        index=False
    )

    # Optional: preserve a selected-feature output
    rf_results.loc[
        rf_results["RF_Importance"] > 0,
        ["CpG_Marker"]
    ].to_csv(
        output_dir / f"Meth_Impt_Features{run}RF.csv",
        index=False,
        header=False
    )

    print(
        "RF-positive CpGs:",
        int((rf_results["RF_Importance"] > 0).sum())
    )


# ============================================================
# ANOVA: one deterministic run
# ============================================================

print("\n==============================")
print("Running No Sampling ANOVA")
print("==============================")

fvals, pvals = f_classif(X, y)

anova_results = pd.DataFrame({
    "CpG_Marker": X.columns,
    "ANOVA_F": fvals,
    "ANOVA_P": pvals,
    "Method": "NoSmp",
    "Run": 1
})

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
    output_dir / "Meth_ANOVA_Statistics_Run1.csv",
    index=False
)

# Optional: preserve selected-feature output
anova_results.loc[
    anova_results["ANOVA_P"] < 0.05,
    ["CpG_Marker"]
].to_csv(
    output_dir / "Meth_Impt_Features1Anova.csv",
    index=False,
    header=False
)

print(
    "ANOVA-significant CpGs:",
    int((anova_results["ANOVA_P"] < 0.05).sum())
)

print("\nNo Sampling analysis completed.")