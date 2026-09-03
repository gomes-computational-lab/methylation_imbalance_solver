#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from pathlib import Path
from imblearn.under_sampling import ClusterCentroids
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# ============================================================
# Output directory
# ============================================================

output_dir = SCRIPT_DIR / "output_files3/ClusterCentroids/Meth"
output_dir.mkdir(parents=True, exist_ok=True)

print("Current working directory:", Path.cwd())
print("Output directory:", output_dir.resolve())


# ============================================================
# Load and clean dataset
# ============================================================

file = DATA_DIR / "BetaData_SimpleImpute_Zero.csv"

meth_df = pd.read_csv(file)

# Remove accidental index columns
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
# Run Cluster Centroids, RF, and ANOVA across 10 runs
# ============================================================

for i in range(1, 11):

    print("\n========================================")
    print(f"Starting Cluster Centroids run {i}")
    print("========================================")

    # sampling_strategy=0.5 gives approximately:
    # 11 normal samples and 22 majority-class tumor centroids
    kmeans = KMeans(
        n_init=10,
        random_state=i
    )

    cc = ClusterCentroids(
        sampling_strategy=0.5,
        estimator=kmeans,
        random_state=i
    )

    X_resampled, y_resampled = cc.fit_resample(
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

    normal_count = y.value_counts().get("normal", 0)
    tumor_count = y.value_counts().get("tumor", 0)

    if normal_count != 11:
        raise ValueError(
            f"Run {i}: expected 11 normal samples, "
            f"but found {normal_count}."
        )

    if tumor_count != 22:
        raise ValueError(
            f"Run {i}: expected 22 tumor centroids, "
            f"but found {tumor_count}."
        )

    # ========================================================
    # Save resampled centroid dataset
    # ========================================================

    # These rows include the retained normal samples and the
    # synthetic majority-class tumor centroids.
    resampled_output = X.copy()
    resampled_output["target"] = y.values
    resampled_output["Method"] = "ClusterCentroids"
    resampled_output["Run"] = i

    # This file will be extremely large because it contains
    # every CpG marker. Keep this export only if needed.
    #
    # resampled_output.to_csv(
    #     output_dir / f"ClusterCentroids_Resampled_Run{i}.csv",
    #     index=False
    # )

    # Save a compact run summary instead
    run_summary = pd.DataFrame({
        "Method": ["ClusterCentroids"],
        "Run": [i],
        "Normal_Count": [normal_count],
        "Tumor_Centroid_Count": [tumor_count],
        "Total_Count": [len(y)],
        "KMeans_N_Init": [10],
        "Random_State": [i]
    })

    run_summary.to_csv(
        output_dir / f"ClusterCentroids_Run_Summary_{i}.csv",
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

    rf_results["Method"] = "ClusterCentroids"
    rf_results["Run"] = i

    rf_results = rf_results.sort_values(
        "RF_Importance",
        ascending=False
    )

    # Save all continuous importance values
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

    anova_results["Method"] = "ClusterCentroids"
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

    # Preserve old selected-feature output
    anova_results.loc[
        anova_results["ANOVA_P"] < 0.05,
        ["CpG_Marker"]
    ].to_csv(
        output_dir / f"Meth_Impt_Features{i}Anova.csv",
        index=False,
        header=False
    )

    print(f"Completed Cluster Centroids run {i}")
