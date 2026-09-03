import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

output_dir = SCRIPT_DIR / "output_files3/Meth"
output_dir.mkdir(parents=True, exist_ok=True)

print("Current working directory:", Path.cwd())
print("Output directory:", output_dir.resolve())

# get dataset
file = DATA_DIR / "BetaData_SimpleImpute_Zero.csv"
# Read dataset
rna_df = pd.read_csv(file)

# 2. Remove accidental saved-index columns
rna_df = rna_df.drop(
    columns=[
        col for col in rna_df.columns
        if str(col).startswith("Unnamed:")
        or str(col).lower() in {"index", "level_0"}
    ],
    errors="ignore"
)


target_names = {
    0: "normal",
    1: "tumor"
}

rna_df["target"] = rna_df["is_tumor"].map(target_names)

# 4. Remove non-feature columns
rna_df = rna_df.drop(
    columns=["is_tumor", "Donor_Sample"],
    errors="raise"
)

# 5. Recreate X_original and y_original AFTER cleaning
X_original = rna_df.drop(columns="target")
y_original = rna_df["target"]

print("rna_df shape:", rna_df.shape)
print("X_original shape:", X_original.shape)
print("First five predictors:", X_original.columns[:5].tolist())
print("Last five predictors:", X_original.columns[-5:].tolist())
print("Target distribution:")
print(y_original.value_counts())


for i in range(1, 11):

    smote = SMOTE(
        k_neighbors=3,
        sampling_strategy=0.5,
        random_state=i
    )

    X_resampled, y_resampled = smote.fit_resample(
        X_original,
        y_original
    )

    X = pd.DataFrame(
        X_resampled,
        columns=X_original.columns
    )

    y = pd.Series(y_resampled, name="target")

    # Optional reproducible shuffle
    combined = X.copy()
    combined["target"] = y.values

    combined = combined.sample(
        frac=1,
        random_state=i
    ).reset_index(drop=True)

    X = combined.drop(columns="target")
    y = combined["target"]

    # -----------------------
    # Random Forest statistics
    # -----------------------
    rf_model = RandomForestClassifier(
        n_estimators=500,
        random_state=i,
        n_jobs=-1
    )

    rf_model.fit(X, y)

    rf_results = pd.DataFrame({
        "CpG_Marker": X.columns,
        "RF_Importance": rf_model.feature_importances_
    }).sort_values(
        "RF_Importance",
        ascending=False
    )

    rf_results.to_csv(
        output_dir / f"Meth_RF_Statistics_Run{i}.csv",
        index=False
    )

    rf_results.loc[
        rf_results["RF_Importance"] > 0,
        ["CpG_Marker"]
    ].to_csv(
        output_dir / f"Meth_Impt_Features{i}RF.csv",
        index=False,
        header=False
    )

    # -----------------------
    # ANOVA statistics
    # -----------------------
    fvals, pvals = f_classif(X, y)

    anova_results = pd.DataFrame({
        "CpG_Marker": X.columns,
        "ANOVA_F": fvals,
        "ANOVA_P": pvals
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

    anova_results.to_csv(
        output_dir / f"Meth_ANOVA_Statistics_Run{i}.csv",
        index=False
    )

    anova_results.loc[
        anova_results["ANOVA_P"] < 0.05,
        ["CpG_Marker"]
    ].to_csv(
        output_dir / f"Meth_Impt_Features{i}Anova.csv",
        index=False,
        header=False
    )
