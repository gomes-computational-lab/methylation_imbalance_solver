#!/usr/bin/env python3
"""Compare imputation methods and write completed methylation datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INPUT = DATA_DIR / "BetaData_AllRounded.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR

ID_COLUMN = "Donor_Sample"
TARGET_COLUMN = "is_tumor"
RANDOM_STATE = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove highly incomplete CpG markers, compare three imputation "
            "methods, and save an imputed dataset for each method."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=30.0,
        help="Drop CpG columns with this percentage of missing values or more.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--metric",
        choices=("balanced_accuracy", "regression_mse"),
        default="balanced_accuracy",
        help=(
            "Evaluation method: classification balanced accuracy or random-forest "
            "regression MSE (default: balanced_accuracy)."
        ),
    )
    return parser.parse_args()


def load_data(input_path: Path) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    dataframe = pd.read_csv(input_path, na_values=[""])
    dataframe = dataframe.drop(
        columns=[
            column
            for column in dataframe.columns
            if str(column).startswith("Unnamed:")
            or str(column).lower() in {"index", "level_0"}
        ],
        errors="ignore",
    )

    required = {ID_COLUMN, TARGET_COLUMN}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError(f"{TARGET_COLUMN} contains missing values")
    if dataframe[ID_COLUMN].isna().any():
        raise ValueError(f"{ID_COLUMN} contains missing values")

    return dataframe


def prepare_features(
    dataframe: pd.DataFrame, missing_threshold: float
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if not 0 <= missing_threshold <= 100:
        raise ValueError("--missing-threshold must be between 0 and 100")

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in {ID_COLUMN, TARGET_COLUMN}
    ]
    if not feature_columns:
        raise ValueError("Input contains no CpG feature columns")

    features = dataframe.loc[:, feature_columns].apply(pd.to_numeric, errors="raise")
    missing_percent = features.isna().mean().mul(100)
    retained_columns = missing_percent[missing_percent < missing_threshold].index
    retained = features.loc[:, retained_columns]

    if retained.empty:
        raise ValueError("No CpG columns remain after missing-value filtering")

    missingness = pd.DataFrame(
        {
            "CpG_Marker": feature_columns,
            "Missing_Count": features.isna().sum().to_numpy(),
            "Missing_Percent": missing_percent.to_numpy(),
            "Retained": missing_percent.lt(missing_threshold).to_numpy(),
        }
    )

    return retained, dataframe[TARGET_COLUMN], missingness


def make_imputers() -> dict[str, object]:
    return {
        "Zero": SimpleImputer(strategy="constant", fill_value=0),
        "KNN": KNNImputer(),
        "Mean": SimpleImputer(strategy="mean"),
    }


def validate_cross_validation(
    y: pd.Series, cv_folds: int, metric: str
) -> None:
    if cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")
    if len(y) < cv_folds:
        raise ValueError(
            f"The dataset needs at least {cv_folds} samples for "
            f"{cv_folds}-fold cross-validation"
        )

    if metric == "balanced_accuracy":
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            raise ValueError(f"{TARGET_COLUMN} must contain at least two classes")
        if class_counts.min() < cv_folds:
            raise ValueError(
                f"Each target class needs at least {cv_folds} samples for "
                f"{cv_folds}-fold cross-validation; counts are "
                f"{class_counts.to_dict()}"
            )


def evaluate_imputers(
    X: pd.DataFrame,
    y: pd.Series,
    imputers: dict[str, object],
    cv_folds: int,
    metric: str,
) -> pd.DataFrame:
    validate_cross_validation(y, cv_folds, metric)

    if metric == "balanced_accuracy":
        estimator = RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
        scoring = "balanced_accuracy"
        cross_validation = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    else:
        estimator = RandomForestRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        scoring = "neg_mean_squared_error"
        cross_validation = KFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    rows = []
    for name, imputer in imputers.items():
        model = make_pipeline(imputer, estimator)
        scores = cross_val_score(
            model,
            X,
            y,
            scoring=scoring,
            cv=cross_validation,
        )
        if metric == "regression_mse":
            scores = -scores
        rows.append(
            {
                "Imputation_Method": name,
                "Metric": metric,
                "Mean_Score": scores.mean(),
                "Std_Score": scores.std(),
            }
        )
        label = "balanced accuracy" if metric == "balanced_accuracy" else "MSE"
        print(f"{name}: {label} = {scores.mean():.4f} +/- {scores.std():.4f}")

    return pd.DataFrame(rows)


def write_imputed_datasets(
    dataframe: pd.DataFrame,
    X: pd.DataFrame,
    imputers: dict[str, object],
    output_dir: Path,
) -> None:
    for name, imputer in imputers.items():
        transformed = imputer.fit_transform(X)
        imputed_features = pd.DataFrame(
            transformed,
            columns=X.columns,
            index=dataframe.index,
        )
        output = pd.concat(
            [
                dataframe[[ID_COLUMN]],
                imputed_features,
                dataframe[[TARGET_COLUMN]],
            ],
            axis=1,
        )
        output_path = output_dir / f"BetaData_SimpleImpute_{name}.csv"
        output.to_csv(output_path, index=False)
        print(f"Wrote {output_path} ({output.shape[0]} rows, {output.shape[1]} columns)")


def plot_missingness(missingness: pd.DataFrame, output_path: Path) -> None:
    retained = missingness.loc[missingness["Retained"], "Missing_Percent"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(missingness["Missing_Percent"], bins=10)
    axes[0].set_title("Before CpG filtering")
    axes[1].hist(retained, bins=10)
    axes[1].set_title("After CpG filtering")
    for axis in axes:
        axis.set_xlabel("Missing values (%)")
        axis.set_ylabel("CpG markers")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_comparison(
    results: pd.DataFrame, metric: str, output_path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(
        results["Imputation_Method"],
        results["Mean_Score"],
        xerr=results["Std_Score"],
        alpha=0.7,
    )
    if metric == "balanced_accuracy":
        axis.set_xlabel("Balanced accuracy")
        axis.set_xlim(0, 1)
    else:
        axis.set_xlabel("Mean squared error (lower is better)")
    axis.set_title("Methylation imputation comparison")
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = load_data(input_path)
    X, y, missingness = prepare_features(dataframe, args.missing_threshold)

    retained_count = int(missingness["Retained"].sum())
    dropped_count = len(missingness) - retained_count
    print(f"Loaded {len(dataframe)} samples and {len(missingness)} CpG markers")
    print(
        f"Retained {retained_count} CpG markers and dropped {dropped_count} "
        f"at the {args.missing_threshold:g}% missing-value threshold"
    )
    print(f"Missing values to impute: {int(X.isna().sum().sum())}")
    print(f"Target counts: {y.value_counts().to_dict()}")

    missingness.to_csv(output_dir / "CpG_Missingness.csv", index=False)
    plot_missingness(missingness, output_dir / "Missingness_Distribution.png")

    imputers = make_imputers()
    results = evaluate_imputers(X, y, imputers, args.cv_folds, args.metric)
    results.to_csv(output_dir / "Imputation_Comparison.csv", index=False)
    plot_comparison(results, args.metric, output_dir / "Imputation_Comparison.png")
    write_imputed_datasets(dataframe, X, imputers, output_dir)


if __name__ == "__main__":
    main()
