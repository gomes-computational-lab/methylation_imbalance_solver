#!/usr/bin/env python3
"""Generate PDAC methylation manuscript and supplementary figures.

This script uses existing outputs in ``continuous_enrichment_results``. It
does not rerun methylation, feature-selection, or enrichment analyses.
"""

# Part 1 — Configuration, utilities, validation, and data loading

from datetime import datetime
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Patch
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "continuous_enrichment_results"
FIGURE_ROOT = RESULTS_DIR / "figures"
MANUSCRIPT_DIR = FIGURE_ROOT / "manuscript"
SUPPLEMENTARY_DIR = FIGURE_ROOT / "supplementary"
SUMMARY_TABLE_DIR = RESULTS_DIR / "summary_tables"
for d in [MANUSCRIPT_DIR, SUPPLEMENTARY_DIR, SUMMARY_TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FILES = {
    "rankings": RESULTS_DIR / "Continuous_Gene_Rankings_All_Methods.csv",
    "gene_stats": RESULTS_DIR / "Gene_Statistics_Across_Runs.csv",
    "hypergeom": RESULTS_DIR / "Hypergeometric_PDAC_Gene_Enrichment.csv",
    "pathway_nes": RESULTS_DIR / "PDAC_Pathway_NES_Summary.csv",
    "ties": RESULTS_DIR / "Ranking_Tie_Diagnostics.csv",
    "selected_summary": RESULTS_DIR / "Selected_Gene_Set_Summary.csv",
    "stability_summary": RESULTS_DIR / "Supplementary_Feature_Stability_Summary.csv",
}
GSEA_FULL_DIR = RESULTS_DIR / "gsea_full_ranked"
GSEA_SENSITIVITY_DIR = RESULTS_DIR / "gsea_cutoff_sensitivity"
METHOD_ORDER = ["NoSmp", "ROS", "SMOTE", "ADASYN", "RUS", "ClusterCentroids", "AllKNN"]
METHOD_LABELS = {
    "NoSmp": "No sampling",
    "ROS": "ROS",
    "SMOTE": "SMOTE",
    "ADASYN": "ADASYN",
    "RUS": "RUS",
    "ClusterCentroids": "Cluster centroids",
    "AllKNN": "AllKNN",
}
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
FIGURE_MANIFEST = []
GENERATION_START = datetime.now()
print(f"Results directory: {RESULTS_DIR}")


def validate_required_files(file_map):
    missing = [str(p) for p in file_map.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:- " + "- ".join(missing))
    print("All required summary files were found.")


def ordered_method_categories(series):
    return pd.Categorical(series, categories=METHOD_ORDER, ordered=True)


def save_figure(fig, filename_stem, description, manuscript=True, dpi=300):
    output_dir = MANUSCRIPT_DIR if manuscript else SUPPLEMENTARY_DIR
    outputs = []
    for ext in ["png", "pdf", "svg"]:
        p = output_dir / f"{filename_stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        outputs.append(str(p))
    FIGURE_MANIFEST.append(
        {
            "Figure": filename_stem,
            "Description": description,
            "Category": "Main manuscript" if manuscript else "Supplementary",
            "PNG": outputs[0],
            "PDF": outputs[1],
            "SVG": outputs[2],
            "DPI": dpi,
            "Generated": datetime.now().isoformat(timespec="seconds"),
        }
    )
    print(f"Saved {filename_stem}")


def save_table(df, filename_stem, index=False):
    csv = SUMMARY_TABLE_DIR / f"{filename_stem}.csv"
    df.to_csv(csv, index=index)
    try:
        df.to_excel(SUMMARY_TABLE_DIR / f"{filename_stem}.xlsx", index=index)
    except Exception as exc:
        print(f"Excel export skipped: {exc}")
    try:
        (SUMMARY_TABLE_DIR / f"{filename_stem}.tex").write_text(
            df.to_latex(index=index), encoding="utf-8"
        )
    except Exception as exc:
        print(f"LaTeX export skipped: {exc}")
    return csv


def add_value_labels(ax, orientation="horizontal", fmt="{:,.0f}", padding=3):
    for patch in ax.patches:
        if orientation == "horizontal":
            v = patch.get_width()
            xy = (v, patch.get_y() + patch.get_height() / 2)
            offset = (padding, 0)
            va = "center"
            ha = "left"
        else:
            v = patch.get_height()
            xy = (patch.get_x() + patch.get_width() / 2, v)
            offset = (0, padding)
            va = "bottom"
            ha = "center"
        ax.annotate(
            fmt.format(v),
            xy,
            xytext=offset,
            textcoords="offset points",
            va=va,
            ha=ha,
            fontsize=8,
        )


def selected_gene_mask(df):
    return (df["RF_Supported_Runs"] > 0) & (df["ANOVA_Significant_Runs"] > 0)


def safe_neg_log10(values):
    values = np.asarray(values, dtype=float)
    pos = values[values > 0]
    floor = pos.min() / 10 if len(pos) else np.finfo(float).tiny
    return -np.log10(np.where(values <= 0, floor, values))


validate_required_files(FILES)

rankings = pd.read_csv(FILES["rankings"])
gene_stats = pd.read_csv(FILES["gene_stats"])
hypergeom = pd.read_csv(FILES["hypergeom"])
pathway_nes = pd.read_csv(FILES["pathway_nes"])
tie_diag = pd.read_csv(FILES["ties"])
selected_summary = pd.read_csv(FILES["selected_summary"])
stability_summary = pd.read_csv(FILES["stability_summary"])
for df in [
    rankings,
    gene_stats,
    hypergeom,
    tie_diag,
    selected_summary,
    stability_summary,
]:
    if "Method" in df.columns:
        df["Method"] = df["Method"].astype(str).str.strip()
        df["Method"] = ordered_method_categories(df["Method"])
rankings["Gene"] = rankings["Gene"].astype(str).str.strip()
gene_stats["Gene"] = gene_stats["Gene"].astype(str).str.strip()
rankings = rankings.sort_values(["Method", "Score"], ascending=[True, False])
print(
    {
        "rankings": rankings.shape,
        "gene_stats": gene_stats.shape,
        "hypergeom": hypergeom.shape,
        "pathway_nes": pathway_nes.shape,
    }
)

dataset_summary = pd.DataFrame(
    {
        "Metric": [
            "Methods",
            "Unique genes in combined rankings",
            "Combined ranking rows",
            "Pathways in PDAC NES summary",
            "Full-ranked GSEA files",
            "Cutoff-sensitivity GSEA files",
        ],
        "Value": [
            rankings["Method"].nunique(),
            rankings["Gene"].nunique(),
            len(rankings),
            len(pathway_nes),
            len(list(GSEA_FULL_DIR.glob("*_Enrichment_Analysis.csv"))),
            len(list(GSEA_SENSITIVITY_DIR.glob("*_Top*_GSEA.csv"))),
        ],
    }
)
print(dataset_summary.to_string(index=False))
save_table(dataset_summary, "Dataset_Summary")


# Part 2 — Main-manuscript figures

# Figure 1 — Study workflow

fig, ax = plt.subplots(figsize=(13, 3.8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
steps = [
    ("DNA methylation", "TCGA-PAAD 450K profiles"),
    ("Preprocessing", "Filtering and imputation"),
    ("Resampling", "NoSmp, ROS, SMOTE, ADASYN,\nRUS, ClusterCentroids, AllKNN"),
    ("Feature statistics", "Random Forest importance\nand ANOVA F-statistics"),
    ("Continuous ranking", "Standardized RF and ANOVA\nstatistics combined"),
    ("Gene enrichment", "PDAC oncogene overlap and\nhypergeometric testing"),
    ("Pathway analysis", "Full-ranked and cutoff-\nsensitivity GSEA"),
]
xs = np.linspace(0.07, 0.93, len(steps))
w = 0.12
h = 0.42
for i, ((title, sub), x) in enumerate(zip(steps, xs)):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, 0.30),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.2,
            facecolor="white",
            edgecolor="black",
        )
    )
    ax.text(x, 0.57, title, ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(x, 0.41, sub, ha="center", va="center", fontsize=8)
    if i < len(steps) - 1:
        ax.add_patch(
            FancyArrowPatch(
                (x + w / 2, 0.51),
                (xs[i + 1] - w / 2, 0.51),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.2,
            )
        )
ax.set_title(
    "Study workflow for continuous gene ranking and biological enrichment",
    fontsize=14,
    pad=14,
)
save_figure(
    fig,
    "Figure1_Study_Workflow",
    "Workflow from methylation preprocessing through continuous ranking and pathway analysis.",
    True,
)
plt.show()


# Figure 2 — Selected genes by resampling method

plot_df = selected_summary.copy().sort_values("Selected_Genes")
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(plot_df["Method"].astype(str), plot_df["Selected_Genes"])
ax.set_xlabel("Number of selected genes")
ax.set_title("Selected gene-set size varies across resampling methods")
add_value_labels(ax)
ax.grid(axis="x", alpha=0.25)
save_figure(
    fig,
    "Figure2_Selected_Genes_By_Method",
    "Number of genes supported by both Random Forest and ANOVA.",
    True,
)
plt.show()


# Figure 3 — Continuous score distributions

print(rankings["Score"].describe())

print("\nMinimum:", rankings["Score"].min())
print("Maximum:", rankings["Score"].max())

print("\nPercentiles")
print(
    rankings["Score"].quantile([0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999])
)

# ============================================================
# Figure 3 — Continuous score distributions and upper-tail behavior
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(15, 5.8),
    gridspec_kw={"width_ratios": [1.2, 1]},
)

ax_a, ax_b = axes

# ------------------------------------------------------------
# Panel A: Violin + boxplot, zoomed to central 99.8%
# ------------------------------------------------------------

data = [
    rankings.loc[rankings["Method"].astype(str) == method, "Score"].dropna().values
    for method in METHOD_ORDER
]

positions = np.arange(1, len(METHOD_ORDER) + 1)

viol = ax_a.violinplot(
    data,
    positions=positions,
    widths=0.80,
    showmeans=False,
    showmedians=False,
    showextrema=False,
)

for body in viol["bodies"]:
    body.set_alpha(0.45)

box = ax_a.boxplot(
    data,
    positions=positions,
    widths=0.18,
    showfliers=False,
    patch_artist=True,
    medianprops={
        "linewidth": 1.5,
    },
)

for patch in box["boxes"]:
    patch.set_facecolor("white")

# Use the 0.1st and 99.9th percentiles so that extreme values
# do not compress the entire distribution.
lower_limit = rankings["Score"].quantile(0.001)
upper_limit = rankings["Score"].quantile(0.999)

ax_a.set_ylim(lower_limit, upper_limit)
ax_a.axhline(0, linewidth=0.8, linestyle="--")

ax_a.set_xticks(positions)
ax_a.set_xticklabels(
    [METHOD_LABELS[method] for method in METHOD_ORDER],
    rotation=30,
    ha="right",
)

ax_a.set_ylabel("Continuous gene-ranking score")
ax_a.set_title(
    "A. Distribution of continuous scores\n" "(central 99.8% of observations)",
    loc="left",
    fontweight="bold",
)

ax_a.grid(axis="y", alpha=0.25)

# Optional annotation explaining that extreme scores are omitted
ax_a.text(
    0.02,
    0.97,
    (
        "Displayed range: 0.1st–99.9th percentile\n"
        f"({lower_limit:.2f} to {upper_limit:.2f})"
    ),
    transform=ax_a.transAxes,
    ha="left",
    va="top",
    fontsize=8,
)


# ------------------------------------------------------------
# Panel B: Ranked score-decay plot for top genes
# ------------------------------------------------------------

TOP_N = 250

ranked_score_rows = []

for method in METHOD_ORDER:
    method_scores = (
        rankings.loc[rankings["Method"].astype(str) == method, ["Gene", "Score"]]
        .dropna(subset=["Score"])
        .sort_values("Score", ascending=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )

    method_scores["Rank"] = np.arange(1, len(method_scores) + 1)
    method_scores["Method"] = method

    ranked_score_rows.append(method_scores)

ranked_score_df = pd.concat(ranked_score_rows, ignore_index=True)

for method in METHOD_ORDER:
    subset = ranked_score_df[ranked_score_df["Method"] == method]

    ax_b.plot(
        subset["Rank"],
        subset["Score"],
        linewidth=2.2,
        label=METHOD_LABELS[method],
    )

# Annotate only the single highest-scoring gene across all methods
overall_top = ranked_score_df.loc[ranked_score_df["Score"].idxmax()]

ax_b.scatter(
    overall_top["Rank"],
    overall_top["Score"],
    s=28,
    zorder=5,
)

ax_b.annotate(
    f'Maximum score = {overall_top["Score"]:.1f}',
    xy=(overall_top["Rank"], overall_top["Score"]),
    xytext=(8, -2),
    textcoords="offset points",
    fontsize=7,
    va="top",
)

ax_b.set_xlabel("Gene rank")
ax_b.set_ylabel("Continuous gene-ranking score")

ax_b.set_title(
    "B. Continuous-score decay among the highest-ranked genes",
    loc="left",
    fontweight="bold",
)

# Log scaling gives more visual space to the highest-ranked genes
ax_b.set_xscale("log")
ax_b.set_xlim(1, TOP_N)

ax_b.set_xticks([1, 10, 100, 250])
ax_b.set_xticklabels(["1", "10", "100", "250"])

ax_b.grid(alpha=0.25)

ax_b.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    frameon=False,
)

save_table(ranked_score_df, "Figure3_PanelB_Top250_Ranked_Scores")

save_figure(
    fig,
    "Figure3_PanelB_Top250_Ranked_Scores",
    "Continuous score distributions and upper-tail behavior",
    True,
)

plt.show()

# =========================================================
# FIGURE 3 QUANTITATIVE SUMMARY
# =========================================================

threshold_file = SUMMARY_TABLE_DIR / "Figure3_Continuous_Score_Threshold_Data.csv"
top250_file = SUMMARY_TABLE_DIR / "Figure3_PanelB_Top250_Ranked_Scores.csv"

threshold_df = pd.read_csv(threshold_file)
top250_df = pd.read_csv(top250_file)

print("=========================================================")
print("FIGURE 3 DATA")
print("=========================================================")

print("\nThreshold data shape:", threshold_df.shape)
print("Top-250 data shape:", top250_df.shape)

print("\nThreshold columns:")
print(threshold_df.columns.tolist())

print("\nTop-250 columns:")
print(top250_df.columns.tolist())


# =========================================================
# PANEL A: SUMMARY AT SELECTED THRESHOLDS
# =========================================================

target_thresholds = [0, 1, 3, 5]

threshold_rows = []

for method in threshold_df["Method"].unique():

    temp = threshold_df[threshold_df["Method"] == method].copy()

    for target in target_thresholds:

        idx = (temp["Threshold"] - target).abs().idxmin()
        row = temp.loc[idx]

        threshold_rows.append(
            {
                "Method": method,
                "Requested_Threshold": target,
                "Actual_Threshold": float(row["Threshold"]),
                "Genes_Above_Threshold": int(row["Genes_Above_Threshold"]),
                "Percent_Above_Threshold": float(row["Percent_Above_Threshold"]),
            }
        )

threshold_summary = pd.DataFrame(threshold_rows)

print("\n\n=========================================================")
print("PANEL A: SCORE THRESHOLD SUMMARY")
print("=========================================================\n")

print(threshold_summary.round(3).to_string(index=False))


# Easier-to-read wide table
panelA_summary = threshold_summary.pivot(
    index="Method",
    columns="Requested_Threshold",
    values=["Actual_Threshold", "Genes_Above_Threshold", "Percent_Above_Threshold"],
)

print(panelA_summary.round(3).to_string())


# =========================================================
# PANEL B: SCORES AT SELECTED RANKS
# =========================================================

ranks_to_check = [1, 10, 25, 50, 100, 250]

rank_rows = []

for method in top250_df["Method"].unique():

    temp = top250_df[top250_df["Method"] == method].sort_values("Rank")

    for rank in ranks_to_check:

        match = temp[temp["Rank"] == rank]

        if not match.empty:
            rank_rows.append(
                {"Method": method, "Rank": rank, "Score": float(match.iloc[0]["Score"])}
            )

rank_summary_long = pd.DataFrame(rank_rows)

rank_summary = rank_summary_long.pivot(
    index="Method", columns="Rank", values="Score"
).reset_index()

rank_summary.columns = [
    "Method" if col == "Method" else f"Rank_{int(col)}" for col in rank_summary.columns
]


# =========================================================
# SCORE DECAY STATISTICS
# =========================================================

rank_summary["Rank10_Percent_of_Max"] = (
    rank_summary["Rank_10"] / rank_summary["Rank_1"] * 100
)

rank_summary["Rank25_Percent_of_Max"] = (
    rank_summary["Rank_25"] / rank_summary["Rank_1"] * 100
)

rank_summary["Rank50_Percent_of_Max"] = (
    rank_summary["Rank_50"] / rank_summary["Rank_1"] * 100
)

rank_summary["Rank100_Percent_of_Max"] = (
    rank_summary["Rank_100"] / rank_summary["Rank_1"] * 100
)

rank_summary["Rank250_Percent_of_Max"] = (
    rank_summary["Rank_250"] / rank_summary["Rank_1"] * 100
)

print("\n\n=========================================================")
print("PANEL B: RANKED SCORE SUMMARY")
print("=========================================================\n")

print(rank_summary.round(3).to_string(index=False))


# =========================================================
# AUTOMATIC KEY OBSERVATIONS
# =========================================================

highest_max = rank_summary.loc[rank_summary["Rank_1"].idxmax()]

highest_rank10 = rank_summary.loc[rank_summary["Rank_10"].idxmax()]

highest_rank100 = rank_summary.loc[rank_summary["Rank_100"].idxmax()]

highest_rank250 = rank_summary.loc[rank_summary["Rank_250"].idxmax()]

fastest_decay = rank_summary.loc[rank_summary["Rank10_Percent_of_Max"].idxmin()]

slowest_decay = rank_summary.loc[rank_summary["Rank10_Percent_of_Max"].idxmax()]

print("\n=========================================================")
print("KEY FIGURE 3 OBSERVATIONS")
print("=========================================================")

print(
    f"Highest maximum score: "
    f"{highest_max['Method']} "
    f"({highest_max['Rank_1']:.2f})"
)

print(
    f"Highest rank-10 score: "
    f"{highest_rank10['Method']} "
    f"({highest_rank10['Rank_10']:.2f})"
)

print(
    f"Highest rank-100 score: "
    f"{highest_rank100['Method']} "
    f"({highest_rank100['Rank_100']:.2f})"
)

print(
    f"Highest rank-250 score: "
    f"{highest_rank250['Method']} "
    f"({highest_rank250['Rank_250']:.2f})"
)

print(
    f"Fastest early decay: "
    f"{fastest_decay['Method']} retained "
    f"{fastest_decay['Rank10_Percent_of_Max']:.1f}% "
    f"of its maximum by rank 10."
)

print(
    f"Slowest early decay: "
    f"{slowest_decay['Method']} retained "
    f"{slowest_decay['Rank10_Percent_of_Max']:.1f}% "
    f"of its maximum by rank 10."
)


# =========================================================
# SAVE OUTPUTS
# =========================================================

threshold_summary.to_csv("Figure3_Threshold_Summary.csv", index=False)

rank_summary.to_csv("Figure3_Rank_Summary.csv", index=False)

print("\nSaved:")
print(" - Figure3_Threshold_Summary.csv")
print(" - Figure3_Rank_Summary.csv")


# Figure 4 — Random Forest versus ANOVA agreement

# ============================================================
# Figure 4 — Agreement between Random Forest and ANOVA
# ============================================================

agreement_parts = []
correlation_rows = []

# ------------------------------------------------------------
# Prepare all valid genes and calculate method-specific
# Pearson and Spearman correlations
# ------------------------------------------------------------

for method in METHOD_ORDER:

    method_df = (
        rankings.loc[
            rankings["Method"].astype(str) == method,
            ["Gene", "RF_Z", "ANOVA_Z", "Score"],
        ]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["RF_Z", "ANOVA_Z"])
        .copy()
    )

    method_df["Method"] = method
    agreement_parts.append(method_df)

    pearson_r, pearson_p = stats.pearsonr(method_df["RF_Z"], method_df["ANOVA_Z"])

    spearman_rho, spearman_p = stats.spearmanr(method_df["RF_Z"], method_df["ANOVA_Z"])

    correlation_rows.append(
        {
            "Method": method,
            "N_Genes": len(method_df),
            "Pearson_r": pearson_r,
            # "Pearson_p": pearson_p,
            "Spearman_rho": spearman_rho,
            # "Spearman_p": spearman_p,
        }
    )

agreement_df = pd.concat(agreement_parts, ignore_index=True)

correlation_table = pd.DataFrame(correlation_rows)

correlation_table["Method"] = pd.Categorical(
    correlation_table["Method"], categories=METHOD_ORDER, ordered=True
)

correlation_table = correlation_table.sort_values("Method").reset_index(drop=True)

print(correlation_table.to_string(index=False))

save_table(correlation_table, "Figure4_RF_ANOVA_Correlations_All_Genes")


# ------------------------------------------------------------
# Create two-panel figure
# ------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=1, ncols=2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.15, 1]}
)

ax_a, ax_b = axes


# ============================================================
# Panel A — Hexbin density plot for No Sampling baseline
# ============================================================

baseline_df = agreement_df[agreement_df["Method"].astype(str) == "NoSmp"].copy()

hexbin = ax_a.hexbin(
    baseline_df["RF_Z"],
    baseline_df["ANOVA_Z"],
    gridsize=55,
    mincnt=1,
    bins="log",
    linewidths=0.15,
)

colorbar = fig.colorbar(hexbin, ax=ax_a, fraction=0.046, pad=0.03)

colorbar.set_label("Gene density (log scale)")


# Retrieve No Sampling correlation values
baseline_stats = correlation_table[
    correlation_table["Method"].astype(str) == "NoSmp"
].iloc[0]


# Add correlation summary
ax_a.text(
    0.97,
    0.96,
    (
        f'Genes: {int(baseline_stats["N_Genes"]):,}\n'
        f'Spearman ρ = {baseline_stats["Spearman_rho"]:.3f}\n'
        f'Pearson r = {baseline_stats["Pearson_r"]:.3f}'
    ),
    transform=ax_a.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    bbox={
        "boxstyle": "round,pad=0.35",
        "facecolor": "white",
        "edgecolor": "0.7",
        "alpha": 0.9,
    },
)

ax_a.axhline(
    0,
    linewidth=0.8,
    linestyle=":",
)

ax_a.axvline(
    0,
    linewidth=0.8,
    linestyle=":",
)

ax_a.set_xlabel("Random Forest standardized statistic (RF_Z)")

ax_a.set_ylabel("ANOVA standardized statistic (ANOVA_Z)")

ax_a.set_title(
    "A. RF–ANOVA relationship without resampling", loc="left", fontweight="bold"
)

ax_a.grid(alpha=0.15)


# ============================================================
# Panel B — Pearson and Spearman correlations by method
# ============================================================

x_positions = np.arange(len(METHOD_ORDER))

bar_width = 0.36

pearson_bars = ax_b.bar(
    x_positions - bar_width / 2,
    correlation_table["Pearson_r"],
    width=bar_width,
    label="Pearson r",
    color="lightgray",
)

spearman_bars = ax_b.bar(
    x_positions + bar_width / 2,
    correlation_table["Spearman_rho"],
    width=bar_width,
    label="Spearman ρ",
    color="steelblue",
)


# Add numerical labels above positive bars and below negative bars
for bars in [pearson_bars, spearman_bars]:

    for bar in bars:

        height = bar.get_height()

        vertical_offset = 3 if height >= 0 else -4
        vertical_alignment = "bottom" if height >= 0 else "top"

        ax_b.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, vertical_offset),
            textcoords="offset points",
            ha="center",
            va=vertical_alignment,
            fontsize=8,
        )


ax_b.set_xticks(x_positions)

ax_b.set_xticklabels(
    [METHOD_LABELS[method] for method in METHOD_ORDER], rotation=30, ha="right"
)

# Let the axis support both positive and negative correlations
min_corr = min(
    correlation_table["Pearson_r"].min(), correlation_table["Spearman_rho"].min()
)

max_corr = max(
    correlation_table["Pearson_r"].max(), correlation_table["Spearman_rho"].max()
)

lower_y = min(-0.1, min_corr - 0.08)
upper_y = min(1.0, max_corr + 0.12)

ax_b.set_ylim(lower_y, upper_y)

ax_b.axhline(0, linewidth=0.9, color="black")

ax_b.set_ylabel("Correlation coefficient")

ax_b.set_title(
    "B. Method-specific agreement between Random Forest and ANOVA rankings",
    loc="left",
    fontweight="bold",
)

ax_b.grid(axis="y", alpha=0.25)

ax_b.legend(frameon=False, loc="upper left")


# ------------------------------------------------------------
# Overall formatting and saving
# ------------------------------------------------------------

fig.suptitle(
    "Agreement between machine-learning and statistical feature rankings",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)

fig.tight_layout()

save_figure(
    fig,
    "Figure4_RF_ANOVA_Agreement",
    (
        "Panel A shows the density relationship between standardized "
        "Random Forest and ANOVA statistics for all valid genes in the "
        "No Sampling baseline. Panel B compares method-specific Pearson "
        "and Spearman correlations calculated using all valid genes."
    ),
    manuscript=True,
)

plt.show()


print(
    "Spearman correlation is emphasized because both feature-selection methods produce ranked importance measures rather than strictly linear relationships."
)


# Figure 5 — Hypergeometric PDAC-gene enrichment

plot_df = hypergeom.copy()

plot_df["Minus_Log10_P"] = safe_neg_log10(plot_df["Hypergeometric_P"])

# Replace internal method names with publication-friendly labels
plot_df["Method_Label"] = plot_df["Method"].astype(str).map(METHOD_LABELS)

plot_df = plot_df.sort_values("Minus_Log10_P")

fig, ax = plt.subplots(figsize=(8, 5.2))

highlight = {"SMOTE", "NoSmp", "RUS"}

colors = [
    "steelblue" if method in highlight else "lightgray"
    for method in plot_df["Method"].astype(str)
]

ax.barh(
    plot_df["Method_Label"],
    plot_df["Minus_Log10_P"],
    color=colors,
    edgecolor="black",
    linewidth=0.5,
)

for p, k in zip(ax.patches, plot_df["PDAC_Overlap_k"]):
    ax.annotate(
        f"k={int(k)}",
        (p.get_width(), p.get_y() + p.get_height() / 2),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
    )

ax.axvline(
    -np.log10(0.05),
    linestyle="--",
    linewidth=1,
    label="P = 0.05",
)

ax.set_xlabel("−log10(hypergeometric P value)")
ax.set_title("Enrichment of curated PDAC-associated genes")

ax.legend(frameon=False)

ax.grid(axis="x", alpha=0.25)

save_figure(
    fig,
    "Figure5_Hypergeometric_PDAC_Enrichment",
    "Hypergeometric enrichment significance with PDAC overlap counts.",
    True,
)

plt.show()


# Figure 6 — PDAC pathway NES heatmap

# ============================================================
# Figure 6 — PDAC pathway enrichment heatmap
# ============================================================

# ------------------------------------------------------------
# Load or copy pathway summary data
# ------------------------------------------------------------

heatmap_df = pathway_nes.copy()


# ------------------------------------------------------------
# Rename NES columns using publication-friendly labels
# ------------------------------------------------------------

heatmap_df = heatmap_df.rename(
    columns={
        "NES (NoSmp)": "No sampling",
        "NES (ROS)": "ROS",
        "NES (SMOTE)": "SMOTE",
        "NES (ADASYN)": "ADASYN",
        "NES (RUS)": "RUS",
        "NES (ClusterCentroids)": "Cluster centroids",
        "NES (AllKNN)": "AllKNN",
    }
)

method_cols = [
    "No sampling",
    "ROS",
    "SMOTE",
    "ADASYN",
    "RUS",
    "Cluster centroids",
    "AllKNN",
]


# ------------------------------------------------------------
# Keep only pathway and NES columns
# ------------------------------------------------------------

# This removes FDR (best) and any other metadata columns.
heatmap_df = heatmap_df[["Pathway"] + method_cols].copy()

# Ensure all NES columns are numeric.
heatmap_df[method_cols] = heatmap_df[method_cols].apply(
    pd.to_numeric,
    errors="coerce",
)


# ------------------------------------------------------------
# Reorder pathways by mean NES
# ------------------------------------------------------------

heatmap_df["Mean_NES"] = heatmap_df[method_cols].mean(
    axis=1,
    skipna=True,
)

heatmap_df = (
    heatmap_df.sort_values("Mean_NES", ascending=False)
    .drop(columns="Mean_NES")
    .set_index("Pathway")
)


# ------------------------------------------------------------
# Safety check
# ------------------------------------------------------------

print("Heatmap columns:")
print(heatmap_df.columns.tolist())

print(heatmap_df.to_string())


# ------------------------------------------------------------
# Plot configuration
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10.5, 6.5))

# Fixed color limits for reproducibility.
vmin = 1.5
vmax = 2.5

# Midpoint used to switch text from white to black.
text_threshold = (vmin + vmax) / 2

# Use viridis and display missing values in light gray.
cmap = plt.cm.viridis.copy()
cmap.set_bad(color="lightgray")

heatmap_values = heatmap_df.to_numpy(dtype=float)

masked_values = np.ma.masked_invalid(heatmap_values)

image = ax.imshow(
    masked_values,
    cmap=cmap,
    aspect="auto",
    interpolation="nearest",
    vmin=vmin,
    vmax=vmax,
)


# ------------------------------------------------------------
# Axis labels
# ------------------------------------------------------------

ax.set_xticks(np.arange(len(method_cols)))

ax.set_xticklabels(
    method_cols,
    rotation=35,
    ha="right",
)

ax.set_yticks(np.arange(len(heatmap_df.index)))

ax.set_yticklabels(heatmap_df.index)

ax.set_xlabel("")
ax.set_ylabel("")

ax.set_title(
    "Comparison of PDAC pathway enrichment across resampling methods",
    fontsize=14,
    fontweight="bold",
    pad=10,
)


# ------------------------------------------------------------
# Annotate NES values
# ------------------------------------------------------------

for row_index in range(heatmap_df.shape[0]):

    row_values = heatmap_values[row_index, :]

    # Skip rows containing no reported NES values.
    if np.all(np.isnan(row_values)):
        continue

    row_max = np.nanmax(row_values)

    for column_index, value in enumerate(row_values):

        # Do not write text inside missing-value cells.
        if np.isnan(value):
            continue

        # Dark backgrounds receive white text.
        # Light backgrounds receive black text.
        text_color = "white" if value < text_threshold else "black"

        # Bold the highest NES within each pathway.
        font_weight = (
            "bold"
            if np.isclose(
                value,
                row_max,
                rtol=1e-8,
                atol=1e-8,
            )
            else "normal"
        )

        text = ax.text(
            column_index,
            row_index,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight=font_weight,
            color=text_color,
        )

        # Add a subtle contrasting outline for readability.
        outline_color = "black" if text_color == "white" else "white"

        text.set_path_effects(
            [
                pe.withStroke(
                    linewidth=1.0,
                    foreground=outline_color,
                )
            ]
        )


# ------------------------------------------------------------
# Colorbar
# ------------------------------------------------------------

colorbar = fig.colorbar(
    image,
    ax=ax,
    fraction=0.046,
    pad=0.04,
)

colorbar.set_label(
    "Normalized enrichment score (NES)",
    fontsize=11,
)


# ------------------------------------------------------------
# Cosmetic cleanup
# ------------------------------------------------------------

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(
    axis="both",
    labelsize=10,
)

fig.tight_layout()


# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------

save_figure(
    fig,
    "Figure6_PDAC_Pathway_Heatmap",
    (
        "Heatmap of normalized enrichment scores for PDAC-relevant "
        "pathways across resampling strategies. Bold values indicate "
        "the highest NES within each pathway. Gray cells indicate "
        "pathways for which no NES value was reported."
    ),
    manuscript=True,
)

plt.show()


# Part 3 — Supplementary figures

# Supplementary Figure S1 — RF support stability

# ============================================================
# Supplementary Figure S1 — RF gene-support stability
# ============================================================

support_rows = []

for method in METHOD_ORDER:
    method_values = (
        gene_stats.loc[gene_stats["Method"].astype(str) == method, "RF_Supported_Runs"]
        .dropna()
        .astype(int)
    )

    total_genes = len(method_values)

    # Collapse support counts into four interpretable categories
    support_category = pd.cut(
        method_values,
        bins=[-0.5, 0.5, 1.5, 2.5, np.inf],
        labels=["0 runs", "1 run", "2 runs", "≥3 runs"],
    )

    category_counts = support_category.value_counts().reindex(
        ["0 runs", "1 run", "2 runs", "≥3 runs"],
        fill_value=0,
    )

    for category, count in category_counts.items():
        support_rows.append(
            {
                "Method": method,
                "Support_Category": category,
                "Gene_Count": int(count),
                "Gene_Percent": (
                    100 * count / total_genes if total_genes > 0 else np.nan
                ),
            }
        )

support_df = pd.DataFrame(support_rows)

category_order = [
    "0 runs",
    "1 run",
    "2 runs",
    "≥3 runs",
]

support_pivot = (
    support_df.pivot(
        index="Method",
        columns="Support_Category",
        values="Gene_Percent",
    )
    .reindex(METHOD_ORDER)
    .reindex(columns=category_order)
    .fillna(0)
)

# Percentage supported in at least one RF run
supported_at_least_once = 100 - support_pivot["0 runs"]


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10.5, 5.8))

x_positions = np.arange(len(METHOD_ORDER))
bottom = np.zeros(len(METHOD_ORDER))

category_colors = {
    "0 runs": "lightgray",
    "1 run": "lightsteelblue",
    "2 runs": "cornflowerblue",
    "≥3 runs": "steelblue",
}

for category in category_order:
    values = support_pivot[category].to_numpy()

    ax.bar(
        x_positions,
        values,
        bottom=bottom,
        label=category,
        color=category_colors[category],
        edgecolor="white",
        linewidth=0.5,
    )

    bottom += values


# ------------------------------------------------------------
# Annotate percentage supported at least once
# ------------------------------------------------------------

for x, supported_percent in zip(
    x_positions,
    supported_at_least_once,
):
    ax.text(
        x,
        101.2,
        f"{supported_percent:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        clip_on=False,
    )


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

ax.set_xticks(x_positions)

ax.set_xticklabels(
    [METHOD_LABELS[m] for m in METHOD_ORDER],
    rotation=25,
    ha="right",
)

ax.set_ylabel("Genes within method (%)")

ax.set_ylim(0, 106)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.legend(
    title="RF-supported runs",
    frameon=False,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

ax.text(
    0.99,
    1.02,
    "Labels above bars show genes supported in ≥1 RF run",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
)

fig.tight_layout()


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

save_figure(
    fig,
    "SupplementaryFigureS1_RF_Support_Stability",
    (
        "Percentage of genes supported in zero, one, two, or at least "
        "three Random Forest runs. Labels above bars indicate the "
        "percentage supported in at least one run."
    ),
    manuscript=False,
)

plt.show()

save_table(
    support_df,
    "SupplementaryFigureS1_RF_Support_Stability_Data",
)


print("Reproducibility of Random Forest gene selection across repeated model training")


# Supplementary Figure S2 — Ranking tie diagnostics

# ============================================================
# Supplementary Figure S2 — Ranking tie diagnostics
# ============================================================

plot_df = tie_diag.copy()

plot_df["Method_Label"] = plot_df["Method"].astype(str).map(METHOD_LABELS)

plot_df = plot_df.sort_values("Percent_Genes_In_Ties")

colors = [
    "steelblue" if method == "RUS" else "lightgray"
    for method in plot_df["Method"].astype(str)
]

fig, ax = plt.subplots(figsize=(8.5, 5.2))

ax.barh(
    plot_df["Method_Label"],
    plot_df["Percent_Genes_In_Ties"],
    color=colors,
    edgecolor="black",
    linewidth=0.5,
)

for patch, value in zip(
    ax.patches,
    plot_df["Percent_Genes_In_Ties"],
):
    ax.annotate(
        f"{value:.1f}%",
        (
            patch.get_width(),
            patch.get_y() + patch.get_height() / 2,
        ),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
    )

ax.set_xlabel("Genes involved in tied continuous scores (%)")

ax.set_ylabel("")

ax.set_title(
    "Residual ranking ties after continuous-score construction",
    fontweight="bold",
)

ax.set_xlim(0, plot_df["Percent_Genes_In_Ties"].max() + 2)

ax.grid(
    axis="x",
    alpha=0.25,
)

fig.tight_layout()

save_figure(
    fig,
    "SupplementaryFigureS2_Ranking_Tie_Diagnostics",
    (
        "Percentage of genes involved in tied continuous-ranking "
        "scores for each resampling strategy. Lower values indicate "
        "greater ranking differentiation."
    ),
    manuscript=False,
)

plt.show()


# Supplementary Figure S3 — Top-gene score heatmap

# ============================================================
# Supplementary Figure S3 — Recurrent top-ranked genes
# ============================================================

TOP_PER_METHOD = 10
MAX_GENES_DISPLAYED = 20

# Add other annotation categories here if needed.
EXCLUDE_LABELS = {
    "OPENSEA",
    "",
    "NAN",
    "NONE",
}

# ------------------------------------------------------------
# Identify top genes within each method
# ------------------------------------------------------------

top_gene_records = []

for method in METHOD_ORDER:

    method_top = (
        rankings.loc[rankings["Method"].astype(str) == method, ["Gene", "Score"]]
        .dropna(subset=["Gene", "Score"])
        .sort_values("Score", ascending=False)
        .head(TOP_PER_METHOD)
        .copy()
    )

    method_top["Gene"] = method_top["Gene"].astype(str).str.strip()

    method_top = method_top[~method_top["Gene"].str.upper().isin(EXCLUDE_LABELS)]

    method_top["Method"] = method
    method_top["Top_Rank"] = np.arange(1, len(method_top) + 1)

    top_gene_records.append(method_top)

top_membership_df = pd.concat(top_gene_records, ignore_index=True)

# ------------------------------------------------------------
# Calculate true top-10 recurrence
# ------------------------------------------------------------

recurrence_df = (
    top_membership_df.groupby("Gene")
    .agg(
        Methods_In_Top10=("Method", "nunique"),
        Mean_Top10_Score=("Score", "mean"),
        Maximum_Top10_Score=("Score", "max"),
        Best_Rank=("Top_Rank", "min"),
    )
    .reset_index()
)

recurrence_df = recurrence_df.sort_values(
    [
        "Methods_In_Top10",
        "Mean_Top10_Score",
        "Maximum_Top10_Score",
    ],
    ascending=[False, False, False],
)

selected_genes = recurrence_df.head(MAX_GENES_DISPLAYED)["Gene"].tolist()

# ------------------------------------------------------------
# Build score matrix across all methods
# ------------------------------------------------------------

heat_matrix = (
    rankings.loc[rankings["Gene"].isin(selected_genes), ["Gene", "Method", "Score"]]
    .pivot_table(
        index="Gene",
        columns="Method",
        values="Score",
        aggfunc="max",
    )
    .reindex(columns=METHOD_ORDER)
)

# Preserve recurrence-based row ordering
gene_order = (
    recurrence_df[recurrence_df["Gene"].isin(selected_genes)]
    .sort_values(
        [
            "Methods_In_Top10",
            "Mean_Top10_Score",
            "Maximum_Top10_Score",
        ],
        ascending=[False, False, False],
    )["Gene"]
    .tolist()
)

heat_matrix = heat_matrix.reindex(gene_order)

# Lookup recurrence for labels
recurrence_lookup = recurrence_df.set_index("Gene")["Methods_In_Top10"]

# ------------------------------------------------------------
# Prepare export table
# ------------------------------------------------------------

heat_export = heat_matrix.reset_index().merge(
    recurrence_df,
    on="Gene",
    how="left",
)

# ------------------------------------------------------------
# Robust color scaling
# ------------------------------------------------------------

heat_values = heat_matrix.to_numpy(dtype=float)

valid_values = heat_values[np.isfinite(heat_values)]

vmin = np.nanpercentile(valid_values, 2)

vmax = np.nanpercentile(valid_values, 98)

if np.isclose(vmin, vmax):
    vmin = np.nanmin(valid_values)
    vmax = np.nanmax(valid_values)

text_threshold = (vmin + vmax) / 2

cmap = plt.cm.viridis.copy()
cmap.set_bad(color="lightgray")

masked_values = np.ma.masked_invalid(heat_values)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig_height = max(6.5, 0.36 * len(heat_matrix))

fig, ax = plt.subplots(figsize=(10.5, fig_height))

image = ax.imshow(
    masked_values,
    aspect="auto",
    interpolation="nearest",
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
)

# Method labels
ax.set_xticks(np.arange(len(METHOD_ORDER)))

ax.set_xticklabels(
    [METHOD_LABELS[method] for method in METHOD_ORDER],
    rotation=35,
    ha="right",
)

# Gene labels with true top-10 recurrence
gene_labels = [
    (f"{gene} " f"({int(recurrence_lookup.loc[gene])}/" f"{len(METHOD_ORDER)})")
    for gene in heat_matrix.index
]

ax.set_yticks(np.arange(len(heat_matrix.index)))

ax.set_yticklabels(
    gene_labels,
    fontsize=8,
)

ax.set_xlabel("")
ax.set_ylabel("")

ax.set_title(
    "Most recurrent top-10 genes across resampling methods",
    fontweight="bold",
    pad=10,
)

# ------------------------------------------------------------
# Annotate each gene's highest score
# ------------------------------------------------------------

for row_index in range(heat_matrix.shape[0]):

    row_values = heat_values[row_index, :]

    if np.all(np.isnan(row_values)):
        continue

    row_max = np.nanmax(row_values)

    for column_index, value in enumerate(row_values):

        if np.isnan(value):
            continue

        if np.isclose(
            value,
            row_max,
            rtol=1e-8,
            atol=1e-8,
        ):

            text_color = "white" if value < text_threshold else "black"

            ax.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color=text_color,
            )

# ------------------------------------------------------------
# Colorbar and formatting
# ------------------------------------------------------------

colorbar = fig.colorbar(
    image,
    ax=ax,
    fraction=0.035,
    pad=0.025,
)

colorbar.set_label("Continuous gene-ranking score")

ax.tick_params(
    axis="both",
    length=0,
)

fig.tight_layout()

# ------------------------------------------------------------
# Save figure and data
# ------------------------------------------------------------

save_figure(
    fig,
    "SupplementaryFigureS3_Recurrent_Top_Gene_Heatmap",
    (
        "Heatmap of the most recurrent top-ranked genes across "
        "resampling methods. Parenthetical values indicate the "
        "number of methods in which each gene appeared among the "
        f"top {TOP_PER_METHOD} candidates. Each row's maximum score "
        "is annotated."
    ),
    manuscript=False,
)

plt.show()

save_table(
    heat_export,
    "SupplementaryFigureS3_Recurrent_Top_Gene_Matrix",
)

save_table(
    top_membership_df,
    "SupplementaryFigureS3_Method_Specific_Top10_Genes",
)

print(recurrence_df.head(MAX_GENES_DISPLAYED).to_string(index=False))

print(
    "Note: Parenthetical values indicate the number of methods in which each gene appeared among the top 10. Heatmap cells show the continuous score under every method, including methods where the gene fell outside the top 10."
)

for method in METHOD_ORDER:
    print(
        method,
        rankings.loc[rankings["Method"] == method]
        .nlargest(10, "Score")["Gene"]
        .tolist(),
    )


# Supplementary Figure S4 — Jaccard similarity

# ============================================================
# Supplementary Figure S4 — Selected-gene-set similarity
# ============================================================

# Build selected-gene sets
selected_sets = {
    method: set(
        rankings.loc[
            (rankings["Method"].astype(str) == method) & selected_gene_mask(rankings),
            "Gene",
        ]
    )
    for method in METHOD_ORDER
}

# Compute pairwise Jaccard similarity
jaccard = pd.DataFrame(
    index=METHOD_ORDER,
    columns=METHOD_ORDER,
    dtype=float,
)

for method_a in METHOD_ORDER:
    for method_b in METHOD_ORDER:

        union_set = selected_sets[method_a] | selected_sets[method_b]

        intersection_set = selected_sets[method_a] & selected_sets[method_b]

        jaccard.loc[method_a, method_b] = (
            len(intersection_set) / len(union_set) if union_set else np.nan
        )

# ------------------------------------------------------------
# Mean off-diagonal similarity for each method
# ------------------------------------------------------------

mean_similarity = {}

for method in METHOD_ORDER:
    values = jaccard.loc[method].drop(labels=method)
    mean_similarity[method] = values.mean()

mean_similarity_df = pd.DataFrame(
    {
        "Method": METHOD_ORDER,
        "Mean_Off_Diagonal_Jaccard": [mean_similarity[m] for m in METHOD_ORDER],
    }
)

print(mean_similarity_df.to_string(index=False))

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.8, 6.8))

cmap = plt.cm.viridis.copy()

image = ax.imshow(
    jaccard.values.astype(float),
    vmin=0,
    vmax=1,
    aspect="equal",
    cmap=cmap,
)

ax.set_xticks(np.arange(len(METHOD_ORDER)))

ax.set_xticklabels(
    [METHOD_LABELS[m] for m in METHOD_ORDER],
    rotation=40,
    ha="right",
)

ax.set_yticks(np.arange(len(METHOD_ORDER)))

ax.set_yticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])

# ------------------------------------------------------------
# Annotate values
# ------------------------------------------------------------

text_threshold = 0.45

for row_index in range(len(METHOD_ORDER)):

    for column_index in range(len(METHOD_ORDER)):

        value = jaccard.iloc[row_index, column_index]

        if np.isnan(value):
            continue

        text_color = "white" if value < text_threshold else "black"

        font_weight = (
            "bold"
            if row_index != column_index
            and value == np.nanmax(np.delete(jaccard.iloc[row_index].values, row_index))
            else "normal"
        )

        ax.text(
            column_index,
            row_index,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color=text_color,
            fontweight=font_weight,
        )

# ------------------------------------------------------------
# Colorbar and title
# ------------------------------------------------------------

colorbar = fig.colorbar(
    image,
    ax=ax,
    fraction=0.04,
    pad=0.03,
)

colorbar.set_label("Jaccard similarity")

ax.set_title(
    "Similarity of selected gene sets across resampling methods",
    fontweight="bold",
    pad=10,
)

ax.tick_params(
    axis="both",
    length=0,
)

fig.tight_layout()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

save_figure(
    fig,
    "SupplementaryFigureS4_Selected_Gene_Jaccard",
    (
        "Pairwise Jaccard similarity of selected gene sets across "
        "resampling methods. Higher values indicate greater overlap "
        "between method-specific selected-gene sets."
    ),
    manuscript=False,
)

plt.show()

save_table(
    jaccard.reset_index().rename(columns={"index": "Method"}),
    "Jaccard_Similarity_Matrix",
)

save_table(
    mean_similarity_df,
    "SupplementaryFigureS4_Mean_Jaccard_By_Method",
)


# Supplementary Figure S5 — Top 20 genes per method

# ============================================================
# Supplementary Figure S5 — Top genes within each method
# ============================================================

TOP_N = 15

EXCLUDE_LABELS = {
    "OPENSEA",
    "",
    "NAN",
    "NONE",
}

top_gene_frames = []

# ------------------------------------------------------------
# Prepare top genes for each method
# ------------------------------------------------------------

for method in METHOD_ORDER:

    method_top = (
        rankings.loc[rankings["Method"].astype(str) == method, ["Gene", "Score"]]
        .dropna(subset=["Gene", "Score"])
        .copy()
    )

    method_top["Gene"] = method_top["Gene"].astype(str).str.strip()

    method_top = method_top[~method_top["Gene"].str.upper().isin(EXCLUDE_LABELS)]

    method_top = (
        method_top.sort_values("Score", ascending=False)
        .drop_duplicates(subset="Gene")
        .head(TOP_N)
        .copy()
    )

    method_top["Method"] = method
    method_top["Rank"] = np.arange(1, len(method_top) + 1)

    top_gene_frames.append(method_top)

top_genes_df = pd.concat(top_gene_frames, ignore_index=True)

# Use one shared x-axis range for every panel
global_max_score = top_genes_df["Score"].max()

# ------------------------------------------------------------
# Create 2 × 4 panel layout
# ------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(18, 11),
    sharex=True,
)

axes = axes.flatten()

for ax, method in zip(
    axes,
    METHOD_ORDER,
):

    method_df = top_genes_df[top_genes_df["Method"] == method].sort_values(
        "Score", ascending=True
    )

    bars = ax.barh(
        method_df["Gene"],
        method_df["Score"],
        color="steelblue",
        edgecolor="black",
        linewidth=0.35,
    )

    # Score labels
    for bar, score in zip(
        bars,
        method_df["Score"],
    ):
        ax.annotate(
            f"{score:.1f}",
            xy=(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
            ),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_title(
        METHOD_LABELS[method],
        loc="left",
        fontweight="bold",
        fontsize=11,
    )

    ax.set_xlim(
        0,
        global_max_score * 1.10,
    )

    ax.tick_params(
        axis="y",
        labelsize=9,
    )

    ax.grid(
        axis="x",
        alpha=0.20,
    )

    ax.set_axisbelow(True)

# Hide unused eighth panel
for unused_ax in axes[len(METHOD_ORDER) :]:
    unused_ax.axis("off")

# Shared labels and title
fig.suptitle(
    f"Top {TOP_N} continuous-ranking genes within each resampling method",
    fontsize=15,
    fontweight="bold",
    y=1.01,
)

fig.supxlabel(
    "Continuous gene-ranking score",
    fontsize=11,
)

fig.tight_layout()

# ------------------------------------------------------------
# Save figure and data
# ------------------------------------------------------------

save_figure(
    fig,
    "SupplementaryFigureS5_Top_Genes_Per_Method",
    (
        f"Top {TOP_N} genes ranked by continuous score within each "
        "resampling method. All panels use the same score axis to "
        "support direct comparison across methods."
    ),
    manuscript=False,
)

plt.show()

save_table(
    top_genes_df,
    "SupplementaryFigureS5_Top_Genes_Per_Method_Data",
)


# Supplementary Figure S6 — GSEA cutoff sensitivity

for p in sorted(GSEA_SENSITIVITY_DIR.glob("*_Top*_GSEA.csv")):
    print(p.name)

for p in sorted(GSEA_SENSITIVITY_DIR.glob("*_Top*_GSEA.csv")):
    df = pd.read_csv(p)

    print(p.name, df.shape, df["Method"].unique(), df["Nominal_Cutoff"].unique())

print("Available CSV files:")
for csv_path in sorted(RESULTS_DIR.rglob("*.csv")):
    print(csv_path)

# ============================================================
# Supplementary Figure S6
# Pathway agreement across sampling methods
# using full continuous GSEA results
# ============================================================

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

FDR_THRESHOLD = 0.25

# Minimum number of sampling methods in which a pathway must
# be significant to appear in the figure.
MIN_METHODS = 2

# Maximum number of pathways shown.
MAX_PATHWAYS = 20

# Set True to show only positive enrichment.
# Set False to retain both positive and negative NES values.
POSITIVE_ONLY = True


# ------------------------------------------------------------
# Locate and read full-ranked GSEA output files
# ------------------------------------------------------------
#
# Expected files might look like:
#   NoSmp_GSEA.csv
#   ROS_GSEA.csv
#   SMOTE_GSEA.csv
#   ADASYN_GSEA.csv
#   RUS_GSEA.csv
#   ClusterCentroids_GSEA.csv
#   AllKNN_GSEA.csv
#
# The wildcard below is intentionally broad. If it captures
# cutoff-sensitivity files, the code removes rows that contain
# Nominal_Cutoff.


FULL_GSEA_DIR = GSEA_FULL_DIR

gsea_frames = []

for csv_path in sorted(FULL_GSEA_DIR.glob("*_Enrichment_Analysis.csv")):

    df = pd.read_csv(csv_path)

    required_columns = {
        "Term",
        "NES",
        "FDR q-val",
    }

    if not required_columns.issubset(df.columns):
        print(
            f"Skipping {csv_path.name}: "
            f"missing required columns "
            f"{sorted(required_columns - set(df.columns))}"
        )
        continue

    if "Method" not in df.columns:
        method_name = csv_path.stem.replace("_Enrichment_Analysis", "")
        df["Method"] = method_name

    df["Source_File"] = csv_path.name

    gsea_frames.append(
        df[
            [
                "Method",
                "Term",
                "NES",
                "FDR q-val",
                "Source_File",
            ]
        ].copy()
    )

if not gsea_frames:
    raise FileNotFoundError(
        f"No valid full-ranked GSEA files were found in {FULL_GSEA_DIR}"
    )

gsea_df = pd.concat(gsea_frames, ignore_index=True)

print("Loaded methods:")
print(sorted(gsea_df["Method"].unique()))

print("\nRows loaded:")
print(gsea_df.shape)


# ------------------------------------------------------------
# Clean columns
# ------------------------------------------------------------

gsea_df["Method"] = gsea_df["Method"].astype(str).str.strip()

gsea_df["Term"] = gsea_df["Term"].astype(str).str.strip()

gsea_df["NES"] = pd.to_numeric(
    gsea_df["NES"],
    errors="coerce",
)

gsea_df["FDR q-val"] = pd.to_numeric(
    gsea_df["FDR q-val"],
    errors="coerce",
)

gsea_df = gsea_df.dropna(
    subset=[
        "Method",
        "Term",
        "NES",
        "FDR q-val",
    ]
)


# ------------------------------------------------------------
# Standardize method names if needed
# ------------------------------------------------------------
#
# Add alternate file labels here if your saved method names
# differ from METHOD_ORDER.

METHOD_ALIASES = {
    "No Sampling": "NoSmp",
    "No_Sampling": "NoSmp",
    "NoSampling": "NoSmp",
    "RandomOverSampler": "ROS",
    "Random Oversampling": "ROS",
    "RandomUnderSampler": "RUS",
    "Random Undersampling": "RUS",
    "Cluster Centroids": "ClusterCentroids",
    "Cluster_Centroids": "ClusterCentroids",
    "All KNN": "AllKNN",
    "All_KNN": "AllKNN",
}

gsea_df["Method"] = gsea_df["Method"].replace(METHOD_ALIASES)

# Keep only expected methods.
gsea_df = gsea_df[gsea_df["Method"].isin(METHOD_ORDER)].copy()

if gsea_df.empty:
    raise ValueError(
        "No GSEA rows matched METHOD_ORDER. "
        "Inspect the method names in the input files."
    )


# ------------------------------------------------------------
# Collapse duplicate pathway rows within each method
# ------------------------------------------------------------
#
# Some GSEA libraries can contain duplicate or repeated pathway
# names. Retain the row with the smallest FDR for each
# method-pathway pair.

gsea_df = gsea_df.sort_values(
    [
        "Method",
        "Term",
        "FDR q-val",
    ]
).drop_duplicates(
    subset=[
        "Method",
        "Term",
    ],
    keep="first",
)


# ------------------------------------------------------------
# Identify significant pathway-method combinations
# ------------------------------------------------------------

significant_df = gsea_df[gsea_df["FDR q-val"] < FDR_THRESHOLD].copy()

if POSITIVE_ONLY:
    significant_df = significant_df[significant_df["NES"] > 0].copy()

if significant_df.empty:
    raise ValueError("No pathways met the selected FDR and NES criteria.")


# ------------------------------------------------------------
# Summarize agreement across methods
# ------------------------------------------------------------

pathway_summary = (
    significant_df.groupby("Term")
    .agg(
        Significant_Methods=(
            "Method",
            "nunique",
        ),
        Mean_NES=(
            "NES",
            "mean",
        ),
        Maximum_NES=(
            "NES",
            "max",
        ),
        Minimum_FDR=(
            "FDR q-val",
            "min",
        ),
    )
    .reset_index()
)


# ------------------------------------------------------------
# Select pathways
# ------------------------------------------------------------

PDAC_RELEVANT_PATHWAYS = [
    "MAPK signaling pathway",
    "PI3K-Akt signaling pathway",
    "mTOR signaling pathway",
    "AMPK signaling pathway",
    "TGF-beta signaling pathway",
    "Hedgehog signaling pathway",
    "Hippo signaling pathway",
    "Notch signaling pathway",
    "Wnt signaling pathway",
    "Focal adhesion",
    "ECM-receptor interaction",
    "Regulation of actin cytoskeleton",
    "Cell cycle",
    "DNA replication",
    "Apoptosis",
]

available_pathways = set(pathway_summary["Term"])

selected_pathways = [
    pathway for pathway in PDAC_RELEVANT_PATHWAYS if pathway in available_pathways
]

if not selected_pathways:
    raise ValueError(
        "None of the requested PDAC-relevant pathways were found "
        "in the full-ranked GSEA results."
    )

selected_pathway_summary = pathway_summary[
    pathway_summary["Term"].isin(selected_pathways)
].copy()

# Preserve the order in PDAC_RELEVANT_PATHWAYS
selected_pathway_summary["Pathway_Order"] = selected_pathway_summary["Term"].map(
    {pathway: index for index, pathway in enumerate(PDAC_RELEVANT_PATHWAYS)}
)

selected_pathway_summary = selected_pathway_summary.sort_values("Pathway_Order").drop(
    columns="Pathway_Order"
)

selected_pathways = selected_pathway_summary["Term"].tolist()


# ------------------------------------------------------------
# Build NES matrix
# ------------------------------------------------------------

nes_matrix = significant_df[significant_df["Term"].isin(selected_pathways)].pivot_table(
    index="Term",
    columns="Method",
    values="NES",
    aggfunc="first",
)

nes_matrix = nes_matrix.reindex(columns=METHOD_ORDER)

nes_matrix = nes_matrix.reindex(selected_pathways)


# ------------------------------------------------------------
# Build FDR matrix for annotation/export
# ------------------------------------------------------------

fdr_matrix = significant_df[significant_df["Term"].isin(selected_pathways)].pivot_table(
    index="Term",
    columns="Method",
    values="FDR q-val",
    aggfunc="first",
)

fdr_matrix = fdr_matrix.reindex(
    index=selected_pathways,
    columns=METHOD_ORDER,
)


# ------------------------------------------------------------
# Add recurrence information to row labels
# ------------------------------------------------------------

row_labels = selected_pathways

# ------------------------------------------------------------
# Prepare plotting values
# ------------------------------------------------------------

plot_values = nes_matrix.to_numpy(dtype=float)

masked_values = np.ma.masked_invalid(plot_values)

valid_values = plot_values[np.isfinite(plot_values)]

if len(valid_values) == 0:
    raise ValueError("The selected NES matrix contains no valid values.")


# ------------------------------------------------------------
# Choose color scaling
# ------------------------------------------------------------

if POSITIVE_ONLY:

    vmin = float(np.nanmin(valid_values))
    vmax = float(np.nanmax(valid_values))

    # Add a small margin so the minimum and maximum
    # are not mapped to the absolute ends of the color scale.
    margin = 0.05 * (vmax - vmin)

    vmin = vmin - margin
    vmax = vmax + margin

    color_map = plt.get_cmap("Blues").copy()

    norm = None

else:

    max_abs = max(
        abs(float(np.nanmin(valid_values))),
        abs(float(np.nanmax(valid_values))),
    )

    color_map = plt.get_cmap("coolwarm").copy()

    norm = TwoSlopeNorm(
        vmin=-max_abs,
        vcenter=0,
        vmax=max_abs,
    )


# Color for nonsignificant or missing combinations.
color_map.set_bad("#E5E5E5")


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig_height = max(
    6.5,
    0.45 * len(selected_pathways),
)

fig, ax = plt.subplots(figsize=(11.5, fig_height))

if POSITIVE_ONLY:

    image = ax.imshow(
        masked_values,
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        vmin=vmin,
        vmax=vmax,
    )

else:

    image = ax.imshow(
        masked_values,
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        norm=norm,
    )


# ------------------------------------------------------------
# Annotate significant cells with NES
# ------------------------------------------------------------

threshold = (np.nanmin(valid_values) + np.nanmax(valid_values)) / 2

ANNOTATION_THRESHOLD = 2.40

for row_index in range(nes_matrix.shape[0]):
    for column_index in range(nes_matrix.shape[1]):

        value = plot_values[row_index, column_index]

        if np.isfinite(value) and value >= ANNOTATION_THRESHOLD:

            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

# ------------------------------------------------------------
# Axis labels
# ------------------------------------------------------------

column_labels = [
    METHOD_LABELS.get(
        method,
        method,
    )
    for method in METHOD_ORDER
]

ax.set_xticks(np.arange(len(column_labels)))

ax.set_xticklabels(
    column_labels,
    rotation=45,
    ha="right",
    fontsize=9,
)

ax.set_yticks(np.arange(len(row_labels)))

ax.set_yticklabels(
    row_labels,
    fontsize=8,
)

ax.set_xlabel("")
ax.set_ylabel("")

ax.set_title(
    "Consistency of biologically relevant pathway enrichment across resampling strategies",
    fontsize=15,
    fontweight="bold",
    pad=14,
)

# ------------------------------------------------------------
# Cell borders
# ------------------------------------------------------------

ax.set_xticks(
    np.arange(
        -0.5,
        nes_matrix.shape[1],
        1,
    ),
    minor=True,
)

ax.set_yticks(
    np.arange(
        -0.5,
        nes_matrix.shape[0],
        1,
    ),
    minor=True,
)

ax.grid(
    which="minor",
    linewidth=0.7,
    color="white",
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False,
)

ax.tick_params(
    axis="both",
    length=0,
)


# ------------------------------------------------------------
# Color bar and missing-value legend
# ------------------------------------------------------------

colorbar = fig.colorbar(
    image,
    ax=ax,
    fraction=0.035,
    pad=0.03,
)

colorbar.set_label(
    "Normalized enrichment score (NES)",
    fontsize=9,
)

missing_patch = Patch(
    facecolor="#E5E5E5",
    edgecolor="none",
    label=(f"Not significant " f"(FDR q ≥ {FDR_THRESHOLD}) " "or not returned"),
)

ax.legend(
    handles=[missing_patch],
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        -0.12,
    ),
    frameon=False,
)


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

fig.tight_layout(
    rect=[
        0,
        0.05,
        1,
        1,
    ]
)


# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------

save_figure(
    fig,
    "SupplementaryFigureS6_Pathway_Agreement_Across_Methods",
    (
        "Agreement of significant pathway enrichment across "
        "sampling methods using the full continuous gene rankings. "
        f"Only pathways significant at FDR q-value < "
        f"{FDR_THRESHOLD} in at least {MIN_METHODS} sampling methods "
        "were displayed. Cell color and annotations represent the "
        "normalized enrichment score. Gray cells indicate pathways "
        "that were not significant or were not returned for a method. "
        "Parenthetical values indicate the number of sampling methods "
        "in which each pathway was significant."
    ),
    manuscript=False,
)

plt.show()


# ------------------------------------------------------------
# Save NES matrix
# ------------------------------------------------------------

nes_export = nes_matrix.copy().reset_index().rename(columns={"Term": "Pathway"})

save_table(
    nes_export,
    "SupplementaryFigureS6_Pathway_Agreement_NES_Matrix",
)


# ------------------------------------------------------------
# Save FDR matrix
# ------------------------------------------------------------

fdr_export = fdr_matrix.copy().reset_index().rename(columns={"Term": "Pathway"})

save_table(
    fdr_export,
    "SupplementaryFigureS6_Pathway_Agreement_FDR_Matrix",
)


# ------------------------------------------------------------
# Save pathway agreement summary
# ------------------------------------------------------------

save_table(
    pathway_summary,
    "SupplementaryFigureS6_Pathway_Agreement_Summary",
)


# Display selected pathways
print(selected_pathway_summary.to_string(index=False))


# Supplementary Figure S8 — Feature stability summary

# ============================================================
# Supplementary Figure S8
# Agreement between Random Forest and ANOVA feature selection
# ============================================================

# ------------------------------------------------------------
# Prepare data
# ------------------------------------------------------------

plot_df = stability_summary.copy()

required_columns = {
    "Method",
    "RF_Supported_Genes",
    "ANOVA_Significant_Genes",
    "Supported_By_Both",
}

missing_columns = required_columns - set(plot_df.columns)

if missing_columns:
    raise ValueError(
        f"stability_summary is missing required columns: " f"{sorted(missing_columns)}"
    )

# Convert counts to numeric
for column in [
    "RF_Supported_Genes",
    "ANOVA_Significant_Genes",
    "Supported_By_Both",
]:
    plot_df[column] = pd.to_numeric(
        plot_df[column],
        errors="coerce",
    )

plot_df = plot_df.dropna(subset=list(required_columns)).copy()

# ------------------------------------------------------------
# Calculate mutually exclusive categories
# ------------------------------------------------------------

plot_df["RF_Only"] = plot_df["RF_Supported_Genes"] - plot_df["Supported_By_Both"]

plot_df["ANOVA_Only"] = (
    plot_df["ANOVA_Significant_Genes"] - plot_df["Supported_By_Both"]
)

# Guard against small negative values caused by rounding
plot_df["RF_Only"] = plot_df["RF_Only"].clip(lower=0)
plot_df["ANOVA_Only"] = plot_df["ANOVA_Only"].clip(lower=0)

# ------------------------------------------------------------
# Preserve preferred method order
# ------------------------------------------------------------

plot_df["Method"] = pd.Categorical(
    plot_df["Method"],
    categories=METHOD_ORDER,
    ordered=True,
)

plot_df = plot_df.sort_values("Method").reset_index(drop=True)

# ------------------------------------------------------------
# Plot stacked bars
# ------------------------------------------------------------

x = np.arange(len(plot_df))

fig, ax = plt.subplots(figsize=(11, 6.2))

both_bars = ax.bar(
    x,
    plot_df["Supported_By_Both"],
    label="Supported by both RF and ANOVA",
)

rf_only_bars = ax.bar(
    x,
    plot_df["RF_Only"],
    bottom=plot_df["Supported_By_Both"],
    label="RF only",
)

anova_bottom = plot_df["Supported_By_Both"] + plot_df["RF_Only"]

anova_only_bars = ax.bar(
    x,
    plot_df["ANOVA_Only"],
    bottom=anova_bottom,
    label="ANOVA only",
)

# ------------------------------------------------------------
# Annotate RF-only segments
# ------------------------------------------------------------
for i, row in plot_df.iterrows():

    shared = row["Supported_By_Both"]
    rf_only = row["RF_Only"]

    # Center of RF-only segment
    y = shared + rf_only / 2

    # Tiny segments: place label just above the blue bar
    if rf_only < 250:
        y = shared + 250

    ax.text(
        i,
        y,
        f"{int(rf_only):,}",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="black",
    )

# ------------------------------------------------------------
# Axis formatting
# ------------------------------------------------------------

ax.set_xticks(x)

ax.set_xticklabels(
    [
        METHOD_LABELS.get(str(method), str(method))
        for method in plot_df["Method"].astype(str)
    ],
    rotation=25,
    ha="right",
)

ax.set_ylabel("Number of genes")

ax.set_xlabel("")

ax.set_title(
    "Overlap between Random Forest and ANOVA selected genes across resampling methods",
    fontsize=15,
    fontweight="bold",
    pad=12,
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.legend(
    frameon=False,
    loc="upper right",
)

# ------------------------------------------------------------
# Add total union count above each bar
# ------------------------------------------------------------

plot_df["Union_Genes"] = (
    plot_df["Supported_By_Both"] + plot_df["RF_Only"] + plot_df["ANOVA_Only"]
)

vertical_offset = plot_df["Union_Genes"].max() * 0.012

for index, total in enumerate(plot_df["Union_Genes"]):

    ax.text(
        index,
        total + vertical_offset,
        f"{int(total):,}",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )

# Leave space above the tallest bar
ax.set_ylim(
    0,
    plot_df["Union_Genes"].max() * 1.10,
)

fig.tight_layout()

# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------

save_figure(
    fig,
    "SupplementaryFigureS8_RF_ANOVA_Agreement",
    (
        "Agreement between Random Forest and ANOVA feature-selection "
        "approaches across resampling strategies. Bars partition the "
        "union of supported genes into genes supported by both methods, "
        "genes supported only by Random Forest, and genes significant "
        "only by ANOVA. Values above the bars indicate the total number "
        "of genes supported by at least one feature-selection approach."
    ),
    False,
)

plt.show()

# ------------------------------------------------------------
# Save supporting table
# ------------------------------------------------------------

s8_table = plot_df[
    [
        "Method",
        "Supported_By_Both",
        "RF_Only",
        "ANOVA_Only",
        "Union_Genes",
        "RF_Supported_Genes",
        "ANOVA_Significant_Genes",
    ]
].copy()

save_table(
    s8_table,
    "SupplementaryFigureS8_RF_ANOVA_Agreement_Data",
)

print(s8_table.to_string(index=False))


# Part 4 — Summary tables, manifest, and reproducibility report

print(selected_summary.columns.tolist())
print(selected_summary.head())
print("####")
print(plot_df.columns.tolist())
print(plot_df.head())

for name in list(globals()):
    obj = globals()[name]
    if isinstance(obj, pd.DataFrame):
        if {"Supported_By_Both", "RF_Only", "ANOVA_Only"}.issubset(obj.columns):
            print(name)

save_table(
    hypergeom.sort_values("Hypergeometric_P"), "Table_Hypergeometric_PDAC_Enrichment"
)

save_table(pathway_nes, "Table_PDAC_Pathway_NES_Summary")

save_table(stability_summary, "Supplementary_Table_Feature_Stability")

save_table(s8_table, "Supplementary_Table_S8_Gene_Selection_Agreement")

save_table(tie_diag, "Supplementary_Table_Ranking_Tie_Diagnostics")

manifest_df = pd.DataFrame(FIGURE_MANIFEST)
manifest_path = RESULTS_DIR / "Figure_Manifest.csv"
manifest_df.to_csv(manifest_path, index=False)
print(manifest_df[["Figure", "Description", "Category", "PNG"]].to_string(index=False))
print(manifest_path)

# ============================================================
# Optional project bookkeeping (not manuscript outputs)
# ============================================================

end = datetime.now()
versions = {
    "Python": sys.version.replace("\n", " "),
    "Platform": platform.platform(),
    "NumPy": np.__version__,
    "pandas": pd.__version__,
    "Matplotlib": matplotlib.__version__,
}
lines = [
    "PDAC Methylation Figure Generation Report",
    "=" * 70,
    f"Generated: {end.isoformat(timespec='seconds')}",
    f"Elapsed time: {end-GENERATION_START}",
    f"Project root: {PROJECT_ROOT}",
    f"Results directory: {RESULTS_DIR}",
    "",
    "Dataset summary",
    "-" * 70,
    f"Methods analyzed: {rankings['Method'].nunique()}",
    f"Unique genes: {rankings['Gene'].nunique():,}",
    f"Ranking rows: {len(rankings):,}",
    f"Pathways summarized: {len(pathway_nes):,}",
    f"Figures generated: {len(FIGURE_MANIFEST):,}",
    "",
    "Software environment",
    "-" * 70,
]
for k, v in versions.items():
    lines.append(f"{k}: {v}")
lines += (
    ["", "Input files", "-" * 70]
    + [f"{k}: {v}" for k, v in FILES.items()]
    + [
        "",
        "Output directories",
        "-" * 70,
        f"Main figures: {MANUSCRIPT_DIR}",
        f"Supplementary figures: {SUPPLEMENTARY_DIR}",
        f"Summary tables: {SUMMARY_TABLE_DIR}",
        f"Manifest: {manifest_path}",
    ]
)
report = RESULTS_DIR / "Methods_Report.txt"
report.write_text("\n".join(lines), encoding="utf-8")
print(report)


# Expected output structure
#
# continuous_enrichment_results/
# ├── Figure_Manifest.csv
# ├── Methods_Report.txt
# ├── figures/
# │   ├── manuscript/
# │   └── supplementary/
# └── summary_tables/
