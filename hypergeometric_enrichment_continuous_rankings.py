#!/usr/bin/env python3
"""Run PDAC methylation enrichment using continuous feature statistics.

The analysis reads per-run random forest and ANOVA files, maps CpG markers to
genes, constructs continuous gene-level rankings, performs hypergeometric
enrichment against a curated PDAC gene set, and runs preranked KEGG GSEA.

Before running, verify the method directories and the CpG-to-gene annotation
path and column names in the configuration section.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

try:
    import gseapy as gp
except ImportError as exc:
    raise ImportError("Install gseapy with: pip install gseapy") from exc

# 1. Configuration

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURE_SELECTION_DIR = PROJECT_ROOT / "FeatureSelection"

METHOD_DIRS = {
    "NoSmp": FEATURE_SELECTION_DIR / "NoSampling/output_files3/NoSmp/Meth",
    "ROS": FEATURE_SELECTION_DIR / "OverSampling/output_files3/ROS/Meth",
    "SMOTE": FEATURE_SELECTION_DIR / "OverSampling/output_files3/Meth",
    "ADASYN": (
        FEATURE_SELECTION_DIR / "OverSampling/output_files3/AdaptiveSynthetic/Meth"
    ),
    "RUS": FEATURE_SELECTION_DIR / "UnderSampling/output_files3/RUS/Meth",
    "ClusterCentroids": (
        FEATURE_SELECTION_DIR / "UnderSampling/output_files3/ClusterCentroids/Meth"
    ),
    "AllKNN": FEATURE_SELECTION_DIR / "UnderSampling/output_files3/AllKNN/Meth",
}

CPG_GENE_MAP_FILE = DATA_DIR / "450k_All.csv"
CPG_COLUMN_IN_MAP = "Composite_Element_REF"
GENE_COLUMN_IN_MAP = "Gene_Symbol"
GENE_SEPARATOR = ";"

OUTPUT_DIR = PROJECT_ROOT / "continuous_enrichment_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_RF_SUPPORTED_RUNS = 1
MIN_ANOVA_SIGNIFICANT_RUNS = 1
ANOVA_ALPHA = 0.05

GENE_SET_LIBRARY = "KEGG_2021_Human"
GSEA_PERMUTATIONS = 1000
GSEA_SEED = 42

for method, path in METHOD_DIRS.items():
    print(f"{method:18s} -> {path.resolve()} | exists={path.exists()}")
print(
    "CpG mapping:", CPG_GENE_MAP_FILE.resolve(), "| exists=", CPG_GENE_MAP_FILE.exists()
)

# ---------------------------------------------------------
# Count CpG markers in ANOVA statistics files
# ---------------------------------------------------------

summary_rows = []
run_rows = []

for method, method_dir in METHOD_DIRS.items():

    anova_files = sorted(method_dir.glob("Meth_ANOVA_Statistics_Run*.csv"))

    if not anova_files:
        print(f"No ANOVA files found for {method}")
        continue

    all_cpgs = set()

    for file in anova_files:

        df = pd.read_csv(file)

        if "CpG_Marker" not in df.columns:
            print(f"Missing CpG_Marker column: {file}")
            continue

        cpgs = df["CpG_Marker"].dropna().astype(str).str.strip()

        unique_cpgs = set(cpgs)

        all_cpgs.update(unique_cpgs)

        # Extract run number from filename
        run = file.stem.replace("Meth_ANOVA_Statistics_Run", "")

        run_rows.append(
            {
                "Method": method,
                "Run": run,
                "Rows": len(df),
                "Unique_CpGs": len(unique_cpgs),
            }
        )

    summary_rows.append(
        {
            "Method": method,
            "ANOVA_Files_Found": len(anova_files),
            "Unique_CpGs_Across_Runs": len(all_cpgs),
        }
    )


run_cpg_counts = pd.DataFrame(run_rows)
method_cpg_summary = pd.DataFrame(summary_rows)

print("\nCpG counts by run:")
print(run_cpg_counts.to_string(index=False))

print("\nCpG summary by method:")
print(method_cpg_summary.to_string(index=False))


# 2. Curated PDAC reference genes

table1_genes = [
    "GSTM2",
    "NGFB",
    "PTCH2",
    "TAL1",
    "ALK",
    "DES",
    "FRZB",
    "TMEFF2",
    "IGFBP5",
    "POMC",
    "CASP3",
    "CASP6",
    "EPHA5",
    "FGF2",
    "FGF5",
    "HHIP",
    "IGFBP7",
    "KDR",
    "KIT",
    "PITX2",
    "SLIT2",
    "FLT4",
    "SCGB3A1",
    "ESR1",
    "EYA4",
    "COL1A2",
    "FZD9",
    "HOXA11",
    "HOXA9",
    "NPY",
    "SMO",
    "TFPI2",
    "TWIST1",
    "FGFR1",
    "MOS",
    "NEFL",
    "NRG1",
    "PENK",
    "SFRP1",
    "SOX17",
    "TUSC3",
    "APBA1",
    "DBC1",
    "FANCG",
    "GAS1",
    "TMEFF1",
    "FGF8",
    "RET",
    "ASCL2",
    "BDNF",
    "FGF3",
    "FLI1",
    "HSD17B12",
    "IGF2AS",
    "LMO1",
    "MYOD1",
    "THY1",
    "WT1",
    "CCND2",
    "ITPR2",
    "CCNA1",
    "FLT1",
    "FLT3",
    "SOX1",
    "CHGA",
    "DLK1",
    "NTRK3",
    "RASGRF1",
    "CDH13",
    "HS3ST2",
    "MMP2",
    "MYH11",
    "TUBB3",
    "ALOX12",
    "COL1A1",
    "GAS7",
    "HIC1",
    "ADCYAP1",
    "GALR1",
    "NTSR1",
    "ERG",
    "HIC2",
    "SEZ6L",
    "TBX1",
    "TIMP3",
    "DHCR24",
    "EPHX1",
    "HDAC1",
    "MUC1",
    "NBL1",
    "NES",
    "NID1",
    "S100A2",
    "SFN",
    "TGFB2",
    "CASP8",
    "CLK1",
    "GLI2",
    "IHH",
    "IL1RN",
    "UGT1A1",
    "VAMP8",
    "ACVR2B",
    "CCR5",
    "MST1R",
    "PPARG",
    "TNFSF10",
    "CCKAR",
    "CXCL9",
    "IL2",
    "IL8",
    "SPP1",
    "CSF1R",
    "CSF2",
    "FGF1",
    "FGFR4",
    "IL12B",
    "ITK",
    "CCND3",
    "FRK",
    "NOTCH4",
    "PPARD",
    "SLC22A3",
    "SPDEF",
    "ASB4",
    "CLDN4",
    "CPA4",
    "EPHB4",
    "FLJ20712",
    "LMTK2",
    "MEST",
    "NOS3",
    "PRSS1",
    "SEMA3C",
    "TRIP6",
    "CDH17",
    "DLC1",
    "E2F5",
    "PLAT",
    "PSCA",
    "SFTPC",
    "TNFRSF10A",
    "DAPK1",
    "LCN2",
    "SYK",
    "VAV2",
    "ABCC2",
    "CYP2E1",
    "FGFR2",
    "MAP3K8",
    "SFTPA1",
    "SNCG",
    "TCF7L2",
    "APOA1",
    "CCND1",
    "DDB2",
    "INS",
    "MMP1",
    "MMP7",
    "PHLDA2",
    "SPI1",
    "TMPRSS4",
    "TRIM29",
    "TSG101",
    "ARHGDIB",
    "IAPP",
    "IFNG",
    "KRT1",
    "PTHLH",
    "PTPN6",
    "UNG",
    "BMP4",
    "KIAA0125",
    "RIPK3",
    "APBA2",
    "MAGEL2",
    "NDN",
    "TJP1",
    "CARD15",
    "CREBBP",
    "NQO1",
    "PRSS8",
    "BRCA1",
    "CRK",
    "CSF3",
    "ITGB4",
    "LIG3",
    "NOS2A",
    "SEPT9",
    "SERPINB5",
    "BAX",
    "BSG",
    "CEACAM1",
    "EMR3",
    "JAK3",
    "NOTCH3",
    "PLAUR",
    "PTPRH",
    "DNMT3B",
    "ID1",
    "MMP9",
    "PI3",
    "PTK6",
    "SRC",
    "B3GALT5",
    "RIPK4",
    "TFF1",
    "TFF2",
    "BCR",
    "LIF",
    "BGN",
    "GRPR",
    "GUCY2F",
    "MAGEA1",
]
common_pdac_genes = ["KRAS", "BRCA1", "BRCA2", "PALB2", "CDKN2A", "TP53", "SMAD4"]
PDAC_REFERENCE_GENES = {
    str(g).strip().upper() for g in table1_genes + common_pdac_genes if str(g).strip()
}
print("Unique curated PDAC genes:", len(PDAC_REFERENCE_GENES))


# 3. Load CpG-to-gene annotation


def load_cpg_gene_map(path, cpg_col, gene_col, separator=";"):
    mapping = pd.read_csv(path, usecols=[cpg_col, gene_col], low_memory=False)

    mapping.columns = mapping.columns.astype(str).str.strip()

    mapping = mapping.rename(columns={cpg_col: "CpG_Marker", gene_col: "Gene"})

    mapping = mapping.dropna(subset=["CpG_Marker", "Gene"]).copy()

    mapping["CpG_Marker"] = mapping["CpG_Marker"].astype(str).str.strip()

    mapping["Gene"] = mapping["Gene"].astype(str).str.strip()

    if separator:
        mapping["Gene"] = mapping["Gene"].str.split(separator)
        mapping = mapping.explode("Gene")

    mapping["Gene"] = mapping["Gene"].astype(str).str.strip().str.upper()

    mapping = mapping[
        (mapping["CpG_Marker"] != "")
        & (mapping["Gene"] != "")
        & (mapping["Gene"] != "NAN")
    ]

    return mapping.drop_duplicates().reset_index(drop=True)


cpg_gene_map = load_cpg_gene_map(
    CPG_GENE_MAP_FILE, CPG_COLUMN_IN_MAP, GENE_COLUMN_IN_MAP, GENE_SEPARATOR
)
print(cpg_gene_map.head())
print("Unique mapped CpGs:", cpg_gene_map["CpG_Marker"].nunique())
print("Unique mapped genes:", cpg_gene_map["Gene"].nunique())


# 4. Validate and load new per-run files

RF_RUNS = {
    "SMOTE": 10,
    "ADASYN": 10,
    "ROS": 10,
    "RUS": 10,
    "ClusterCentroids": 10,
    "AllKNN": 10,
    "NoSmp": 10,
}

ANOVA_RUNS = {
    "SMOTE": 10,
    "ADASYN": 10,
    "ROS": 10,
    "RUS": 10,
    "ClusterCentroids": 10,
    "AllKNN": 10,
    "NoSmp": 1,
}


def validate_method_files(directory, n_rf_runs, n_anova_runs):
    missing = []

    for run in range(1, n_rf_runs + 1):
        rf_path = directory / f"Meth_RF_Statistics_Run{run}.csv"
        if not rf_path.exists():
            missing.append(rf_path)

    for run in range(1, n_anova_runs + 1):
        anova_path = directory / f"Meth_ANOVA_Statistics_Run{run}.csv"
        if not anova_path.exists():
            missing.append(anova_path)

    return missing


all_missing = []

for method, directory in METHOD_DIRS.items():
    missing = validate_method_files(
        directory=directory, n_rf_runs=RF_RUNS[method], n_anova_runs=ANOVA_RUNS[method]
    )

    print(f"{method}: " f"{'OK' if not missing else str(len(missing)) + ' missing'}")

    all_missing.extend(missing)

if all_missing:
    for path in all_missing[:20]:
        print(" -", path)

    raise FileNotFoundError("Correct METHOD_DIRS or regenerate missing files.")


def load_method_statistics(method, directory, n_rf_runs=10, n_anova_runs=10):
    rf_frames = []
    anova_frames = []

    for run in range(1, n_rf_runs + 1):
        rf_file = directory / f"Meth_RF_Statistics_Run{run}.csv"

        rf = pd.read_csv(rf_file, usecols=["CpG_Marker", "RF_Importance"])

        rf["CpG_Marker"] = rf["CpG_Marker"].astype(str).str.strip()

        rf["RF_Importance"] = pd.to_numeric(
            rf["RF_Importance"], errors="coerce"
        ).fillna(0.0)

        rf["Method"] = method
        rf["Run"] = run
        rf_frames.append(rf)

    for run in range(1, n_anova_runs + 1):
        anova_file = directory / f"Meth_ANOVA_Statistics_Run{run}.csv"

        anova = pd.read_csv(anova_file, usecols=["CpG_Marker", "ANOVA_F", "ANOVA_P"])

        anova["CpG_Marker"] = anova["CpG_Marker"].astype(str).str.strip()

        anova["ANOVA_F"] = (
            pd.to_numeric(anova["ANOVA_F"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        anova["ANOVA_P"] = (
            pd.to_numeric(anova["ANOVA_P"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(1.0)
        )

        anova["Method"] = method
        anova["Run"] = run
        anova_frames.append(anova)

    return (
        pd.concat(rf_frames, ignore_index=True),
        pd.concat(anova_frames, ignore_index=True),
    )


# 5. Convert CpG statistics to gene statistics


def aggregate_cpg_to_gene(rf, anova, mapping):
    """
    Convert CpG statistics to gene statistics within each run.
    RF and ANOVA are kept separate because they may have
    different numbers of runs.
    """

    rf_gene_by_run = (
        rf.merge(mapping, on="CpG_Marker", how="inner")
        .groupby(["Method", "Run", "Gene"], as_index=False)
        .agg(
            RF_Importance=("RF_Importance", "max"),
            RF_CpG_Count=("CpG_Marker", "nunique"),
        )
    )

    anova_gene_by_run = (
        anova.merge(mapping, on="CpG_Marker", how="inner")
        .groupby(["Method", "Run", "Gene"], as_index=False)
        .agg(
            ANOVA_F=("ANOVA_F", "max"),
            ANOVA_Min_P=("ANOVA_P", "min"),
            ANOVA_CpG_Count=("CpG_Marker", "nunique"),
        )
    )

    return rf_gene_by_run, anova_gene_by_run


def summarize_across_runs(
    rf_gene_by_run, anova_gene_by_run, expected_rf_runs, expected_anova_runs, alpha=0.05
):
    rf_summary = rf_gene_by_run.groupby(["Method", "Gene"], as_index=False).agg(
        RF_Mean=("RF_Importance", "mean"),
        RF_SD=("RF_Importance", "std"),
        RF_Max=("RF_Importance", "max"),
        RF_Supported_Runs=("RF_Importance", lambda x: int((x > 0).sum())),
    )
    anova_summary = anova_gene_by_run.groupby(["Method", "Gene"], as_index=False).agg(
        ANOVA_F_Mean=("ANOVA_F", "mean"),
        ANOVA_F_SD=("ANOVA_F", "std"),
        ANOVA_F_Max=("ANOVA_F", "max"),
        ANOVA_Significant_Runs=("ANOVA_Min_P", lambda x: int((x < alpha).sum())),
        ANOVA_Min_P=("ANOVA_Min_P", "min"),
    )
    summary = rf_summary.merge(anova_summary, on=["Method", "Gene"], how="outer")

    summary["RF_Mean"] = summary["RF_Mean"].fillna(0.0)
    summary["RF_SD"] = summary["RF_SD"].fillna(0.0)
    summary["RF_Max"] = summary["RF_Max"].fillna(0.0)
    summary["RF_Supported_Runs"] = summary["RF_Supported_Runs"].fillna(0).astype(int)

    summary["ANOVA_F_Mean"] = summary["ANOVA_F_Mean"].fillna(0.0)
    summary["ANOVA_F_SD"] = summary["ANOVA_F_SD"].fillna(0.0)
    summary["ANOVA_F_Max"] = summary["ANOVA_F_Max"].fillna(0.0)
    summary["ANOVA_Significant_Runs"] = (
        summary["ANOVA_Significant_Runs"].fillna(0).astype(int)
    )
    summary["ANOVA_Min_P"] = summary["ANOVA_Min_P"].fillna(1.0)

    summary["Expected_RF_Runs"] = expected_rf_runs
    summary["Expected_ANOVA_Runs"] = expected_anova_runs

    return summary


all_rf_gene_by_run = []
all_anova_gene_by_run = []
all_gene_summaries = []

for method, directory in METHOD_DIRS.items():
    print(f"Processing {method}...")

    rf, anova = load_method_statistics(
        method=method,
        directory=directory,
        n_rf_runs=RF_RUNS[method],
        n_anova_runs=ANOVA_RUNS[method],
    )

    rf_gene_by_run, anova_gene_by_run = aggregate_cpg_to_gene(rf, anova, cpg_gene_map)

    gene_summary = summarize_across_runs(
        rf_gene_by_run=rf_gene_by_run,
        anova_gene_by_run=anova_gene_by_run,
        expected_rf_runs=RF_RUNS[method],
        expected_anova_runs=ANOVA_RUNS[method],
        alpha=ANOVA_ALPHA,
    )

    all_rf_gene_by_run.append(rf_gene_by_run)
    all_anova_gene_by_run.append(anova_gene_by_run)
    all_gene_summaries.append(gene_summary)

    print(
        f"  genes={gene_summary['Gene'].nunique():,}; "
        f"RF gene-run rows={len(rf_gene_by_run):,}; "
        f"ANOVA gene-run rows={len(anova_gene_by_run):,}"
    )

rf_gene_by_run_all = pd.concat(all_rf_gene_by_run, ignore_index=True)

anova_gene_by_run_all = pd.concat(all_anova_gene_by_run, ignore_index=True)

gene_summary_all = pd.concat(all_gene_summaries, ignore_index=True)

rf_gene_by_run_all.to_csv(OUTPUT_DIR / "RF_Gene_Statistics_By_Run.csv", index=False)

anova_gene_by_run_all.to_csv(
    OUTPUT_DIR / "ANOVA_Gene_Statistics_By_Run.csv", index=False
)

gene_summary_all.to_csv(OUTPUT_DIR / "Gene_Statistics_Across_Runs.csv", index=False)

s1 = gene_summary_all.copy()

s1 = s1.sort_values(
    ["Method", "RF_Mean", "ANOVA_F_Mean"], ascending=[True, False, False]
).reset_index(drop=True)

s1.insert(0, "Rank", range(1, len(s1) + 1))

s1.to_csv(
    OUTPUT_DIR / "Supplementary_Table_S1_Consensus_Gene_Rankings.csv", index=False
)


# 6. Construct continuous rankings for GSEA


def zscore_log1p(series):
    values = np.log1p(series.astype(float).clip(lower=0))
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - values.mean()) / sd


def build_rankings(summary):
    frames, ranked_lists = [], {}
    for method, frame in summary.groupby("Method", sort=False):
        frame = frame.copy()
        frame["RF_Z"] = zscore_log1p(frame["RF_Mean"])
        frame["ANOVA_Z"] = zscore_log1p(frame["ANOVA_F_Mean"])
        frame["Score"] = (frame["RF_Z"] + frame["ANOVA_Z"]) / 2.0
        frame = frame.sort_values(
            ["Score", "ANOVA_F_Mean", "RF_Mean", "Gene"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        frames.append(frame)
        ranked_lists[method] = frame[["Gene", "Score"]].copy()
    return pd.concat(frames, ignore_index=True), ranked_lists


ranked_gene_summary, ranked_lists = build_rankings(gene_summary_all)
ranked_gene_summary.to_csv(
    OUTPUT_DIR / "Continuous_Gene_Rankings_All_Methods.csv", index=False
)
for method, ranking in ranked_lists.items():
    ranking.to_csv(OUTPUT_DIR / f"{method}_Continuous_Gene_Ranking.csv", index=False)

tie_rows = []
for method, ranking in ranked_lists.items():
    tie_rows.append(
        {
            "Method": method,
            "Genes": len(ranking),
            "Unique_Scores": ranking["Score"].nunique(),
            "Percent_Genes_In_Ties": 100
            * ranking["Score"].duplicated(keep=False).mean(),
        }
    )
tie_diagnostics = pd.DataFrame(tie_rows)
tie_diagnostics.to_csv(OUTPUT_DIR / "Ranking_Tie_Diagnostics.csv", index=False)


# 7. Build discrete selected-gene sets for hypergeometric enrichment
# A gene enters the discrete set when it has positive RF importance in at least
# the configured number of runs and ANOVA significance in at least the
# configured number of runs.

selected_gene_sets = {}
selection_rows = []
for method, frame in gene_summary_all.groupby("Method"):
    selected = frame[
        (frame["RF_Supported_Runs"] >= MIN_RF_SUPPORTED_RUNS)
        & (frame["ANOVA_Significant_Runs"] >= MIN_ANOVA_SIGNIFICANT_RUNS)
    ]["Gene"]
    selected_gene_sets[method] = set(selected)
    selection_rows.append(
        {"Method": method, "Selected_Genes": len(selected_gene_sets[method])}
    )
selection_summary = pd.DataFrame(selection_rows)
selection_summary.to_csv(OUTPUT_DIR / "Selected_Gene_Set_Summary.csv", index=False)


# 8. Hypergeometric enrichment against curated PDAC genes

# ---------------------------------------------------------
# 1. Define the measured background universe
# ---------------------------------------------------------
measured_gene_universe = set(
    cpg_gene_map["Gene"].dropna().astype(str).str.strip().str.upper()
)

# Restrict the literature-derived PDAC reference genes
# to genes represented in the measured universe
reference_in_universe = {
    str(gene).strip().upper() for gene in PDAC_REFERENCE_GENES
} & measured_gene_universe

M = len(measured_gene_universe)
K = len(reference_in_universe)

print(f"Background universe size (M): {M:,}")
print(f"PDAC reference genes in universe (K): {K:,}")


# ---------------------------------------------------------
# 2. Run hypergeometric enrichment for each method
# ---------------------------------------------------------
rows = []

for method, selected_genes in selected_gene_sets.items():

    # Clean and standardize the selected gene symbols
    selected = {
        str(gene).strip().upper()
        for gene in selected_genes
        if pd.notna(gene) and str(gene).strip()
    }

    # Keep only genes represented in the measured universe
    selected = selected & measured_gene_universe

    # Determine overlap with the PDAC reference set
    overlap = selected & reference_in_universe

    n = len(selected)
    k = len(overlap)

    # Expected number of PDAC genes under random selection
    expected_overlap = (n * K / M) if M > 0 else np.nan

    # One-sided hypergeometric enrichment p-value:
    # probability of observing k or more PDAC genes
    p_value = hypergeom.sf(k - 1, M, K, n) if n > 0 and M > 0 and K > 0 else np.nan

    # Fold enrichment:
    # observed overlap divided by expected overlap
    fold_enrichment = k / expected_overlap if expected_overlap > 0 else np.nan

    # Proportion of selected genes that are PDAC-associated
    pdac_percentage = 100 * k / n if n > 0 else np.nan

    rows.append(
        {
            "Method": method,
            "Universe_Size_M": M,
            "PDAC_Genes_In_Universe_K": K,
            "Selected_Genes_n": n,
            "PDAC_Overlap_k": k,
            "Expected_PDAC_Overlap": expected_overlap,
            "Fold_Enrichment": fold_enrichment,
            "PDAC_Percentage": pdac_percentage,
            "Hypergeometric_P": p_value,
            "Negative_Log10_P": (
                -np.log10(p_value) if pd.notna(p_value) and p_value > 0 else np.inf
            ),
            "Overlap_Genes": ";".join(sorted(overlap)),
        }
    )


# ---------------------------------------------------------
# 3. Create and sort the results table
# ---------------------------------------------------------
hypergeom_results = pd.DataFrame(rows)

hypergeom_results = hypergeom_results.sort_values(
    by="Hypergeometric_P", ascending=True, na_position="last"
).reset_index(drop=True)

hypergeom_results.insert(0, "Enrichment_Rank", range(1, len(hypergeom_results) + 1))


# ---------------------------------------------------------
# 4. Export the full supplementary results
# ---------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

full_output_file = OUTPUT_DIR / "Hypergeometric_PDAC_Gene_Enrichment_Full.csv"

hypergeom_results.to_csv(full_output_file, index=False)

print(f"\nFull results saved to:\n{full_output_file}")


# ---------------------------------------------------------
# 5. Create a concise manuscript-ready table
# ---------------------------------------------------------
manuscript_table = hypergeom_results[
    [
        "Enrichment_Rank",
        "Method",
        "Selected_Genes_n",
        "PDAC_Overlap_k",
        "PDAC_Percentage",
        "Expected_PDAC_Overlap",
        "Fold_Enrichment",
        "Hypergeometric_P",
    ]
].copy()

# Rounded display columns
manuscript_table["PDAC_Percentage"] = manuscript_table["PDAC_Percentage"].round(2)

manuscript_table["Expected_PDAC_Overlap"] = manuscript_table[
    "Expected_PDAC_Overlap"
].round(2)

manuscript_table["Fold_Enrichment"] = manuscript_table["Fold_Enrichment"].round(3)

manuscript_output_file = (
    OUTPUT_DIR / "Hypergeometric_PDAC_Gene_Enrichment_Manuscript_Table.csv"
)

manuscript_table.to_csv(manuscript_output_file, index=False)

print(f"\nManuscript table saved to:\n{manuscript_output_file}")


print(manuscript_table.to_string(index=False))

hypergeom_results = pd.DataFrame(rows).sort_values("Hypergeometric_P")
hypergeom_results.to_csv(
    OUTPUT_DIR / "Hypergeometric_PDAC_Gene_Enrichment.csv", index=False
)


# 9. Primary full-ranked-list KEGG GSEA

gsea_dir = OUTPUT_DIR / "gsea_full_ranked"
gsea_dir.mkdir(parents=True, exist_ok=True)
primary_gsea_results = {}

for method, ranking in ranked_lists.items():
    print(f"Running GSEA for {method} ({len(ranking):,} genes)")
    result = gp.prerank(
        rnk=ranking[["Gene", "Score"]],
        gene_sets=GENE_SET_LIBRARY,
        min_size=10,
        max_size=500,
        permutation_num=GSEA_PERMUTATIONS,
        seed=GSEA_SEED,
        outdir=None,
        verbose=False,
    )
    result_df = result.res2d.copy()
    result_df["Method"] = method
    primary_gsea_results[method] = result_df
    result_df.to_csv(gsea_dir / f"{method}_Enrichment_Analysis.csv", index=False)
print("GSEA complete.")

# ---------------------------------------------------------
# Count all significant KEGG pathways for each method
# ---------------------------------------------------------

significant_pathway_rows = []

for method, df in primary_gsea_results.items():
    temp = df.copy()
    temp.columns = temp.columns.str.strip()

    # Convert relevant columns to numeric
    temp["FDR q-val"] = pd.to_numeric(temp["FDR q-val"], errors="coerce")

    temp["NES"] = pd.to_numeric(temp["NES"], errors="coerce")

    # All pathways significant at FDR <= 0.05
    significant = temp[temp["FDR q-val"] <= 0.05].copy()

    # Optional: count positive and negative enrichment separately
    positively_enriched = significant[significant["NES"] > 0]

    negatively_enriched = significant[significant["NES"] < 0]

    significant_pathway_rows.append(
        {
            "Method": method,
            "Total_Tested_Pathways": len(temp),
            "Significant_Pathways_FDR_0.05": len(significant),
            "Significant_Positive_NES": len(positively_enriched),
            "Significant_Negative_NES": len(negatively_enriched),
            "Minimum_FDR": (
                significant["FDR q-val"].min() if not significant.empty else np.nan
            ),
            "Maximum_Absolute_NES": (
                significant["NES"].abs().max() if not significant.empty else np.nan
            ),
        }
    )


significant_pathway_counts = pd.DataFrame(significant_pathway_rows)

# Apply your preferred method order
significant_pathway_counts = significant_pathway_counts.sort_values(
    "Method"
).reset_index(drop=True)

# Save results
significant_pathway_counts.to_csv(
    OUTPUT_DIR / "Significant_KEGG_Pathway_Counts.csv", index=False
)

print(significant_pathway_counts.to_string(index=False))


# 10. PDAC-focused pathway summary

PDAC_KEYWORDS = [
    "KRAS",
    "MAPK",
    "TGF",
    "SMAD",
    "NOTCH",
    "HEDGEHOG",
    "PI3K",
    "AKT",
    "FOCAL",
    "ADHESION",
    "EMT",
    "CELL CYCLE",
    "CELL_CYCLE",
    "DNA REPLICATION",
    "DNA_REPLICATION",
]
keyword_pattern = "|".join(PDAC_KEYWORDS)
method_pdac = {}

for method, df in primary_gsea_results.items():
    temp = df.copy()
    temp.columns = temp.columns.str.strip()
    temp["FDR q-val"] = pd.to_numeric(temp["FDR q-val"], errors="coerce")
    temp["NES"] = pd.to_numeric(temp["NES"], errors="coerce")
    filtered = temp[
        (temp["FDR q-val"] <= 0.05)
        & temp["Term"]
        .astype(str)
        .str.contains(keyword_pattern, case=False, na=False, regex=True)
    ].copy()
    method_pdac[method] = filtered[
        [c for c in ["Term", "NES", "FDR q-val", "Lead_genes"] if c in filtered.columns]
    ]

all_terms = sorted({term for df in method_pdac.values() for term in df.get("Term", [])})
summary_rows = []
for term in all_terms:
    row, fdrs = {"Pathway": term}, []
    for method, df in method_pdac.items():
        match = df[df["Term"] == term]
        row[f"NES ({method})"] = np.nan if match.empty else match.iloc[0]["NES"]
        if not match.empty:
            fdrs.append(match.iloc[0]["FDR q-val"])
    row["FDR (best)"] = min(fdrs) if fdrs else np.nan
    summary_rows.append(row)

pdac_pathway_summary = pd.DataFrame(summary_rows)
pdac_pathway_summary.to_csv(OUTPUT_DIR / "PDAC_Pathway_NES_Summary.csv", index=False)

# ---------------------------------------------------------
# Count significant PDAC-focused pathways for each method
# ---------------------------------------------------------

pdac_pathway_counts = pd.DataFrame(
    {
        "Method": method_pdac.keys(),
        "Significant_PDAC_Focused_Pathways": [len(df) for df in method_pdac.values()],
    }
)

preferred_order = [
    "NoSmp",
    "ROS",
    "SMOTE",
    "ADASYN",
    "RUS",
    "ClusterCentroids",
    "AllKNN",
]

pdac_pathway_counts["Method"] = pd.Categorical(
    pdac_pathway_counts["Method"], categories=preferred_order, ordered=True
)

pdac_pathway_counts = pdac_pathway_counts.sort_values("Method").reset_index(drop=True)

pdac_pathway_counts.to_csv(
    OUTPUT_DIR / "Significant_PDAC_Focused_Pathway_Counts.csv", index=False
)

print(pdac_pathway_counts.to_string(index=False))

FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Optional display order
METHOD_ORDER = ["NoSmp", "ROS", "SMOTE", "ADASYN", "RUS", "ClusterCentroids", "AllKNN"]

# Keep only methods that actually occur in the results
METHOD_ORDER = [
    method for method in METHOD_ORDER if method in gene_summary_all["Method"].unique()
]

print("Figure directory:", FIGURE_DIR.resolve())
print("Methods:", METHOD_ORDER)

selection_counts = (
    gene_summary_all.assign(
        RF_Selected=lambda df: df["RF_Supported_Runs"] > 0,
        ANOVA_Selected=lambda df: df["ANOVA_Significant_Runs"] > 0,
        Both_Selected=lambda df: (
            (df["RF_Supported_Runs"] > 0) & (df["ANOVA_Significant_Runs"] > 0)
        ),
    )
    .groupby("Method", as_index=False)
    .agg(
        RF_Selected=("RF_Selected", "sum"),
        ANOVA_Selected=("ANOVA_Selected", "sum"),
        Both_Selected=("Both_Selected", "sum"),
    )
)

selection_counts["Method"] = pd.Categorical(
    selection_counts["Method"], categories=METHOD_ORDER, ordered=True
)

selection_counts = selection_counts.sort_values("Method")

x = np.arange(len(selection_counts))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(x - width, selection_counts["RF_Selected"], width, label="RF supported")

ax.bar(x, selection_counts["ANOVA_Selected"], width, label="ANOVA significant")

ax.bar(x + width, selection_counts["Both_Selected"], width, label="Supported by both")

ax.set_xticks(x)
ax.set_xticklabels(selection_counts["Method"], rotation=35, ha="right")

ax.set_ylabel("Number of genes")
ax.set_xlabel("Sampling method")
ax.set_title("Gene selection by sampling method")
ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(FIGURE_DIR / "Gene_Selection_Counts.png", dpi=300, bbox_inches="tight")

fig.savefig(FIGURE_DIR / "Gene_Selection_Counts.pdf", bbox_inches="tight")

plt.show()


# 11. Supplementary feature-stability summary

stability_rows = []
for method, frame in gene_summary_all.groupby("Method"):
    rf_supported = frame["RF_Supported_Runs"] > 0
    anova_supported = frame["ANOVA_Significant_Runs"] > 0
    stability_rows.append(
        {
            "Method": method,
            "Mapped_Genes": frame["Gene"].nunique(),
            "RF_Supported_Genes": int(rf_supported.sum()),
            "ANOVA_Significant_Genes": int(anova_supported.sum()),
            "Supported_By_Both": int((rf_supported & anova_supported).sum()),
            "Median_RF_Mean_Importance": frame["RF_Mean"].median(),
            "Median_ANOVA_F_Mean": frame["ANOVA_F_Mean"].median(),
            "Median_RF_Supported_Runs": frame["RF_Supported_Runs"].median(),
            "Median_ANOVA_Significant_Runs": frame["ANOVA_Significant_Runs"].median(),
        }
    )
stability_summary = pd.DataFrame(stability_rows)
stability_summary.to_csv(
    OUTPUT_DIR / "Supplementary_Feature_Stability_Summary.csv", index=False
)


# Interpretation notes
# - RF and ANOVA statistics are used as ranking evidence, not as formal causal or clinical inference.
# - For synthetic or duplicated samples, ANOVA p-values should be interpreted cautiously; the F-statistic is the main ranking quantity.
# - Cluster Centroids produces synthetic majority-class prototypes.
# - AllKNN is deterministic for fixed data and parameters; its ANOVA results may be identical across runs.
# - Hypergeometric enrichment uses a discrete selected set, while GSEA uses the full continuous ranking.
