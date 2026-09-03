# Methylation Imbalance Solver

This repository contains a Python workflow for evaluating missing-value
imputation, studying class-imbalance strategies in DNA methylation data,
ranking CpG markers and genes, and generating PDAC enrichment results and
manuscript figures.

The workflow compares an untreated baseline with random oversampling, SMOTE,
ADASYN, random undersampling, Cluster Centroids, and AllKNN. Random Forest
importance and ANOVA statistics are calculated for each strategy and then
used for gene-level enrichment analysis.

## Repository layout

```text
.
├── data/                                      # Local input data (Git-ignored)
├── Imputation/
│   └── Missing_Value_450K_Imputation_All.py
├── FeatureSelection/
│   ├── NoSampling/
│   │   └── no_sampling.py
│   ├── OverSampling/
│   │   ├── adaptive_synthetic.py
│   │   ├── random_oversampling.py
│   │   └── smote.py
│   └── UnderSampling/
│       ├── allknn.py
│       ├── cluster_centroids.py
│       └── random_undersampling.py
├── hypergeometric_enrichment_continuous_rankings.py
├── PDAC_Methylation_Figure_Generation.py
└── Appendix - TCGA Numbers.csv
```

Generated feature-selection outputs remain next to their corresponding
scripts under `output_files3/`. Enrichment outputs are written to
`continuous_enrichment_results/`. These locations are excluded from Git.

## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- Matplotlib
- SciPy
- scikit-learn
- imbalanced-learn
- gseapy
- openpyxl (optional, for Excel versions of figure summary tables)

One way to install the dependencies is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy pandas matplotlib scipy scikit-learn \
  imbalanced-learn gseapy openpyxl
```

Preranked KEGG analysis uses the gseapy gene-set service and therefore
requires network access unless the gene-set library is already available
locally.

## Local data

Create a `data` directory in the repository root. It is intentionally ignored
by Git so that large or restricted datasets are not committed.

```text
data/
├── BetaData_AllRounded.csv
├── BetaData_SimpleImpute_Zero.csv
└── 450k_All.csv
```

### `BetaData_AllRounded.csv`

Raw methylation matrix used by the imputation stage. It must contain:

- `Donor_Sample`: sample identifier
- `is_tumor`: target value
- One or more numeric CpG columns

The target and identifier columns may occur anywhere in the file.

### `BetaData_SimpleImpute_Zero.csv`

Zero-imputed methylation matrix used by every feature-selection script. It is
created automatically by the imputation stage, although an existing prepared
version can be placed in `data/` if imputation does not need to be rerun.

### `450k_All.csv`

CpG annotation used by enrichment analysis. It must contain:

- `Composite_Element_REF`: CpG marker identifier
- `Gene_Symbol`: gene symbol; multiple symbols may be separated by semicolons

All paths are derived from the scripts' locations, so commands can be launched
from the repository root without editing machine-specific paths.

## Workflow

### 1. Impute missing methylation values

```bash
python3 Imputation/Missing_Value_450K_Imputation_All.py
```

The script removes CpG columns with 30% or more missing values, evaluates zero,
KNN, and mean imputation with five-fold cross-validation, and writes:

- `data/BetaData_SimpleImpute_Zero.csv`
- `data/BetaData_SimpleImpute_KNN.csv`
- `data/BetaData_SimpleImpute_Mean.csv`
- `data/CpG_Missingness.csv`
- `data/Missingness_Distribution.png`
- `data/Imputation_Comparison.csv`
- `data/Imputation_Comparison.png`

Balanced classification accuracy is the default evaluation metric:

```bash
python3 Imputation/Missing_Value_450K_Imputation_All.py \
  --metric balanced_accuracy
```

To reproduce the earlier regression-based comparison:

```bash
python3 Imputation/Missing_Value_450K_Imputation_All.py \
  --metric regression_mse
```

Additional options are available through:

```bash
python3 Imputation/Missing_Value_450K_Imputation_All.py --help
```

### 2. Run feature selection

Run the baseline and desired resampling strategies from the repository root:

```bash
python3 FeatureSelection/NoSampling/no_sampling.py

python3 FeatureSelection/OverSampling/random_oversampling.py
python3 FeatureSelection/OverSampling/smote.py
python3 FeatureSelection/OverSampling/adaptive_synthetic.py

python3 FeatureSelection/UnderSampling/random_undersampling.py
python3 FeatureSelection/UnderSampling/cluster_centroids.py
python3 FeatureSelection/UnderSampling/allknn.py
```

Each strategy writes per-run Random Forest and ANOVA results beneath its own
`output_files3/` directory. Typical files include:

- `Meth_RF_Statistics_Run<N>.csv`
- `Meth_ANOVA_Statistics_Run<N>.csv`
- `Meth_Impt_Features<N>RF.csv`
- `Meth_Impt_Features<N>Anova.csv`

The baseline runs Random Forest ten times and ANOVA once. Resampling strategies
produce ten Random Forest and ten ANOVA runs.

### 3. Run gene-level enrichment

After all feature-selection outputs are present, run:

```bash
python3 hypergeometric_enrichment_continuous_rankings.py
```

This stage:

- Maps CpG markers to genes
- Aggregates Random Forest and ANOVA evidence by method and run
- Builds continuous gene rankings
- Tests overlap with a curated PDAC reference set using a hypergeometric test
- Runs preranked KEGG GSEA
- Writes tables and figures under `continuous_enrichment_results/`

Important outputs include:

- `Gene_Statistics_Across_Runs.csv`
- `Continuous_Gene_Rankings_All_Methods.csv`
- `Hypergeometric_PDAC_Gene_Enrichment.csv`
- `PDAC_Pathway_NES_Summary.csv`
- `Ranking_Tie_Diagnostics.csv`
- `Selected_Gene_Set_Summary.csv`
- `Supplementary_Feature_Stability_Summary.csv`
- `gsea_full_ranked/`

### 4. Generate manuscript figures

```bash
python3 PDAC_Methylation_Figure_Generation.py
```

The figure script reads existing enrichment results and writes PNG, PDF, and
SVG figures under:

```text
continuous_enrichment_results/figures/manuscript/
continuous_enrichment_results/figures/supplementary/
```

Tables are written to:

```text
continuous_enrichment_results/summary_tables/
```

The script also creates `Figure_Manifest.csv` and `Methods_Report.txt`.

In addition to the enrichment outputs listed above, figure generation expects:

```text
continuous_enrichment_results/summary_tables/
└── Figure3_Continuous_Score_Threshold_Data.csv
```

Ensure this table exists before running the complete figure script.

## Reproducibility and interpretation

- Random seeds are fixed where the underlying method is stochastic.
- Feature-selection output and local data directories are excluded from Git;
  archive important results separately.
- Random Forest importance and ANOVA F statistics provide ranking evidence,
  not causal or clinical inference.
- Synthetic and duplicated samples can affect the interpretation of ANOVA
  p-values.
- The continuous enrichment score combines standardized Random Forest and
  ANOVA evidence. It does not indicate biological effect direction.
- Hypergeometric enrichment depends on the chosen measured-gene universe and
  curated PDAC reference set.

## License

See [LICENSE](LICENSE).
