# OGH: Ordered Gromov-Hausdorff Metric for Protein Structure Comparison

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the implementation of the **Ordered Gromov-Hausdorff (OGH)** metric for comparing 3D protein structures with respect to amino acid residue order. The metric penalizes residue rearrangements while preserving sensitivity to geometric shape.

**OGH is not a replacement for TM-score, but a complementary tool** that provides unique discriminatory information precisely where TM-score is weakest: in the diverse structural zone (TM ≤ 0.3) and for proteins with substantial length differences.

---

## 📚 About the Research

Traditional metrics like RMSD and TM-score ignore the linear order of residues along the polypeptide chain. This leads to incorrect conclusions when comparing proteins with swapped domains or insertions/deletions. OGH addresses this limitation by combining:

- **Geometric component**: Euclidean distances between Cα atoms after normalization
- **Order component**: Exponential penalty for order violations
- **Monotonic alignment**: Greedy path finding that preserves sequential order

### Key Results (SCOP/ASTRAL Benchmark, 1,400 pairs)

| Metric | Value |
|--------|-------|
| **OGH AUC (Overall)** | 0.665 |
| **TM-score AUC (Overall)** | 0.653 |
| **Combined AUC (Overall)** | **0.731** |
| **OGH AUC (Diverse zone, TM ≤ 0.3)** | 0.587 |
| **TM-score AUC (Diverse zone)** | 0.397 |
| **Improvement in Diverse zone** | **+0.190** |
| **OGH AUC (Length ratio > 2.5)** | 0.677 |
| **TM-score AUC (Length ratio > 2.5)** | 0.369 |
| **Improvement for length ratio > 2.5** | **+0.308** |

### What's new (saturation-constant analysis)

The saturation constant `C = 20` in the order penalty
`w_ord(i,j) = C * (1 - exp(-sigma * |i/n - j/m|))` is empirical. Two new
scripts quantify its role:

| Script | What it does |
|--------|--------------|
| `sensitivity_saturation_constant.py` | AUC and correlation with TM-score as a function of `C` (C ∈ {2, 5, 10, 15, 20, 30, 40, 50, 100}). |
| `order_sensitivity_vs_C.py` | Order sensitivity (shuffling, circular permutation) and relative contribution of the order term as a function of `C` (C ∈ {2, 5, 10, 20, 50}). |

**Key finding.** Smaller `C` improves AUC only by suppressing the order
term: at `C = 2` the order component contributes 5.6% of the final score
(vs 34.2% at `C = 20`), and OGH sensitivity to residue shuffling drops by
16%. Optimising `C` against AUC alone therefore degenerates OGH toward a
purely geometric metric. `C = 20` is retained as the value at which the
order term remains functionally significant while AUC stays competitive.

---

## 📁 Repository Structure
ogh-reproducibility/
│
├── data/ # PDB files (downloaded automatically)
│ └── pdb_scop/ # SCOP/ASTRAL PDB structures
│
├── src/ # Source code
│ ├── parse_homologs.py # Extract sequences from Supplementary Material S1.pdf
│ ├── clean_database.py # Clean and curate the homologs database
│ ├── homology_validation_real_ogh_v2_eng.p.py # MAIN: ROC/AUC validation (1,400 pairs)
│ ├── parameter_sensitivity_analysis_new.py # Sensitivity: w, γ, α
│ ├── sensitivity_saturation_constant.py # NEW: sensitivity to C (AUC/correlation)
│ ├── order_sensitivity_vs_C.py # NEW: order sensitivity vs C
│ ├── compute_all_metrics_scop.py # ALL metrics for Tables 2 & 3
│ ├── generate_figures_v2.py # Generate all figures
│ ├── generate_final_tables.py # Compile final tables
│ ├── generate_supplementary_table_s2.py # Supplementary Table S2
│ ├── homology_validation_length_ratio.py # Length-ratio analysis (Table 7)
│ ├── optimize_combined_model.py # Optimize OGH + TM-score combination
│ └── test_circular_permutation.py # Circular permutation sensitivity test
│
├── results/ # Output (generated)
│ ├── homology_real_ogh_results.csv
│ ├── zone_auc_results.csv
│ ├── sensitivity_analysis.csv
│ ├── sensitivity_saturation_constant.csv # NEW
│ ├── sensitivity_saturation_constant.png # NEW
│ ├── order_sensitivity_vs_C.csv # NEW
│ ├── order_sensitivity_shuffling_vs_C.csv # NEW
│ ├── order_sensitivity_circular_vs_C.csv # NEW
│ ├── order_sensitivity_frac_vs_C.csv # NEW
│ ├── order_sensitivity_vs_C.png # NEW
│ ├── shuffling_results.csv
│ ├── table2_scop.csv
│ ├── table3_scop.csv
│ ├── combined_optimization_results.csv
│ └── circular_permutation_results.csv
│
├── tables/ # Final tables for manuscript
│ ├── table_3_stratified_auc.csv
│ ├── table_4_alpha_optimization.csv
│ ├── table_5_alpha_validation.csv
│ ├── supplementary_table_s3_*.csv
│ ├── supplementary_table_s3b_saturation_constant.csv # NEW
│ └── all_tables.tex
│
├── figures/ # Publication figures
│ ├── fig1_ogh_shuffling.png
│ ├── fig2_ogh_concept.png
│ ├── fig3_ogh_vs_tm_scatter.png
│ ├── figC_order_sensitivity_vs_C.png # NEW
│ ├── combined_optimization_plot.png
│ └── circular_permutation_plot.png
│
├── LICENSE
├── README.md
└── requirements.txt

text

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Internet connection (for downloading PDB files from RCSB)

### Steps

```bash
# Clone repository
git clone https://github.com/andytimoffilim/OGH.git
cd OGH

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Dependencies
text
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
scikit-learn>=1.0.0
biopython>=1.79
matplotlib>=3.4.0
requests>=2.25.0
openpyxl>=3.0.0
📊 Data
SCOP/ASTRAL Dataset
The validation dataset is derived from the SCOP/ASTRAL database (http://astral.berkeley.edu/):

Property	Value
Total protein chains	4,146
Protein pairs	1,400 (700 homologous + 700 non-homologous)
SCOP families	40
Length range	10–500 residues
Input Files Required
File	Description	Source
Supplementary Material S1.pdf	FASTA-like sequences with SCOP annotations	Provided separately
homologs_database_clean.csv	Curated database (generated by Step 1)	Generated
PDB Files
PDB files are automatically downloaded from RCSB (https://files.rcsb.org/) during script execution. No manual download is required.

🏃 Running the Pipeline
Complete Pipeline (All Steps)
bash
# Step 1: Parse and clean data
python src/parse_homologs.py
python src/clean_database.py

# Step 2: Main validation (LONGEST STEP — downloads PDB files, computes OGH for 1,400 pairs)
python src/homology_validation_real_ogh_v2_eng.p.py

# Step 3: Parameter sensitivity analysis (w, γ, α)
python src/parameter_sensitivity_analysis_new.py

# Step 3b: Saturation-constant sensitivity (AUC / correlation)
python src/sensitivity_saturation_constant.py

# Step 3c: Saturation-constant sensitivity (order sensitivity, fraction of order term)
python src/order_sensitivity_vs_C.py

# Step 4: Shuffling test (Table 1, Figure 1)
python src/generate_figures_v2.py

# Step 5: All metrics for Tables 2 & 3
python src/compute_all_metrics_scop.py

# Step 6: Generate final tables
python src/generate_final_tables.py

# Step 7: Generate all figures
python src/generate_figures_v2.py

# Step 8: Supplementary tables
python src/generate_supplementary_table_s2.py

# Step 9: Length-ratio analysis (Table 7)
python src/homology_validation_length_ratio.py

# Step 10: Optimize combined model (optional, validates simple average)
python src/optimize_combined_model.py

# Step 11: Circular permutation test (optional, proof-of-concept)
python src/test_circular_permutation.py
Note on working directory. sensitivity_saturation_constant.py and
order_sensitivity_vs_C.py read homology_real_ogh_results.csv and
homology_coords.pkl from the current working directory. Run them from
the folder where these files live (typically data/ or src/):

bash
cd data
python ../src/sensitivity_saturation_constant.py
python ../src/order_sensitivity_vs_C.py
Pipeline Overview
Step	Script	Input	Output	Description
1	parse_homologs.py	Supplementary Material S1.pdf	homologs_database.csv	Extract sequences and SCOP annotations from PDF
2	clean_database.py	homologs_database.csv	homologs_database_clean.csv	Clean, remove duplicates, validate sequences
3	homology_validation_real_ogh_v2_eng.p.py	homologs_database_clean.csv + PDB	homology_real_ogh_results.csv, zone_auc_results.csv	MAIN: Build 1,400 pairs, compute OGH/TM-score, calculate AUC, generate ROC plots
3b	sensitivity_saturation_constant.py	homology_real_ogh_results.csv + homology_coords.pkl	sensitivity_saturation_constant.csv/.png	AUC and correlation with TM-score across C
3c	order_sensitivity_vs_C.py	homology_real_ogh_results.csv + PDB	order_sensitivity_*.csv, order_sensitivity_vs_C.png	Shuffling/circular permutation and order-term fraction across C
4	parameter_sensitivity_analysis_new.py	homology_real_ogh_results.csv + PDB	sensitivity_analysis.csv	Test w ∈ [10,25], γ ∈ [1.0,3.0], α ∈ [0.0,1.0]
5	generate_figures_v2.py (shuffling)	homologs_database_clean.csv + PDB	shuffling_results.csv, fig1_ogh_shuffling.png	Shuffling test: 5 proteins, 0–100% shuffling
6	compute_all_metrics_scop.py	homology_real_ogh_results.csv	scop_all_metrics.csv, table2_scop.csv, table3_scop.csv	All metrics: OGH, GH, RMSD, Seq-RMSD; correlations with TM-score
7	generate_final_tables.py	homology_real_ogh_results.csv, zone_auc_results.csv, sensitivity_analysis.csv	Tables 3–5 + Supplementary Table S3	Compile final tables in CSV and LaTeX
8	generate_figures_v2.py	homology_real_ogh_results.csv, shuffling_results.csv	All figures	Generate all publication figures
9	generate_supplementary_table_s2.py	homology_real_ogh_results.csv	supplementary_table_s2.csv	Complete list of 1,400 pairs
10	homology_validation_length_ratio.py	homologs_database_clean.csv	supplementary_table_s4_length_ratio.csv	Length-ratio analysis (Table 7)
11	optimize_combined_model.py	homology_real_ogh_results.csv	combined_optimization_results.csv, combined_optimization_plot.png	Compare Simple Average vs Weighted Sum vs Logistic Regression
12	test_circular_permutation.py	homologs_database_clean.csv + PDB	circular_permutation_results.csv, circular_permutation_plot.png	Circular permutation test: Proof-of-concept sensitivity to order rearrangements
📈 Expected Results
Table 3 — AUC values for homology detection across TM-score zones
TM-score zone	N	OGH	TM-score	Combined	Improvement
Diverse (TM ≤ 0.3)	1057	0.587	0.397	0.481	+0.190
Twilight (0.3 < TM ≤ 0.5)	145	0.541	0.689	0.695	-0.148
Homologous (TM > 0.5)	198	0.696	0.988	0.924	-0.292
Overall	1400	0.665	0.653	0.731	+0.012
Table 7 — AUC values by length-ratio bin
Length ratio	N	OGH	TM-score	Combined	Advantage (OGH - TM)
1.0–1.2	600	0.703	0.751	0.783	-0.048
1.2–1.5	600	0.453	0.416	0.433	+0.037
1.5–2.0	200	0.668	0.499	0.715	+0.169
2.0–2.5	144	0.738	0.674	0.785	+0.064
2.5–10.0	226	0.677	0.369	0.638	+0.308
Saturation-constant sensitivity (NEW)
AUC and correlation with TM-score vs C:

C	AUC (overall)	Pearson r	Spearman ρ
2	0.721	-0.744	-0.511
5	0.710	-0.705	-0.458
10	0.694	-0.638	-0.389
15	0.678	-0.578	-0.337
20	0.665	-0.529	-0.302
30	0.647	-0.459	-0.250
40	0.632	-0.408	-0.210
50	0.623	-0.373	-0.188
100	0.598	-0.270	-0.145
Order sensitivity and order-term fraction vs C:

C	OGH at 100% shuffling	OGH at circular perm. (cut=0.7)	frac_order (mean)	frac_order (median)
2	0.619	0.653	0.056	0.055
5	0.641	0.656	0.126	0.127
10	0.677	0.660	0.217	0.225
20	0.738	0.673	0.342	0.368
50	0.896	0.686	0.525	0.590
Interpretation. Smaller C increases AUC only by suppressing the order
term. At C = 2, OGH is 94% geometric and loses the order sensitivity
that defines it. C = 20 is the value at which the order term remains
functionally significant (≈34% of the score) while AUC stays competitive.

Circular Permutation Test
The test_circular_permutation.py script demonstrates OGH sensitivity to order rearrangements:

Cut position	OGH
0.0	0.0000
0.1	0.6366
0.2	0.6575
0.3	0.6693
0.4	0.6711
0.5	0.6239
0.6	0.6684
0.7	0.6793
0.8	0.6586
0.9	0.6359
Conclusion: OGH increases from 0.000 (cut = 0) to a maximum of 0.679 (cut = 0.7), confirming sensitivity to circular permutations.

Combined Model Optimization
Method	AUC (Test)	Accuracy	F1-score
Simple Average (baseline)	0.773	0.682	0.544
Weighted Sum (w=0.58)	0.773	0.718	0.618
Logistic Regression	0.762	0.754	0.718
Conclusion: Simple average is nearly optimal (within 0.04% of weighted sum) and was adopted as the final combined predictor.

🔧 Configuration Parameters
Parameter	Description	Recommended	Range
α (order_weight)	Balance between geometry and order	0.5	0–1
w (window)	Search window for greedy alignment	15	10–25
γ (gap_penalty)	Penalty for skipped residues	2.0	1.0–3.0
σ (sigma)	Order penalty steepness	3.0	—
C (saturation constant)	Upper bound of the order penalty	20	not tuned against AUC; see order_sensitivity_vs_C.py
use_normalization	Normalize coordinates	True	Boolean
Note on C. Unlike w and γ, C should not be tuned
against AUC: smaller C improves AUC only by suppressing the order term
and degenerating OGH toward a purely geometric score. The value C = 20
keeps the order term at ≈34% of the final score while AUC stays above
the TM-score baseline.

📝 Citation
If you use this code in your research, please cite:

bibtex
@article{Timofeev2026OGH,
  title={Ordered Gromov--Hausdorff Metric: A Complementary Tool for Protein Homology Detection},
  author={Timofeev, A.V. and Anufriev, A.S.},
  journal={Journal of Bioinformatics and Computational Biology},
  year={2026},
  note={In revision}
}
📧 Contact
Corresponding author: Andrey V. Timofeev
Email: tav@centeraisco.com