# OGH: Ordered Gromov-Hausdorff Metric for Protein Structure Comparison

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the implementation of the **Ordered Gromov-Hausdorff (OGH)** metric for comparing 3D protein structures with respect to amino acid residue order. The metric penalizes residue rearrangements while preserving sensitivity to geometric shape.

---

## 📚 About the Research

Traditional metrics like RMSD and TM-score ignore the linear order of residues along the polypeptide chain. This leads to incorrect conclusions when comparing proteins with swapped domains or insertions/deletions. OGH addresses this limitation by combining:

- **Geometric component**: Euclidean distances between Cα atoms after normalization
- **Order component**: Exponential penalty for order violations
- **Monotonic alignment**: Greedy/DP path finding that preserves sequential order

The method was validated on the **VAD (Viral Affinity Dataset)** containing 28 viral proteins from HIV-1, SARS-CoV-2, and MERS-CoV.

### Key Results

- OGH monotonically increases with shuffling degree (0 → 0.065)
- Positive correlation with TM-score (r = 0.438) — complementary property
- High correlation with GH approximations (r = 0.745–0.861)
- Optimal order weight α = 0.5–0.7 for balanced comparison

---

## 📁 Repository Structure
ogh_reproducibility/
│
├── data/ # PDB files (not included)
│ └── vad_500mb/ # VAD dataset structures
│ ├── 1E0J.pdb
│ ├── 1F5K.pdb
│ └── ...
│
├── src/ # Source code
│ ├── ogh_VAD_clean.py # Basic OGH implementation
│ ├── ogh_VAD_advanced.py # α-sensitivity & method comparison
│ └── rin_ogh_advanced.py # RIN-OGH with weighted edges
│
├── visualization/ # Visualization scripts
│ ├── visualize_ogh_vs_gh.py # 6 figures: GH vs OGH
│ ├── visualize_ogh_simple.py # 2 figures: simple illustration
│ └── visualize_ogh_different_sizes.py # 4 figures: size effects
│
├── docs/ # Documentation
│ ├── NN_OGH_without_therem.pdf # Full paper (Russian/English)
│ └── Math_OGH.pdf # Mathematical proofs
│
├── results/ # Output (generated)
│ ├── vad_ogh_results.csv
│ ├── alpha_sensitivity.png
│ └── ...
│
├── requirements.txt # Python dependencies
├── setup.py # Package installation
└── README.md # This file


---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- PDB files from VAD dataset (see [Data](#-data) section)

### Steps

```bash
# Clone repository
git clone https://github.com/yourusername/ogh-reproducibility.git
cd ogh-reproducibility

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
Dependencies

numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
biopython>=1.79
matplotlib>=3.4.0     # for visualization
networkx>=2.6.0       # for RIN visualization (optional)
📊 Data
VAD Dataset (Viral Affinity Dataset)
The dataset includes 28 high-quality X-ray structures of viral proteins:

Virus	Proteins	PDB IDs
HIV-1	Protease, Integrase, Reverse transcriptase	1E0J, 1F5K, 1K28, 2WR7, 3HMX, 4N5J, 4ZMJ, 5KOK, 6E6J, 6NUS, 6QJ4, 6VXX, 6W01, 6WAQ, 6Y2E
SARS-CoV-2	Main protease (Mpro), RBD	5T6T, 6LU7, 6M0J, 7BW4, 7KRS, 7L0E, 7N0S, 7RBY, 7T83, 7T9N
SARS-CoV	Protease	6M17
MERS-CoV	Protease	2H2C, 6B8V
Download instructions:

bash
# Create data directory
mkdir -p data/vad_500mb

# Download PDB files (example with wget)
cd data/vad_500mb
for pdb in 1E0J 1F5K 1K28 2H2C 2WR7 3HMX 4N5J 4ZMJ 5KOK 5T6T 6B8V 6E6J 6LU7 6M0J 6M17 6NUS 6QJ4 6VXX 6W01 6WAQ 6Y2E 7BW4 7KRS 7L0E 7N0S 7RBY 7T83 7T9N; do
    wget https://files.rcsb.org/download/${pdb}.pdb
done
🏃 Running the Code
1. Basic OGH Analysis (Main Results)
bash
python src/ogh_VAD_clean.py
Output:

vad_ogh_results.csv — all metrics for each protein pair

vad_ogh_sensitivity.csv — shuffling sensitivity test

Console output with statistics and top/bottom pairs

2. Advanced Analysis (α Optimization & Method Comparison)
bash
python src/ogh_VAD_advanced.py
Output:

alpha_sensitivity.csv — correlation vs α

ordered_methods_comparison.csv — comparison with CE, MATRAS, SALIGN

statistical_significance.csv — p-values and bootstrap CIs

PNG plots (if matplotlib available)

3. RIN-OGH (Residue Interaction Networks)
bash
python src/rin_ogh_advanced.py
Output:

rin_graph_stats.png — graph statistics visualization

Console output with comparison of binary vs weighted edges

4. Visualization
bash
# Generate all 12 figures from the paper
python visualization/visualize_ogh_vs_gh.py
python visualization/visualize_ogh_simple.py
python visualization/visualize_ogh_different_sizes.py
Generated figures:

fig1_gh_concept.png — GH metric concept

fig2_gh_problem.png — GH limitations (order blindness)

fig3_ogh_how_it_works.png — how OGH works

fig4_gh_vs_ogh_comparison.png — GH vs OGH comparison

fig5_ogh_algorithm.png — algorithm flowchart

fig6_vad_results.png — VAD results interpretation

fig7_ogh_different_sizes.png — handling different lengths

fig8_ogh_size_effect.png — size effect analysis

fig9_monotonic_alignment.png — monotonic alignment visualization

fig10_vad_length_effect.png — VAD length distribution

### 5. Publication-Ready Figures

Generate the three final figures used in the article:

```bash
python generate_figures.py

📈 Reproducing Paper Results
Table 2 (Sensitivity to shuffling)
python
from src.ogh_VAD_clean import OrderedGromovHausdorff
import numpy as np

ogh = OrderedGromovHausdorff(order_weight=0.5)
# Run test on 1E0J, 1F5K, 2H2C with shuffling levels 0%, 30%, 50%, 70%, 100%
Table 3 (Basic statistics)
Run ogh_VAD_clean.py — statistics printed to console

Table 4 (Correlations with TM-score)
Results from ogh_VAD_advanced.py — pearsonr/spearmanr calculations

Table 6 (α sensitivity)
Results from alpha_sensitivity.csv generated by ogh_VAD_advanced.py

Table 7 (Comparison with ordered methods)
Results from ordered_methods_comparison.csv

🔧 Configuration Parameters
Parameter	Description	Recommended	Range
order_weight (α)	Balance between geometry and order	0.5	0-1
window	Search window for greedy alignment	15	5-30
gap_penalty	Penalty for skipped residues	2.0	1-10
use_normalization	Normalize coordinates	True	Boolean
max_length_ratio	Max length ratio for comparison	2.0	1.5-3.0
📊 Expected Results
After running ogh_VAD_clean.py, you should see:


TEST: SENSITIVITY TO ORDER
1E0J (500 residues):
   0% shuffled: OGH=0.000000, GH=0.000000
  30% shuffled: OGH=0.003816, GH=0.000000
  50% shuffled: OGH=0.006704, GH=0.000000
  70% shuffled: OGH=0.009990, GH=0.000000
 100% shuffled: OGH=0.017520, GH=0.000000

CORRELATIONS WITH TM-SCORE:
OGH               vs TM_score: r= 0.438
GH KS             vs TM_score: r= 0.416
GH Histogram      vs TM_score: r= 0.168
📝 Citation
If you use this code in your research, please cite:

bibtex
@article{Timofeev2024OGH,
  title={Ordered Gromov-Hausdorff Metric for Comparative Analysis of Viral Protein Structures from VAD Dataset},
  author={Timofeev, A.V. and Anufriev, A.S.},
  journal={AI Center for SCO+ Countries},
  year={2024}
}
🤝 Contributing
Contributions are welcome! Please:

Fork the repository

Create a feature branch

Submit a Pull Request

📧 Contact
Timofeev A.V. - timofeev@ai-center.ru

Anufriev A.S. - anufriev@ai-center.ru

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
VAD (Viral Affinity Dataset) contributors

RCSB Protein Data Bank

Biopython, NumPy, SciPy, pandas communities

Keywords: Gromov-Hausdorff metric, ordered metric, protein structure comparison, RIN, VAD, TM-score, viral proteins, evolutionary bioinformatics

