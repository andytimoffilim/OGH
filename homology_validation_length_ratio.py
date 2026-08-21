#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
homology_validation_length_ratio.py — OGH validation with length ratio analysis

Description:
    Validates OGH on proteins with controlled length differences.
    Tests the hypothesis that OGH's advantages become more pronounced
    as the length ratio between compared proteins increases.

Key Features:
    - Filters protein pairs by length ratio (max/min)
    - Compares OGH vs TM-score performance across length ratio bins
    - Tests length ratios: 1.0–1.2, 1.2–1.5, 1.5–2.0, 2.0–2.5, 2.5–3.0, 3.0–5.0

Input:
    homologs_database_clean.csv — Curated database with SCOP annotations

Output:
    length_ratio_results.csv — All metrics with length ratio annotation
    length_ratio_analysis.csv — AUC by length ratio bins
    length_ratio_plot.png — Visualization of AUC vs length ratio

Parameters (configurable):
    MAX_PAIRS = 300               — Pairs per length ratio bin
    MAX_FAMILIES = 40             — SCOP families to use
    LENGTH_RATIO_MIN = 1.0        — Minimum length ratio (optional filter)
    LENGTH_RATIO_MAX = 10.0       — Maximum length ratio (optional filter)

Dependencies:
    numpy, scipy, pandas, sklearn, matplotlib, Bio.PDB, requests

Usage:
    python homology_validation_length_ratio.py

Author: Andrey V. Timofeev, Alexandr S. Anufriev
Date: 2026
License: MIT
"""

import os
import warnings
import requests
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import svd as scipy_svd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, Superimposer, Structure, Model, Chain, Residue, Atom

warnings.filterwarnings('ignore')

# ============================================================================
# 0. FIX PROTEINBUILDER ERROR
# ============================================================================

def coords_to_structure_fixed(coords, name):
    """Create structure from coordinates (without ProteinBuilder)."""
    struct = Structure.Structure(name)
    model = Model.Model(0)
    chain = Chain.Chain('A')

    for i, (x, y, z) in enumerate(coords):
        res = Residue.Residue((' ', i + 1, ' '), 'ALA', ' ')
        atom = Atom.Atom('CA', np.array([x, y, z]), 1.0, 1.0, 'C', 'CA', 'ALA')
        res.add(atom)
        chain.add(res)

    model.add(chain)
    struct.add(model)
    return struct

def tm_score_biopython(X, Y):
    """TM-score via Biopython Superimposer."""
    struct1 = coords_to_structure_fixed(X, 'prot1')
    struct2 = coords_to_structure_fixed(Y, 'prot2')

    atoms1 = [a for a in struct1.get_atoms() if a.get_name() == 'CA']
    atoms2 = [a for a in struct2.get_atoms() if a.get_name() == 'CA']

    n = min(len(atoms1), len(atoms2))
    atoms1 = atoms1[:n]
    atoms2 = atoms2[:n]

    superimposer = Superimposer()
    superimposer.set_atoms(atoms1, atoms2)
    superimposer.apply(atoms2)

    coords1 = np.array([a.get_coord() for a in atoms1])
    coords2 = np.array([a.get_coord() for a in atoms2])

    distances = np.linalg.norm(coords1 - coords2, axis=1)
    d0 = 1.24 * (n ** (1/3) - 1) + 1.5
    tm = np.sum(1.0 / (1 + (distances / d0) ** 2)) / n
    return tm

def tm_score_fallback(X, Y):
    """Fallback TM-score implementation."""
    from scipy.optimize import linear_sum_assignment

    n, m = len(X), len(Y)
    L_min = min(n, m)

    if n != m:
        dist_matrix = cdist(X, Y)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        X_subset = X[row_ind]
        Y_subset = Y[col_ind]
    else:
        X_subset, Y_subset = X.copy(), Y.copy()

    X_centered = X_subset - np.mean(X_subset, axis=0)
    Y_centered = Y_subset - np.mean(Y_subset, axis=0)

    H = X_centered.T @ Y_centered
    U, S, Vt = scipy_svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    X_aligned = X_centered @ R.T
    distances = np.linalg.norm(X_aligned - Y_centered, axis=1)

    d0 = 1.24 * (L_min ** (1.0/3.0) - 1.0) + 1.5
    return np.sum(1.0 / (1 + (distances / d0) ** 2)) / L_min

def tm_score_verified(X, Y):
    """Validated TM-score implementation."""
    try:
        return tm_score_biopython(X, Y)
    except Exception as e:
        print(f"⚠️  Biopython error: {e}, using fallback")
        return tm_score_fallback(X, Y)

# ============================================================================
# 1. EXPERIMENT PARAMETERS
# ============================================================================

MAX_PAIRS_PER_BIN = 300      # Pairs per length ratio bin
MAX_FAMILIES = 40            # SCOP families to use
ORDER_WEIGHT = 0.5           # α for scoring
WINDOW_SIZE = 15             # Window size w
GAP_PENALTY = 2.0            # Gap penalty γ
SIGMA = 3.0                  # Order penalty parameter

# Length ratio bins for stratification
# LENGTH_BINS = [0, 1.2, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
# LENGTH_LABELS = ['1.0-1.2', '1.2-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-5.0', '5.0-10.0']

LENGTH_BINS = [0, 1.2, 1.5, 2.0, 2.5, 10.0]
LENGTH_LABELS = ['1.0-1.2', '1.2-1.5', '1.5-2.0', '2.0-2.5', '2.5-10.0']

print(f"""
⚙️  EXPERIMENT PARAMETERS:
   MAX_PAIRS_PER_BIN = {MAX_PAIRS_PER_BIN}
   MAX_FAMILIES      = {MAX_FAMILIES}
   ORDER_WEIGHT (α)  = {ORDER_WEIGHT}
   WINDOW_SIZE (w)   = {WINDOW_SIZE}
   GAP_PENALTY (γ)   = {GAP_PENALTY}
   SIGMA (σ)         = {SIGMA}
   LENGTH BINS       = {LENGTH_LABELS}
""")


# ============================================================================
# 2. OGH CLASS (optimized, vectorized)
# ============================================================================

class OrderedGromovHausdorff:
    def __init__(self, alpha=0.5, align_sigma=3.0, align_order_weight=1.0,
                 window_size=15, gap_penalty=2.0, use_normalization=True):
        self.alpha = alpha
        self.align_sigma = align_sigma
        self.align_order_weight = align_order_weight
        self.window_size = window_size
        self.gap_penalty = gap_penalty
        self.use_normalization = use_normalization

    @staticmethod
    def _build_contact_map(coords):
        return cdist(coords, coords, metric='euclidean')

    def compute(self, X, Y):
        n, m = len(X), len(Y)
        L = min(n, m)

        if self.use_normalization:
            X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
            Y = (Y - np.mean(Y, axis=0)) / (np.std(Y, axis=0) + 1e-8)

        D_X = self._build_contact_map(X)
        D_Y = self._build_contact_map(Y)

        D_X = D_X[:, :L]
        D_Y = D_Y[:, :L]

        # Vectorized geometry
        d_geom = cdist(D_X, D_Y, metric='euclidean')

        # Order penalty
        i_norm = np.arange(n).reshape(-1, 1) / n
        j_norm = np.arange(m).reshape(1, -1) / m
        w_ord = 20.0 * (1.0 - np.exp(-self.align_sigma * np.abs(i_norm - j_norm)))

        # Alignment
        cost_align = d_geom + self.align_order_weight * w_ord
        alignment = self._greedy_monotone_alignment(cost_align)

        if len(alignment) == 0:
            return np.inf, {}

        align_i, align_j = zip(*alignment)
        matched_geom = d_geom[align_i, align_j]
        matched_ord = w_ord[align_i, align_j]
        k = len(alignment)

        metric_dist = np.mean(matched_geom)
        order_dist = np.mean(matched_ord)

        ogh_dist = ((1.0 - self.alpha) * metric_dist + self.alpha * order_dist) / np.sqrt(L)

        return ogh_dist, {
            'metric_component': metric_dist,
            'order_component': order_dist,
            'n_matches': k,
            'coverage': k / max(n, m),
            'alpha': self.alpha,
            'len1': n,
            'len2': m
        }

    def _greedy_monotone_alignment(self, cost_matrix):
        n, m = cost_matrix.shape
        w = self.window_size
        gamma = self.gap_penalty
        alignment = []
        i, j = 0, 0

        while i < n and j < m:
            win_i = min(w, n - i)
            win_j = min(w, m - j)
            best_cost = np.inf
            best_di, best_dj = 1, 1

            for di in range(1, win_i + 1):
                for dj in range(1, win_j + 1):
                    gap = gamma * ((di - 1) + (dj - 1))
                    c = cost_matrix[i + di - 1, j + dj - 1] + gap
                    if c < best_cost:
                        best_cost = c
                        best_di, best_dj = di, dj

            i += best_di
            j += best_dj
            alignment.append((i - 1, j - 1))

        return alignment


# ============================================================================
# 3. PDB LOADING
# ============================================================================

def download_pdb_for_id(pdb_id, output_dir='pdb_scop'):
    os.makedirs(output_dir, exist_ok=True)
    clean_id = pdb_id
    if clean_id.startswith('d'):
        clean_id = clean_id[1:]
    if clean_id.endswith('_'):
        clean_id = clean_id[:-1]
    clean_id = clean_id[:4].upper()

    pdb_file = os.path.join(output_dir, f"{clean_id}.pdb")
    if os.path.exists(pdb_file):
        return pdb_file

    url = f"https://files.rcsb.org/download/{clean_id}.pdb"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(pdb_file, 'wb') as f:
                f.write(response.content)
            return pdb_file
    except Exception:
        pass
    return None


def load_pdb_coords(pdb_file):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('prot', pdb_file)
        ca_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        ca_coords.append(residue['CA'].get_coord())
        return np.array(ca_coords) if ca_coords else None
    except Exception:
        return None


# ============================================================================
# 4. PAIR BUILDING WITH LENGTH RATIO FILTERING
# ============================================================================

def build_pairs_by_length_bin(df, bin_idx):
    """
    Build pairs for a specific length ratio bin.
    Ensures balanced homologous/non-homologous pairs.
    """
    ratio_min = LENGTH_BINS[bin_idx]
    ratio_max = LENGTH_BINS[bin_idx + 1]
    label = LENGTH_LABELS[bin_idx]

    print(f"\n  Building pairs for bin {label} (ratio: {ratio_min:.1f}-{ratio_max:.1f})...")

    df = df[df['sequence'].notna() & (df['sequence'] != '')].copy()
    families = df[df['family'] != '']['family'].unique()

    pairs = []
    homolog_count = 0
    non_homolog_count = 0

    # Homologous pairs (same family)
    for family in families[:MAX_FAMILIES]:
        proteins = df[df['family'] == family]
        ids = proteins['id'].tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                len1 = proteins[proteins['id'] == ids[i]]['sequence_length'].iloc[0]
                len2 = proteins[proteins['id'] == ids[j]]['sequence_length'].iloc[0]
                ratio = max(len1, len2) / min(len1, len2)

                if ratio < ratio_min or ratio >= ratio_max:
                    continue

                pdb1 = download_pdb_for_id(ids[i], 'pdb_scop')
                pdb2 = download_pdb_for_id(ids[j], 'pdb_scop')
                if pdb1 and pdb2:
                    coords1 = load_pdb_coords(pdb1)
                    coords2 = load_pdb_coords(pdb2)
                    if coords1 is not None and coords2 is not None:
                        pairs.append({
                            'protein1': ids[i], 'protein2': ids[j],
                            'coords1': coords1, 'coords2': coords2,
                            'is_homolog': 1, 'family': family,
                            'len1': len1, 'len2': len2,
                            'length_ratio': ratio,
                            'length_bin': label
                        })
                        homolog_count += 1
                        if homolog_count >= MAX_PAIRS_PER_BIN:
                            break
            if homolog_count >= MAX_PAIRS_PER_BIN:
                break
        if homolog_count >= MAX_PAIRS_PER_BIN:
            break

    print(f"    Homologous pairs: {homolog_count}")

    # Non-homologous pairs (different families)
    np.random.seed(42)
    all_ids = df['id'].tolist()
    attempts = 0
    max_attempts = 50000

    while non_homolog_count < homolog_count and attempts < max_attempts:
        attempts += 1
        p1, p2 = np.random.choice(all_ids, 2, replace=False)
        f1 = df[df['id'] == p1]['family'].iloc[0] if not df[df['id'] == p1].empty else ''
        f2 = df[df['id'] == p2]['family'].iloc[0] if not df[df['id'] == p2].empty else ''

        if f1 != f2 and f1 != '' and f2 != '':
            len1 = df[df['id'] == p1]['sequence_length'].iloc[0]
            len2 = df[df['id'] == p2]['sequence_length'].iloc[0]
            ratio = max(len1, len2) / min(len1, len2)

            if ratio < ratio_min or ratio >= ratio_max:
                continue

            pdb1 = download_pdb_for_id(p1, 'pdb_scop')
            pdb2 = download_pdb_for_id(p2, 'pdb_scop')
            if pdb1 and pdb2:
                coords1 = load_pdb_coords(pdb1)
                coords2 = load_pdb_coords(pdb2)
                if coords1 is not None and coords2 is not None:
                    pairs.append({
                        'protein1': p1, 'protein2': p2,
                        'coords1': coords1, 'coords2': coords2,
                        'is_homolog': 0, 'family': f"{f1} vs {f2}",
                        'len1': len1, 'len2': len2,
                        'length_ratio': ratio,
                        'length_bin': label
                    })
                    non_homolog_count += 1

    print(f"    Non-homologous pairs: {non_homolog_count}")
    print(f"    Total pairs: {len(pairs)}")

    return pd.DataFrame(pairs)


# ============================================================================
# 5. METRIC COMPUTATION
# ============================================================================

def compute_metrics_for_pairs(df_pairs):
    """Compute OGH and TM-score for all pairs."""
    print(f"\n  Computing metrics for {len(df_pairs)} pairs...")

    ogh = OrderedGromovHausdorff(
        alpha=ORDER_WEIGHT,
        align_sigma=SIGMA,
        align_order_weight=1.0,
        window_size=WINDOW_SIZE,
        gap_penalty=GAP_PENALTY,
        use_normalization=True
    )

    results = []
    for idx, row in df_pairs.iterrows():
        X = row['coords1']
        Y = row['coords2']

        ogh_dist, info = ogh.compute(X, Y)
        tm = tm_score_verified(X, Y)

        results.append({
            'protein1': row['protein1'],
            'protein2': row['protein2'],
            'is_homolog': row['is_homolog'],
            'family': row['family'],
            'd_hat_OGH': ogh_dist,
            'TM_score': tm,
            'coverage': info.get('coverage', np.nan),
            'n_matches': info.get('n_matches', np.nan),
            'len1': len(X),
            'len2': len(Y),
            'length_ratio': row.get('length_ratio', max(len(X), len(Y)) / min(len(X), len(Y))),
            'length_bin': row.get('length_bin', 'unknown'),
            'alpha': ORDER_WEIGHT,
            'window_size': WINDOW_SIZE,
            'gap_penalty': GAP_PENALTY
        })

        if (idx + 1) % 50 == 0:
            print(f"    Processed {idx + 1}/{len(df_pairs)} pairs...")

    return pd.DataFrame(results)


# ============================================================================
# 6. AUC ANALYSIS BY LENGTH BIN
# ============================================================================

def analyze_by_length_bin(df_metrics):
    """
    Compute AUC for each length ratio bin.
    """
    print("\n" + "=" * 80)
    print("📊 AUC BY LENGTH RATIO BIN")
    print("=" * 80)

    results = []

    for label in LENGTH_LABELS:
        group_df = df_metrics[df_metrics['length_bin'] == label]
        if len(group_df) < 10:
            continue

        y_true = group_df['is_homolog'].values

        # OGH
        fpr_ogh, tpr_ogh, _ = roc_curve(y_true, -group_df['d_hat_OGH'].values)
        auc_ogh = auc(fpr_ogh, tpr_ogh)

        # TM-score
        fpr_tm, tpr_tm, _ = roc_curve(y_true, group_df['TM_score'].values)
        auc_tm = auc(fpr_tm, tpr_tm)

        # Combined
        ogh_norm = 1.0 - (group_df['d_hat_OGH'] - group_df['d_hat_OGH'].min()) / \
                         (group_df['d_hat_OGH'].max() - group_df['d_hat_OGH'].min() + 1e-8)
        tm_norm = (group_df['TM_score'] - group_df['TM_score'].min()) / \
                  (group_df['TM_score'].max() - group_df['TM_score'].min() + 1e-8)
        combined = 0.5 * ogh_norm + 0.5 * tm_norm
        fpr_comb, tpr_comb, _ = roc_curve(y_true, combined.values)
        auc_comb = auc(fpr_comb, tpr_comb)

        # Length ratio as control
        length_ratio = group_df['len1'] / (group_df['len2'] + 1)
        fpr_len, tpr_len, _ = roc_curve(y_true, length_ratio.values)
        auc_len = auc(fpr_len, tpr_len)

        improvement = auc_comb - max(auc_ogh, auc_tm)

        results.append({
            'length_bin': label,
            'n_pairs': len(group_df),
            'auc_ogh': auc_ogh,
            'auc_tm': auc_tm,
            'auc_combined': auc_comb,
            'auc_len': auc_len,
            'improvement': improvement
        })

        print(f"\n  {label}:")
        print(f"    Pairs: {len(group_df)}")
        print(f"    AUC OGH:      {auc_ogh:.3f}")
        print(f"    AUC TM-score: {auc_tm:.3f}")
        print(f"    AUC Combined: {auc_comb:.3f}")
        print(f"    AUC Length:   {auc_len:.3f} (control)")
        print(f"    Improvement:  {improvement:+.3f}")

    return pd.DataFrame(results)


# ============================================================================
# 7. VISUALIZATION
# ============================================================================

def plot_length_ratio_results(df_length):
    """
    Create bar plot of AUC vs length ratio.
    """
    if df_length.empty:
        print("No data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: AUC vs length ratio
    ax = axes[0]
    x = np.arange(len(df_length))
    width = 0.25

    ax.bar(x - width, df_length['auc_ogh'], width, label='OGH', color='red', alpha=0.7)
    ax.bar(x, df_length['auc_tm'], width, label='TM-score', color='blue', alpha=0.7)
    ax.bar(x + width, df_length['auc_combined'], width, label='Combined', color='green', alpha=0.7)

    ax.set_xlabel('Length ratio (max/min)')
    ax.set_ylabel('AUC')
    ax.set_title('AUC vs Length Ratio: OGH vs TM-score')
    ax.set_xticks(x)
    ax.set_xticklabels(df_length['length_bin'])
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Plot 2: Improvement (Combined vs best single)
    ax = axes[1]
    ax.bar(df_length['length_bin'], df_length['improvement'], color='purple', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.set_xlabel('Length ratio (max/min)')
    ax.set_ylabel('Improvement (Combined - best single)')
    ax.set_title('Improvement from combining OGH + TM-score')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('length_ratio_analysis_plot.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved: length_ratio_analysis_plot.png")
    plt.show()


# ============================================================================
# 8. SUPPLEMENTARY TABLE GENERATOR
# ============================================================================

def generate_supplementary_table(df_metrics, output_file):
    """
    Generate Supplementary Table S4 with length ratio analysis.
    """
    print(f"\n📋 Generating Supplementary Table S4: {output_file}")

    # Select key columns for the table
    table = df_metrics[['protein1', 'protein2', 'is_homolog', 'length_bin',
                        'length_ratio', 'd_hat_OGH', 'TM_score']].copy()
    table.columns = ['Protein 1', 'Protein 2', 'Is Homolog', 'Length Ratio Bin',
                     'Length Ratio', '\\hat{d}_OGH', 'TM-score']

    # Sort by length bin and ratio
    table = table.sort_values(['Length Ratio Bin', 'Length Ratio']).reset_index(drop=True)

    # Add Pair ID
    table.insert(0, 'Pair ID', [f"P{str(i+1).zfill(4)}" for i in range(len(table))])

    # Save
    table.to_csv(output_file, index=False, encoding='utf-8')
    print(f"  ✅ Saved: {output_file}")

    return table


# ============================================================================
# 9. MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("🔬 LENGTH RATIO ANALYSIS: OGH vs TM-score")
    print("   Testing OGH's advantage on proteins of different lengths")
    print("=" * 80)

    # 1. Load database
    print("\n1. Loading homologs_database_clean.csv...")
    try:
        df = pd.read_csv('homologs_database_clean.csv')
        print(f"   Loaded {len(df)} records")
    except FileNotFoundError:
        print("   ❌ File not found!")
        return

    # 2. Process each length bin
    all_results = []
    bin_summary = []

    for bin_idx in range(len(LENGTH_BINS) - 1):
        label = LENGTH_LABELS[bin_idx]
        print(f"\n{'='*60}")
        print(f"PROCESSING BIN: {label}")
        print(f"{'='*60}")

        df_pairs = build_pairs_by_length_bin(df, bin_idx)

        if len(df_pairs) == 0:
            print(f"  ⚠️ No pairs found for bin {label}")
            continue

        df_metrics = compute_metrics_for_pairs(df_pairs)
        all_results.append(df_metrics)

        # Save bin results
        df_metrics.to_csv(f'length_ratio_bin_{bin_idx}_{label.replace("-", "_")}.csv', index=False)

    if not all_results:
        print("\n❌ No results collected. Check the data.")
        return

    # 3. Combine all results
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv('length_ratio_all_results.csv', index=False)
    print(f"\n✅ All results saved: length_ratio_all_results.csv")

    # 4. Analyze by length bin
    df_length = analyze_by_length_bin(df_all)
    df_length.to_csv('length_ratio_analysis.csv', index=False)
    print(f"\n✅ Length ratio analysis saved: length_ratio_analysis.csv")

    # 5. Visualization
    plot_length_ratio_results(df_length)

    # 6. Generate Supplementary Table
    generate_supplementary_table(df_all, 'supplementary_table_s4_length_ratio.csv')

    # 7. Summary
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)

    for _, row in df_length.iterrows():
        print(f"  {row['length_bin']}: OGH={row['auc_ogh']:.3f}, TM={row['auc_tm']:.3f}, Combined={row['auc_combined']:.3f}")

    # Check if OGH improves with length ratio
    if len(df_length) >= 2:
        ogh_first = df_length.iloc[0]['auc_ogh']
        ogh_last = df_length.iloc[-1]['auc_ogh']
        tm_first = df_length.iloc[0]['auc_tm']
        tm_last = df_length.iloc[-1]['auc_tm']

        print(f"\n  OGH improvement from shortest to longest: {ogh_last - ogh_first:+.3f}")
        print(f"  TM improvement from shortest to longest: {tm_last - tm_first:+.3f}")

        if (ogh_last - ogh_first) > (tm_last - tm_first):
            print("  ✅ OGH shows GREATER improvement with length ratio than TM-score")
        else:
            print("  ⚠️ OGH improvement with length ratio is comparable to or less than TM-score")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()