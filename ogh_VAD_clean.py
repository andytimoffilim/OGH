"""
ogh_VAD_clean.py - Ordered Gromov-Hausdorff Metric for VAD Viral Proteins

Authors: Timofeev A.V., Anufriev A.S.
Organization: AI Center for SCO+ Countries, St. Petersburg, Russia
Date: 2024
License: MIT

Description:
    Implementation of the Ordered Gromov-Hausdorff (OGH) metric for comparing
    3D protein structures with respect to amino acid residue order along the
    polypeptide chain.

Features:
    - Coordinate normalization (centering and scaling)
    - Exponential penalty for order violations
    - Monotonic alignment (greedy algorithm with window w=15)
    - Comparison with classical metrics (GH, RMSD, TM-score, Seq-RMSD)

Dependencies:
    numpy, scipy, Bio.PDB (Biopython), pandas

Input:
    PDB files in the vad_500mb/ directory (VAD viral protein dataset)

Output:
    - vad_ogh_results.csv - metric comparison results for all pairs
    - vad_ogh_sensitivity.csv - shuffling sensitivity test results
    - vad_metrics_correlation.csv - correlation matrix between metrics

Usage:
    >>> python ogh_VAD_clean.py

References:
    - Gromov, M. (1981). Groups of polynomial growth and expanding maps.
    - Zhang, Y., & Skolnick, J. (2004). TM-score: a scoring function.
    - Kabsch, W. (1976). A solution for the best rotation to relate two sets.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.linalg import svd
from Bio.PDB import PDBParser
import glob
import os
import time
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 1. OGH - УПОРЯДОЧЕННАЯ МЕТРИКА ГРОМОВА-ХАУСДОРФА
# ============================================================================

class OrderedGromovHausdorff:
    """Ordered Gromov-Hausdorff метрика с учётом порядка остатков"""

    def __init__(self, order_weight=0.5, use_normalization=True):
        self.order_weight = order_weight
        self.use_normalization = use_normalization

    def compute(self, X, Y):
        n, m = len(X), len(Y)

        if self.use_normalization:
            X_norm = (X - np.mean(X, axis=0)) / (np.std(X) + 1e-8)
            Y_norm = (Y - np.mean(Y, axis=0)) / (np.std(Y) + 1e-8)
        else:
            X_norm, Y_norm = X, Y

        dist_3d = cdist(X_norm, Y_norm)

        order_penalty = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                pos_diff = abs(i / n - j / m)
                order_penalty[i, j] = self.order_weight * 20 * (1 - np.exp(-pos_diff * 3))

        total_cost = dist_3d + order_penalty
        alignment = self._monotonic_alignment(total_cost)

        if len(alignment) == 0:
            return np.inf, {}

        matched_dist = [dist_3d[i, j] for i, j in alignment]
        matched_order = [order_penalty[i, j] for i, j in alignment]

        metric_dist = np.mean(matched_dist)
        order_dist = np.mean(matched_order)

        ogh_dist = (1 - self.order_weight) * metric_dist + self.order_weight * order_dist
        ogh_dist = ogh_dist / np.sqrt(min(n, m))

        return ogh_dist, {
            'metric_component': metric_dist,
            'order_component': order_dist,
            'n_matches': len(alignment),
            'coverage': len(alignment) / max(n, m)
        }

    def _monotonic_alignment(self, cost_matrix):
        n, m = cost_matrix.shape
        alignment = []
        i, j = 0, 0
        visited_i, visited_j = set(), set()

        while i < n and j < m:
            window = min(15, n - i, m - j)
            best_cost = np.inf
            best_i, best_j = i, j

            for di in range(window):
                for dj in range(window):
                    if (i + di) not in visited_i and (j + dj) not in visited_j:
                        cost = cost_matrix[i + di, j + dj] + (di + dj) * 2
                        if cost < best_cost:
                            best_cost = cost
                            best_i, best_j = i + di, j + dj

            if best_cost < np.inf:
                alignment.append((best_i, best_j))
                visited_i.add(best_i)
                visited_j.add(best_j)
                i = best_i + 1
                j = best_j + 1
            else:
                i += 1
                j += 1

        return alignment


# ============================================================================
# 2. КЛАССИЧЕСКИЕ МЕТРИКИ
# ============================================================================

def gh_histogram(X, Y, bins=50):
    if len(X) < 2 or len(Y) < 2:
        return np.inf
    triu_X = pdist(X)
    triu_Y = pdist(Y)
    hist_X, bin_edges = np.histogram(triu_X, bins=bins, density=True)
    hist_Y, _ = np.histogram(triu_Y, bins=bin_edges, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return wasserstein_distance(bin_centers, bin_centers, hist_X, hist_Y)


def gh_ks(X, Y):
    if len(X) < 2 or len(Y) < 2:
        return np.inf
    triu_X = pdist(X)
    triu_Y = pdist(Y)
    ks_stat, _ = ks_2samp(triu_X, triu_Y)
    return ks_stat * 50


def rmsd_with_alignment(X, Y):
    n, m = len(X), len(Y)
    if n != m:
        dist_matrix = cdist(X, Y)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        X_subset = X[row_ind]
        Y_subset = Y[col_ind]
    else:
        X_subset = X
        Y_subset = Y

    X_centered = X_subset - np.mean(X_subset, axis=0)
    Y_centered = Y_subset - np.mean(Y_subset, axis=0)

    H = X_centered.T @ Y_centered
    U, S, Vt = svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    X_aligned = X_centered @ R.T
    diff = X_aligned - Y_centered
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def tm_score(X, Y):
    n, m = len(X), len(Y)
    if n != m:
        dist_matrix = cdist(X, Y)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        X_subset = X[row_ind]
        Y_subset = Y[col_ind]
    else:
        X_subset = X
        Y_subset = Y

    X_centered = X_subset - np.mean(X_subset, axis=0)
    Y_centered = Y_subset - np.mean(Y_subset, axis=0)

    H = X_centered.T @ Y_centered
    U, S, Vt = svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    X_aligned = X_centered @ R.T
    distances = np.linalg.norm(X_aligned - Y_centered, axis=1)

    L = min(n, m)
    d0 = 1.24 * (L ** (1 / 3) - 1) + 1.5
    return np.sum(1.0 / (1 + (distances / d0) ** 2)) / L


def seq_aligned_rmsd(X, Y):
    min_len = min(len(X), len(Y))
    X_sub = X[:min_len]
    Y_sub = Y[:min_len]
    X_centered = X_sub - np.mean(X_sub, axis=0)
    Y_centered = Y_sub - np.mean(Y_sub, axis=0)
    return np.sqrt(np.mean(np.sum((X_centered - Y_centered) ** 2, axis=1)))


# ============================================================================
# 3. ЗАГРУЗКА ДАННЫХ
# ============================================================================

def load_vad_structures(pdb_dir='vad_500mb', max_atoms=500):
    parser = PDBParser(QUIET=True)
    pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))

    structures = {}
    print(f"\nЗагрузка структур из {pdb_dir}:")

    for pdb_file in pdb_files:
        name = os.path.basename(pdb_file)
        try:
            structure = parser.get_structure('prot', pdb_file)
            model = structure[0]
            ca_coords = []
            for chain in model:
                for residue in chain:
                    if 'CA' in residue and len(ca_coords) < max_atoms:
                        ca_coords.append(residue['CA'].get_coord())

            if len(ca_coords) >= 10:
                structures[name] = np.array(ca_coords)
                print(f"  {name}: {len(ca_coords)} CA")
        except:
            pass

    print(f"\nЗагружено: {len(structures)} структур")
    return structures


# ============================================================================
# 4. СРАВНЕНИЕ МЕТРИК
# ============================================================================

def compare_metrics(structures, order_weight=0.5, max_pairs=200):
    names = list(structures.keys())
    n = len(names)
    n_pairs = min(n * (n - 1) // 2, max_pairs) if max_pairs else n * (n - 1) // 2

    print(f"\nСравнение {n} белков, {n_pairs} пар, вес порядка={order_weight}")

    ogh = OrderedGromovHausdorff(order_weight=order_weight)
    results = []
    processed = 0

    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i < j and processed < n_pairs:
                X = structures[name1]
                Y = structures[name2]
                processed += 1

                t0 = time.time()
                ogh_dist, info = ogh.compute(X, Y)
                t_ogh = time.time() - t0

                t0 = time.time()
                gh_hist = gh_histogram(X, Y)
                t_hist = time.time() - t0

                t0 = time.time()
                gh_ks_val = gh_ks(X, Y)
                t_ks = time.time() - t0

                t0 = time.time()
                rmsd_val = rmsd_with_alignment(X, Y)
                t_rmsd = time.time() - t0

                t0 = time.time()
                tm_val = tm_score(X, Y)
                t_tm = time.time() - t0

                t0 = time.time()
                seq_rmsd_val = seq_aligned_rmsd(X, Y)
                t_seq = time.time() - t0

                results.append({
                    'protein1': name1, 'protein2': name2,
                    'len1': len(X), 'len2': len(Y),
                    'OGH': ogh_dist, 'OGH_coverage': info['coverage'],
                    'GH_histogram': gh_hist, 'GH_KS': gh_ks_val,
                    'RMSD': rmsd_val, 'TM_score': tm_val, 'Seq_RMSD': seq_rmsd_val,
                    'time_OGH': t_ogh, 'time_GH_hist': t_hist, 'time_GH_KS': t_ks,
                    'time_RMSD': t_rmsd, 'time_TM': t_tm, 'time_SeqRMSD': t_seq
                })

                if processed % 50 == 0:
                    print(f"  {processed}/{n_pairs} пар")

    return pd.DataFrame(results)


# ============================================================================
# 5. АНАЛИЗ РЕЗУЛЬТАТОВ
# ============================================================================

def analyze_results(df):
    print("\n" + "=" * 80)
    print("СТАТИСТИКА МЕТРИК")
    print("=" * 80)

    metrics = ['OGH', 'GH_histogram', 'GH_KS', 'RMSD', 'TM_score', 'Seq_RMSD']
    for m in metrics:
        vals = df[m].replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) > 0:
            print(f"{m:15} mean={vals.mean():7.3f} std={vals.std():7.3f}")

    print("\n" + "=" * 80)
    print("КОРРЕЛЯЦИИ С TM-SCORE")
    print("=" * 80)

    for m in metrics:
        if m != 'TM_score':
            valid = df[m].notna() & df['TM_score'].notna()
            if valid.sum() > 2:
                corr = df.loc[valid, m].corr(df.loc[valid, 'TM_score'])
                print(f"{m:15} vs TM_score: r={corr:7.3f}")

    print("\n" + "=" * 80)
    print("ТОП-10 ПОХОЖИХ ПАР ПО OGH")
    print("=" * 80)

    top = df.nsmallest(10, 'OGH')[['protein1', 'protein2', 'OGH', 'TM_score']]
    for _, row in top.iterrows():
        print(f"  {row['protein1'][:12]} ↔ {row['protein2'][:12]}: OGH={row['OGH']:.4f}, TM={row['TM_score']:.3f}")

    print("\n" + "=" * 80)
    print("ТОП-10 РАЗЛИЧНЫХ ПАР ПО OGH")
    print("=" * 80)

    bottom = df.nlargest(10, 'OGH')[['protein1', 'protein2', 'OGH', 'TM_score']]
    for _, row in bottom.iterrows():
        print(f"  {row['protein1'][:12]} ↔ {row['protein2'][:12]}: OGH={row['OGH']:.4f}, TM={row['TM_score']:.3f}")

    return df


# ============================================================================
# 6. ТЕСТ ЧУВСТВИТЕЛЬНОСТИ К ПОРЯДКУ
# ============================================================================

def test_order_sensitivity(structures, n_proteins=3):
    print("\n" + "=" * 80)
    print("ТЕСТ ЧУВСТВИТЕЛЬНОСТИ К ПОРЯДКУ")
    print("=" * 80)

    ogh = OrderedGromovHausdorff(order_weight=0.5)
    names = list(structures.keys())[:n_proteins]

    for name in names:
        X = structures[name]
        print(f"\n{name} ({len(X)} остатков):")

        for strength in [0, 0.3, 0.5, 0.7, 1.0]:
            if strength == 0:
                X_shuffled = X
            else:
                n_shuffle = int(len(X) * strength)
                idx = np.arange(len(X))
                if n_shuffle > 1:
                    shuffle_idx = np.random.permutation(n_shuffle)
                    idx[:n_shuffle] = idx[shuffle_idx]
                X_shuffled = X[idx]

            ogh_dist, _ = ogh.compute(X, X_shuffled)
            gh_dist = gh_histogram(X, X_shuffled)
            print(f"  {strength*100:3.0f}% перемешано: OGH={ogh_dist:.6f}, GH={gh_dist:.6f}")


# ============================================================================
# 7. MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("OGH ДЛЯ VAD: Упорядоченная метрика Громова-Хаусдорфа")
    print("=" * 80)

    structures = load_vad_structures('vad_500mb')

    if len(structures) < 2:
        print("Недостаточно структур")
        return

    test_order_sensitivity(structures, n_proteins=3)

    df = compare_metrics(structures, order_weight=0.5, max_pairs=200)

    analyze_results(df)

    df.to_csv('vad_ogh_results.csv', index=False)
    print("\nРезультаты сохранены в vad_ogh_results.csv")


if __name__ == "__main__":
    main()