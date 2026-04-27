"""
ogh_VAD_advanced.py - Extended OGH Analysis: α-sensitivity and Method Comparison

Authors: Timofeev A.V., Anufriev A.S.
Organization: AI Center for SCO+ Countries, St. Petersburg, Russia
Date: 2024
License: MIT

Description:
    Extended analysis of the OGH metric including:
    - Sensitivity analysis to order weight α (from 0 to 1)
    - Comparison with alternative ordered methods:
        * CE-style alignment (greedy with linear penalty)
        * MATRAS-style (geometry + sequence combination)
        * SALIGN-style (controllable order strictness)
    - Statistical significance testing (bootstrap, p-values)

Dependencies:
    numpy, scipy, pandas, matplotlib (optional for plotting)

Output:
    - alpha_sensitivity.csv - correlation vs α
    - ordered_methods_comparison.csv - comparison with other ordered methods
    - statistical_significance.csv - p-values and confidence intervals
    - alpha_sensitivity.png - plot (if matplotlib is available)

Usage:
    >>> python ogh_VAD_advanced.py

Note:
    Requires VAD PDB files in the vad_500mb/ directory.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance, ks_2samp, pearsonr, spearmanr
from scipy.linalg import svd
from Bio.PDB import PDBParser
from Bio.Align import PairwiseAligner
import glob
import os
import time
import warnings
from itertools import combinations

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
# 2.1. ДОПОЛНИТЕЛЬНЫЕ УПОРЯДОЧЕННЫЕ МЕТОДЫ
# ============================================================================

def ce_style_alignment(X, Y, order_penalty_weight=1.0):
    """
    CE-подобное выравнивание с учётом порядка (упрощённая версия)
    Использует жадное выравнивание с штрафом за нарушение порядка
    """
    n, m = len(X), len(Y)

    # Нормализация
    X_norm = (X - np.mean(X, axis=0)) / (np.std(X) + 1e-8)
    Y_norm = (Y - np.mean(Y, axis=0)) / (np.std(Y) + 1e-8)

    # Матрица расстояний
    dist_mat = cdist(X_norm, Y_norm)

    # Штраф за порядок (линейный, как в CE)
    order_penalty = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            pos_diff = abs(i / n - j / m)
            order_penalty[i, j] = order_penalty_weight * 10 * pos_diff

    total_cost = dist_mat + order_penalty

    # Жадное выравнивание
    alignment = []
    i, j = 0, 0

    while i < n and j < m:
        # Ищем лучшее соответствие в окне
        window = min(20, n - i, m - j)
        best_idx = np.unravel_index(
            np.argmin(total_cost[i:i + window, j:j + window]),
            (window, window)
        )
        best_i, best_j = i + best_idx[0], j + best_idx[1]
        alignment.append((best_i, best_j))
        i = best_i + 1
        j = best_j + 1

    if len(alignment) == 0:
        return np.inf

    # Вычисляем среднее расстояние
    mean_dist = np.mean([dist_mat[i, j] for i, j in alignment])
    return mean_dist / np.sqrt(min(n, m))


def matras_style_alignment(X, Y):
    """
    MATRAS-подобное выравнивание (учитывает и структуру, и последовательность)
    Упрощённая версия: комбинация геометрического и sequence-based выравнивания
    """
    n, m = len(X), len(Y)

    # Нормализация координат
    X_norm = (X - np.mean(X, axis=0)) / (np.std(X) + 1e-8)
    Y_norm = (Y - np.mean(Y, axis=0)) / (np.std(Y) + 1e-8)

    # Геометрическая матрица
    geom_dist = cdist(X_norm, Y_norm)

    # Симуляция sequence-based выравнивания (на основе позиций)
    seq_dist = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            seq_dist[i, j] = abs(i / n - j / m)

    # Комбинированная стоимость
    total_cost = 0.5 * geom_dist + 0.5 * seq_dist

    # Динамическое программирование для монотонного выравнивания
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0

    for i in range(n + 1):
        for j in range(m + 1):
            if i > 0 and j > 0:
                dp[i, j] = min(dp[i, j], dp[i - 1, j - 1] + total_cost[i - 1, j - 1])
            if i > 0:
                dp[i, j] = min(dp[i, j], dp[i - 1, j] + 2.0)  # gap penalty
            if j > 0:
                dp[i, j] = min(dp[i, j], dp[i, j - 1] + 2.0)

    # Нормализация
    score = dp[n, m] / min(n, m)
    return score


def salign_style_alignment(X, Y, order_strictness=0.5):
    """
    SALIGN-подобное выравнивание с контролируемой строгостью порядка
    """
    n, m = len(X), len(Y)

    X_norm = (X - np.mean(X, axis=0)) / (np.std(X) + 1e-8)
    Y_norm = (Y - np.mean(Y, axis=0)) / (np.std(Y) + 1e-8)

    geom_dist = cdist(X_norm, Y_norm)

    # Порядковый штраф с контролируемой строгостью
    order_penalty = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            pos_diff = abs(i / n - j / m)
            # Строгость порядка: чем выше strictness, тем сильнее штраф
            order_penalty[i, j] = order_strictness * 15 * pos_diff ** 2

    total_cost = geom_dist + order_penalty

    # DP с ограничением на отклонение от диагонали
    max_deviation = int(max(n, m) * (1 - order_strictness))

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
                    # Дополнительный штраф за большое отклонение
                    deviation_penalty = 0
                    if abs(di - dj) > max_deviation:
                        deviation_penalty = 100
                    cost = total_cost[i + di, j + dj] + deviation_penalty
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

    if len(alignment) == 0:
        return np.inf

    mean_dist = np.mean([geom_dist[i, j] for i, j in alignment])
    return mean_dist / np.sqrt(min(n, m))


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
# 4. АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ К α
# ============================================================================

def alpha_sensitivity_analysis(structures, alpha_values=None, max_pairs=100):
    """
    Анализ чувствительности OGH к весу порядка α
    Строит график корреляции OGH с TM-score в зависимости от α
    """
    if alpha_values is None:
        alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("\n" + "=" * 80)
    print("АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ К ВЕСУ ПОРЯДКА α")
    print("=" * 80)

    # Берём подмножество пар для анализа
    names = list(structures.keys())
    pairs = list(combinations(names, 2))[:max_pairs]

    # Предварительно вычисляем TM-score для всех пар
    print("\nВычисление TM-score...")
    tm_scores = {}
    for name1, name2 in pairs:
        tm = tm_score(structures[name1], structures[name2])
        tm_scores[(name1, name2)] = tm

    results = []

    for alpha in alpha_values:
        print(f"\nТестирование α = {alpha:.1f}...")
        ogh = OrderedGromovHausdorff(order_weight=alpha)

        ogh_values = []
        tm_list = []

        for name1, name2 in pairs:
            ogh_dist, _ = ogh.compute(structures[name1], structures[name2])
            ogh_values.append(ogh_dist)
            tm_list.append(tm_scores[(name1, name2)])

        # Корреляции
        pearson_r, pearson_p = pearsonr(ogh_values, tm_list)
        spearman_r, spearman_p = spearmanr(ogh_values, tm_list)

        results.append({
            'alpha': alpha,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'significant_05': pearson_p < 0.05,
            'significant_01': pearson_p < 0.01
        })

        sig_str = "✅" if pearson_p < 0.05 else "❌"
        print(f"  Pearson r = {pearson_r:.4f} (p={pearson_p:.2e}) {sig_str}")
        print(f"  Spearman ρ = {spearman_r:.4f} (p={spearman_p:.2e})")

    # Находим оптимальный α
    best_alpha = max(results, key=lambda x: abs(x['pearson_r']))
    print("\n" + "=" * 80)
    print("ОПТИМАЛЬНЫЙ ВЕС ПОРЯДКА")
    print("=" * 80)
    print(f"  α = {best_alpha['alpha']:.1f} даёт максимальную корреляцию |r| = {abs(best_alpha['pearson_r']):.4f}")

    return pd.DataFrame(results)


# ============================================================================
# 5. СРАВНЕНИЕ С ДРУГИМИ УПОРЯДОЧЕННЫМИ МЕТОДАМИ
# ============================================================================

def compare_ordered_methods(structures, max_pairs=100):
    """
    Сравнение OGH с другими упорядоченными методами (CE-style, MATRAS-style, SALIGN-style)
    """
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ С ДРУГИМИ УПОРЯДОЧЕННЫМИ МЕТОДАМИ")
    print("=" * 80)

    names = list(structures.keys())
    pairs = list(combinations(names, 2))[:max_pairs]

    results = []

    print("\nВычисление метрик...")

    for idx, (name1, name2) in enumerate(pairs):
        X = structures[name1]
        Y = structures[name2]

        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(pairs)} пар...")

        # OGH
        ogh = OrderedGromovHausdorff(order_weight=0.5)
        ogh_dist, _ = ogh.compute(X, Y)

        # CE-style
        ce_dist = ce_style_alignment(X, Y)

        # MATRAS-style
        matras_dist = matras_style_alignment(X, Y)

        # SALIGN-style (разная строгость)
        salign_loose = salign_style_alignment(X, Y, order_strictness=0.3)
        salign_strict = salign_style_alignment(X, Y, order_strictness=0.7)

        # TM-score как референс
        tm = tm_score(X, Y)

        results.append({
            'protein1': name1,
            'protein2': name2,
            'OGH': ogh_dist,
            'CE_style': ce_dist,
            'MATRAS_style': matras_dist,
            'SALIGN_loose': salign_loose,
            'SALIGN_strict': salign_strict,
            'TM_score': tm
        })

    df = pd.DataFrame(results)

    # Корреляции с TM-score
    print("\n" + "=" * 80)
    print("КОРРЕЛЯЦИИ С TM-SCORE (p-values в скобках)")
    print("=" * 80)

    methods = ['OGH', 'CE_style', 'MATRAS_style', 'SALIGN_loose', 'SALIGN_strict']

    for method in methods:
        valid = df[method].notna() & df['TM_score'].notna()
        if valid.sum() > 2:
            pearson_r, pearson_p = pearsonr(df.loc[valid, method], df.loc[valid, 'TM_score'])
            spearman_r, spearman_p = spearmanr(df.loc[valid, method], df.loc[valid, 'TM_score'])

            sig_pearson = "***" if pearson_p < 0.001 else (
                "**" if pearson_p < 0.01 else ("*" if pearson_p < 0.05 else "ns"))

            print(f"\n{method}:")
            print(f"  Pearson r = {pearson_r:.4f} (p={pearson_p:.2e}) {sig_pearson}")
            print(f"  Spearman ρ = {spearman_r:.4f} (p={spearman_p:.2e})")

    # Tест на значимость различий между OGH и другими методами
    print("\n" + "=" * 80)
    print("СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ РАЗЛИЧИЙ МЕЖДУ OGH И ДРУГИМИ МЕТОДАМИ")
    print("=" * 80)

    from scipy.stats import wilcoxon

    for method in methods:
        if method != 'OGH':
            diff = abs(df['OGH'].values - df[method].values)
            valid = ~np.isnan(diff)
            if valid.sum() > 2:
                # Тест Уилкоксона для парных сравнений
                try:
                    stat, p_val = wilcoxon(df['OGH'].values[valid], df[method].values[valid])
                    print(f"\nOGH vs {method}:")
                    print(f"  Wilcoxon test: statistic={stat:.2f}, p={p_val:.2e}")
                    if p_val < 0.05:
                        print(f"  ✅ Различия статистически значимы (p<0.05)")
                    else:
                        print(f"  ❌ Различия не значимы")
                except:
                    print(f"\nOGH vs {method}: недостаточно данных")

    # Сохраняем результаты
    df.to_csv('ordered_methods_comparison.csv', index=False)
    print("\nРезультаты сохранены в ordered_methods_comparison.csv")

    return df


# ============================================================================
# 6. СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ КОРРЕЛЯЦИЙ
# ============================================================================

def statistical_significance_analysis(df):
    """
    Детальный анализ статистической значимости корреляций
    """
    print("\n" + "=" * 80)
    print("СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ КОРРЕЛЯЦИЙ")
    print("=" * 80)

    metrics = ['OGH', 'GH_histogram', 'GH_KS', 'RMSD', 'Seq_RMSD']

    results = []

    for metric in metrics:
        if metric in df.columns:
            valid = df[metric].notna() & df['TM_score'].notna()
            valid = valid & (df[metric] < 1e6)

            if valid.sum() > 2:
                pearson_r, pearson_p = pearsonr(df.loc[valid, metric], df.loc[valid, 'TM_score'])
                spearman_r, spearman_p = spearmanr(df.loc[valid, metric], df.loc[valid, 'TM_score'])

                # Доверительные интервалы (бутстрап)
                n_bootstrap = 1000
                bootstrap_corrs = []

                np.random.seed(42)
                for _ in range(n_bootstrap):
                    indices = np.random.choice(valid.sum(), valid.sum(), replace=True)
                    boot_x = df.loc[valid, metric].values[indices]
                    boot_y = df.loc[valid, 'TM_score'].values[indices]
                    if len(boot_x) > 2 and np.std(boot_x) > 0 and np.std(boot_y) > 0:
                        boot_r, _ = pearsonr(boot_x, boot_y)
                        bootstrap_corrs.append(boot_r)

                ci_lower = np.percentile(bootstrap_corrs, 2.5)
                ci_upper = np.percentile(bootstrap_corrs, 97.5)

                results.append({
                    'metric': metric,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'ci_95_lower': ci_lower,
                    'ci_95_upper': ci_upper,
                    'significant': pearson_p < 0.05,
                    'n_pairs': valid.sum()
                })

    # Вывод результатов
    print("\n{:<15} {:>10} {:>12} {:>12} {:>12} {:>12} {:>10}".format(
        'Метрика', 'r (Pearson)', 'p-value', 'ρ (Spearman)', 'p-value', '95% CI', 'N'))
    print("-" * 85)

    for r in results:
        sig_str = "***" if r['pearson_p'] < 0.001 else (
            "**" if r['pearson_p'] < 0.01 else ("*" if r['pearson_p'] < 0.05 else ""))
        print("{:<15} {:>10.4f}{:<2} {:>12.2e} {:>12.4f} {:>12.2e} [{:>6.3f}, {:>6.3f}] {:>10d}".format(
            r['metric'], r['pearson_r'], sig_str, r['pearson_p'],
            r['spearman_r'], r['spearman_p'],
            r['ci_95_lower'], r['ci_95_upper'], r['n_pairs']))

    print("\n  *** p<0.001, ** p<0.01, * p<0.05")

    return pd.DataFrame(results)


# ============================================================================
# 7. ВИЗУАЛИЗАЦИЯ
# ============================================================================

def plot_alpha_sensitivity(alpha_df, save=True):
    """Строит график зависимости корреляции от α"""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # График корреляции Пирсона
        ax1 = axes[0]
        ax1.plot(alpha_df['alpha'], alpha_df['pearson_r'], 'b-o', linewidth=2, markersize=8, label='Pearson r')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Вес порядка α')
        ax1.set_ylabel('Корреляция с TM-score (r)')
        ax1.set_title('Зависимость корреляции Пирсона от α')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # График p-value
        ax2 = axes[1]
        ax2.semilogy(alpha_df['alpha'], alpha_df['pearson_p'], 'r-s', linewidth=2, markersize=8)
        ax2.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='p=0.05')
        ax2.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='p=0.01')
        ax2.set_xlabel('Вес порядка α')
        ax2.set_ylabel('p-value')
        ax2.set_title('Статистическая значимость корреляции')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        if save:
            plt.savefig('alpha_sensitivity.png', dpi=300, bbox_inches='tight')
            print("\nГрафик сохранён: alpha_sensitivity.png")

        plt.show()

    except ImportError:
        print("\n Matplotlib не установлен, пропускаем визуализацию")


def plot_methods_comparison(methods_df, save=True):
    """Строит график сравнения методов"""
    try:
        import matplotlib.pyplot as plt

        methods = ['OGH', 'CE_style', 'MATRAS_style', 'SALIGN_loose', 'SALIGN_strict']
        colors = ['blue', 'red', 'green', 'orange', 'purple']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Box plot распределений
        ax1 = axes[0]
        data_to_plot = [methods_df[m].dropna().values for m in methods if m in methods_df.columns]
        bp = ax1.boxplot(data_to_plot, labels=[m for m in methods if m in methods_df.columns], patch_artist=True)

        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax1.set_ylabel('Расстояние')
        ax1.set_title('Сравнение распределений метрик')
        ax1.grid(True, alpha=0.3)

        # Корреляции с TM-score
        ax2 = axes[1]
        correlations = []
        method_labels = []

        for method in methods:
            if method in methods_df.columns:
                valid = methods_df[method].notna() & methods_df['TM_score'].notna()
                if valid.sum() > 2:
                    r, _ = pearsonr(methods_df.loc[valid, method], methods_df.loc[valid, 'TM_score'])
                    correlations.append(abs(r))
                    method_labels.append(method)

        bars = ax2.bar(method_labels, correlations, color=colors[:len(correlations)], alpha=0.7)
        ax2.set_ylabel('|Корреляция с TM-score|')
        ax2.set_title('Сравнение корреляций методов')
        ax2.set_ylim(0, 1)

        for bar, r in zip(bars, correlations):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{r:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save:
            plt.savefig('methods_comparison.png', dpi=300, bbox_inches='tight')
            print("\nГрафик сохранён: methods_comparison.png")

        plt.show()

    except ImportError:
        print("\n Matplotlib не установлен, пропускаем визуализацию")


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("OGH VAD: РАСШИРЕННЫЙ АНАЛИЗ")
    print("Сравнение с другими методами, анализ α, p-values")
    print("=" * 80)

    # Загрузка структур
    structures = load_vad_structures('vad_500mb')

    if len(structures) < 2:
        print("Недостаточно структур")
        return

    # 1. Анализ чувствительности к α
    alpha_df = alpha_sensitivity_analysis(structures, max_pairs=100)
    plot_alpha_sensitivity(alpha_df)

    # 2. Сравнение с другими упорядоченными методами
    methods_df = compare_ordered_methods(structures, max_pairs=100)
    plot_methods_comparison(methods_df)

    # 3. Базовое сравнение метрик (как в оригинальном коде)
    print("\n" + "=" * 80)
    print("БАЗОВОЕ СРАВНЕНИЕ МЕТРИК")
    print("=" * 80)

    ogh = OrderedGromovHausdorff(order_weight=0.5)
    names = list(structures.keys())
    pairs = list(combinations(names, 2))[:100]

    basic_results = []
    for name1, name2 in pairs:
        X = structures[name1]
        Y = structures[name2]

        ogh_dist, _ = ogh.compute(X, Y)
        gh_h = gh_histogram(X, Y)
        gh_k = gh_ks(X, Y)
        rmsd_val = rmsd_with_alignment(X, Y)
        tm_val = tm_score(X, Y)
        seq_rmsd_val = seq_aligned_rmsd(X, Y)

        basic_results.append({
            'protein1': name1, 'protein2': name2,
            'OGH': ogh_dist, 'GH_histogram': gh_h, 'GH_KS': gh_k,
            'RMSD': rmsd_val, 'TM_score': tm_val, 'Seq_RMSD': seq_rmsd_val
        })

    basic_df = pd.DataFrame(basic_results)

    # 4. Статистическая значимость
    stats_df = statistical_significance_analysis(basic_df)

    # 5. Сохранение результатов
    alpha_df.to_csv('alpha_sensitivity.csv', index=False)
    basic_df.to_csv('vad_metrics_advanced.csv', index=False)
    stats_df.to_csv('statistical_significance.csv', index=False)

    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ СОХРАНЕНЫ")
    print("=" * 80)
    print("  📁 alpha_sensitivity.csv         - анализ чувствительности к α")
    print("  📁 ordered_methods_comparison.csv - сравнение с другими методами")
    print("  📁 vad_metrics_advanced.csv      - все метрики")
    print("  📁 statistical_significance.csv  - p-values и доверительные интервалы")

    # Итоговые рекомендации
    print("\n" + "=" * 80)
    print("ИТОГОВЫЕ РЕКОМЕНДАЦИИ")
    print("=" * 80)

    # Находим лучший α
    best_alpha_row = alpha_df.loc[alpha_df['pearson_r'].abs().idxmax()]
    print(f"""
    📊 ОПТИМАЛЬНЫЙ ВЕС ПОРЯДКА:

    α = {best_alpha_row['alpha']:.1f} даёт:
    • Корреляция с TM-score: r = {best_alpha_row['pearson_r']:.4f}
    • p-value = {best_alpha_row['pearson_p']:.2e}
    • Статистически значимо: {'✅ ДА' if best_alpha_row['pearson_p'] < 0.05 else '❌ НЕТ'}

    📊 СРАВНЕНИЕ МЕТОДОВ:

    OGH показал {'наивысшую' if methods_df['OGH'].corr(methods_df['TM_score']) > max([methods_df[m].corr(methods_df['TM_score']) for m in methods_df.columns if m != 'TM_score' and m != 'protein1' and m != 'protein2']) else 'хорошую'} корреляцию с TM-score.

    💡 РЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ:

    1. Для задач, где важен порядок остатков, используйте α = 0.5
    2. Для поиска remote homology комбинируйте OGH с TM-score
    3. Для быстрого скрининга используйте GH Histogram
    """)


if __name__ == "__main__":
    main()