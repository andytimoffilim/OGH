#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parameter_sensitivity_analysis.py — Sensitivity analysis of OGH parameters
Adapted for SCOP/ASTRAL dataset with direct PDB loading
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import svd as scipy_svd
from sklearn.metrics import roc_curve, auc
import warnings
import os
from pathlib import Path
from Bio.PDB import PDBParser
import requests
import time

warnings.filterwarnings('ignore')


# ============================================================================
# 1. OGH КЛАСС (из homology_validation_real_ogh_v2_eng.p.py)
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

        d_geom = cdist(D_X, D_Y, metric='euclidean')

        i_norm = np.arange(n).reshape(-1, 1) / n
        j_norm = np.arange(m).reshape(1, -1) / m
        w_ord = 20.0 * (1.0 - np.exp(-self.align_sigma * np.abs(i_norm - j_norm)))

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
# 2. ЗАГРУЗКА PDB
# ============================================================================

def download_pdb(pdb_id, output_dir='pdb_scop'):
    """Скачивает PDB файл с RCSB."""
    os.makedirs(output_dir, exist_ok=True)

    # Очищаем ID
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
    """Загружает Cα координаты из PDB файла."""
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
# 3. ЗАГРУЗКА ПАР И КООРДИНАТ
# ============================================================================

def load_pairs_with_coords():
    """Загружает пары и их координаты из основных результатов."""
    print("\n1. Загрузка данных...")

    # Загружаем основные результаты
    df_main = pd.read_csv('homology_real_ogh_results.csv')
    print(f"   Загружено {len(df_main)} пар")

    # Загружаем или создаём координаты
    coords_file = 'homology_coords.pkl'

    if os.path.exists(coords_file):
        print(f"   Загрузка координат из {coords_file}...")
        df_coords = pd.read_pickle(coords_file)
        print(f"   Загружено {len(df_coords)} записей с координатами")

        # Объединяем
        df_pairs = df_main.merge(df_coords, on=['protein1', 'protein2'], how='inner')
        print(f"   Объединено {len(df_pairs)} пар с координатами")

        return df_pairs

    # Если координат нет, загружаем PDB файлы
    print("   Координаты не найдены. Загружаем PDB файлы...")

    # Получаем уникальные ID
    all_ids = set(df_main['protein1'].tolist() + df_main['protein2'].tolist())
    print(f"   Уникальных белков: {len(all_ids)}")

    # Загружаем координаты для каждого белка
    coords_dict = {}
    for i, pdb_id in enumerate(all_ids):
        if (i + 1) % 50 == 0:
            print(f"   {i + 1}/{len(all_ids)}...")

        pdb_file = download_pdb(pdb_id, 'pdb_scop')
        if pdb_file:
            coords = load_pdb_coords(pdb_file)
            if coords is not None:
                coords_dict[pdb_id] = coords

    print(f"   Загружено координат для {len(coords_dict)} белков")

    # Создаём датафрейм с координатами
    df_pairs = df_main.copy()
    df_pairs['coords1'] = df_pairs['protein1'].apply(lambda x: coords_dict.get(x))
    df_pairs['coords2'] = df_pairs['protein2'].apply(lambda x: coords_dict.get(x))

    # Удаляем пары без координат
    df_pairs = df_pairs[df_pairs['coords1'].notna() & df_pairs['coords2'].notna()]
    print(f"   Пар с координатами: {len(df_pairs)}")

    # Сохраняем координаты для будущего использования
    df_coords = df_pairs[['protein1', 'protein2', 'coords1', 'coords2']].copy()
    df_coords.to_pickle(coords_file)
    print(f"   Координаты сохранены в {coords_file}")

    return df_pairs


# ============================================================================
# 4. ВЫЧИСЛЕНИЕ OGH ДЛЯ РАЗНЫХ ПАРАМЕТРОВ
# ============================================================================

def compute_ogh_for_params(df_pairs, alpha, window_size, gap_penalty):
    """Вычисляет OGH для заданных параметров и возвращает AUC."""
    print(f"  α={alpha:.2f}, w={window_size}, γ={gap_penalty:.1f}...", end=" ", flush=True)

    ogh = OrderedGromovHausdorff(
        alpha=alpha,
        align_sigma=3.0,
        align_order_weight=1.0,
        window_size=window_size,
        gap_penalty=gap_penalty,
        use_normalization=True
    )

    ogh_values = []

    for idx, row in df_pairs.iterrows():
        X = row['coords1']
        Y = row['coords2']

        ogh_dist, _ = ogh.compute(X, Y)
        ogh_values.append(ogh_dist)

    # Вычисляем AUC
    y_true = df_pairs['is_homolog'].values
    try:
        fpr, tpr, _ = roc_curve(y_true, -np.array(ogh_values))
        auc_value = auc(fpr, tpr)
    except:
        auc_value = np.nan

    print(f"AUC={auc_value:.3f}")
    return auc_value


# ============================================================================
# 5. ОСНОВНАЯ ФУНКЦИЯ АНАЛИЗА
# ============================================================================

def run_sensitivity_analysis(df_pairs):
    """Запускает полный анализ чувствительности."""
    print("\n" + "=" * 80)
    print("🔬 АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ ПАРАМЕТРОВ OGH")
    print("=" * 80)

    window_sizes = [10, 15, 20, 25]
    gap_penalties = [1.0, 2.0, 3.0]
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    results = []
    total_combos = len(window_sizes) * len(gap_penalties) + len(alphas)

    print(f"\n📊 Тестирование {total_combos} комбинаций параметров...")
    print(f"   Всего пар: {len(df_pairs)}")
    print()

    # 1. Тестируем window size и gap penalty (фиксированный alpha=0.5)
    print("1. Влияние window size и gap penalty (α=0.5):")
    for w in window_sizes:
        for g in gap_penalties:
            auc_val = compute_ogh_for_params(df_pairs, 0.5, w, g)
            results.append({
                'window_size': w,
                'gap_penalty': g,
                'alpha': 0.5,
                'AUC': auc_val
            })

    # 2. Тестируем alpha (фиксированные w=15, γ=2.0)
    print("\n2. Влияние alpha (w=15, γ=2.0):")
    for alpha in alphas:
        auc_val = compute_ogh_for_params(df_pairs, alpha, 15, 2.0)
        results.append({
            'window_size': 15,
            'gap_penalty': 2.0,
            'alpha': alpha,
            'AUC': auc_val
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv('sensitivity_analysis.csv', index=False)
    print(f"\n✅ Сохранено: sensitivity_analysis.csv")

    return df_results


# ============================================================================
# 6. ВИЗУАЛИЗАЦИЯ (ИСПРАВЛЕННАЯ)
# ============================================================================

def plot_sensitivity_analysis(df_results):
    """Создает визуализацию результатов с обработкой дубликатов."""
    print("\n📊 Создание графиков...")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Window size vs Gap penalty (α=0.5)
    df_wg = df_results[df_results['alpha'] == 0.5].copy()

    # Удаляем дубликаты (усредняем если есть)
    df_wg = df_wg.groupby(['window_size', 'gap_penalty'], as_index=False)['AUC'].mean()

    # Создаём сводную таблицу
    pivot = df_wg.pivot(index='window_size', columns='gap_penalty', values='AUC')

    im = axes[0].imshow(pivot.values, cmap='viridis', aspect='auto')
    axes[0].set_xticks(range(len(pivot.columns)))
    axes[0].set_xticklabels([f'{x:.1f}' for x in pivot.columns])
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_yticklabels(pivot.index)
    axes[0].set_xlabel('Gap penalty (γ)')
    axes[0].set_ylabel('Window size (w)')
    axes[0].set_title('AUC: w vs γ (α=0.5)')

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "w" if val < 0.66 else "black"
            axes[0].text(j, i, f"{val:.3f}", ha="center", va="center", color=color)
    plt.colorbar(im, ax=axes[0])

    # 2. Alpha sensitivity (w=15, γ=2.0)
    df_alpha = df_results[(df_results['window_size'] == 15) & (df_results['gap_penalty'] == 2.0)].copy()
    df_alpha = df_alpha.groupby('alpha', as_index=False)['AUC'].mean()
    df_alpha = df_alpha.sort_values('alpha')

    axes[1].plot(df_alpha['alpha'], df_alpha['AUC'], 'o-', linewidth=2, markersize=8, color='red')

    # Добавляем baseline для α=0.5
    auc_05 = df_alpha[df_alpha['alpha'] == 0.5]['AUC'].values[0] if 0.5 in df_alpha['alpha'].values else None
    if auc_05 is not None:
        axes[1].axhline(y=auc_05, color='gray', linestyle='--', alpha=0.5, label=f'α=0.5 (AUC={auc_05:.3f})')

    axes[1].set_xlabel('α (order weight)')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Sensitivity to α (w=15, γ=2.0)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Добавляем значения на график
    for _, row in df_alpha.iterrows():
        axes[1].text(row['alpha'] + 0.02, row['AUC'] + 0.001, f"{row['AUC']:.3f}", fontsize=9)

    plt.tight_layout()
    plt.savefig('sensitivity_analysis_plot.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ sensitivity_analysis_plot.png")
    plt.show()


# ============================================================================
# 7. MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("🔬 АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ ПАРАМЕТРОВ OGH")
    print("   SCOP/ASTRAL датасет (1,400 пар)")
    print("=" * 80)

    # Загружаем пары с координатами
    df_pairs = load_pairs_with_coords()

    if len(df_pairs) == 0:
        print("\n❌ Не удалось загрузить пары с координатами.")
        print("   Проверьте наличие PDB файлов или интернет-соединения.")
        return

    # Запускаем анализ
    df_sens = run_sensitivity_analysis(df_pairs)

    # Визуализация
    plot_sensitivity_analysis(df_sens)

    # Вывод статистики
    print("\n" + "=" * 80)
    print("📊 СТАТИСТИКА АНАЛИЗА ЧУВСТВИТЕЛЬНОСТИ")
    print("=" * 80)

    df_wg = df_sens[df_sens['alpha'] == 0.5]
    print(f"\nСтабильность при α=0.5 (w={df_wg['window_size'].unique()}, γ={df_wg['gap_penalty'].unique()}):")
    print(f"  AUC min: {df_wg['AUC'].min():.3f}")
    print(f"  AUC max: {df_wg['AUC'].max():.3f}")
    print(f"  AUC range: {df_wg['AUC'].max() - df_wg['AUC'].min():.3f}")
    print(f"  AUC std: {df_wg['AUC'].std():.3f}")

    df_alpha = df_sens[(df_sens['window_size'] == 15) & (df_sens['gap_penalty'] == 2.0)]
    print(f"\nВлияние α (w=15, γ=2.0):")
    best_alpha = df_alpha.loc[df_alpha['AUC'].idxmax()]
    print(f"  Лучший α: {best_alpha['alpha']:.2f} (AUC={best_alpha['AUC']:.3f})")
    auc_05 = df_alpha[df_alpha['alpha'] == 0.5]['AUC'].values[0]
    print(f"  AUC при α=0.5: {auc_05:.3f}")
    print(f"  Разница: {best_alpha['AUC'] - auc_05:.3f}")

    print("\n✅ Анализ чувствительности завершен!")
    print(f"📁 Результаты сохранены в: sensitivity_analysis.csv")


if __name__ == "__main__":
    main()