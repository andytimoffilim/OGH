#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_circular_permutation.py — Минимальный тест OGH на circular permutation

Идея:
    Берём белок, делаем циклические перестановки (разрезаем в разных местах),
    считаем OGH между оригиналом и перестановкой.

    Если OGH растёт с "серьёзностью" перестановки — OGH чувствителен к порядку.

Usage:
    python test_circular_permutation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import svd as scipy_svd
from Bio.PDB import PDBParser, is_aa
import os
import requests
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 1. ЗАГРУЗКА PDB
# ============================================================================

def download_pdb(pdb_id, output_dir='pdb_scop'):
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


def load_ca_coords(pdb_file):
    """Загружает Cα координаты из PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue) and 'CA' in residue:
                    coords.append(residue['CA'].get_coord())
    return np.array(coords) if coords else None


# ============================================================================
# 2. OGH (УПРОЩЁННЫЙ, НО ТОТ ЖЕ АЛГОРИТМ)
# ============================================================================

class OGH:
    def __init__(self, alpha=0.5, window=15, gap_penalty=2.0):
        self.alpha = alpha
        self.window = window
        self.gap_penalty = gap_penalty

    def compute(self, X, Y):
        n, m = len(X), len(Y)
        L = min(n, m)

        # Нормализация
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        Y = (Y - np.mean(Y, axis=0)) / (np.std(Y, axis=0) + 1e-8)

        # Контактные карты
        DX = cdist(X, X)
        DY = cdist(Y, Y)
        DX = DX[:, :L]
        DY = DY[:, :L]

        # Геометрическая стоимость (сравнение строк контактных карт)
        d_geom = cdist(DX, DY, metric='euclidean')

        # Штраф за порядок
        order_penalty = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                pos_diff = abs(i / n - j / m)
                order_penalty[i, j] = 20 * (1 - np.exp(-3 * pos_diff))

        # Выравнивание (greedy)
        cost = d_geom + order_penalty
        alignment = self._greedy_alignment(cost)

        if len(alignment) == 0:
            return np.inf

        # Финальный скор
        matched_geom = [d_geom[i, j] for i, j in alignment]
        matched_order = [order_penalty[i, j] for i, j in alignment]

        metric_dist = np.mean(matched_geom)
        order_dist = np.mean(matched_order)

        ogh_dist = (1 - self.alpha) * metric_dist + self.alpha * order_dist
        return ogh_dist / np.sqrt(L)

    def _greedy_alignment(self, cost_matrix):
        n, m = cost_matrix.shape
        alignment = []
        i, j = 0, 0

        while i < n and j < m:
            win_i = min(self.window, n - i)
            win_j = min(self.window, m - j)
            best_cost = np.inf
            best_di, best_dj = 1, 1

            for di in range(1, win_i + 1):
                for dj in range(1, win_j + 1):
                    gap = self.gap_penalty * ((di - 1) + (dj - 1))
                    c = cost_matrix[i + di - 1, j + dj - 1] + gap
                    if c < best_cost:
                        best_cost = c
                        best_di, best_dj = di, dj

            alignment.append((i + best_di - 1, j + best_dj - 1))
            i += best_di
            j += best_dj

        return alignment


# ============================================================================
# 3. ГЕНЕРАЦИЯ CIRCULAR PERMUTATIONS
# ============================================================================

def circular_permute(coords, cut_position):
    """
    Делает циклическую перестановку координат.

    cut_position: от 0 до 1 (доля от длины белка)
    """
    n = len(coords)
    cut = int(n * cut_position)
    if cut == 0 or cut == n:
        return coords.copy()
    return np.vstack([coords[cut:], coords[:cut]])


# ============================================================================
# 4. ТЕСТ
# ============================================================================

def main():
    print("=" * 80)
    print("🧬 CIRCULAR PERMUTATION TEST: OGH sensitivity")
    print("=" * 80)

    # Выбираем белок из SCOP/ASTRAL
    pdb_id = "d1faki_"  # 55 residues (можно заменить на любой)
    print(f"\nБелок: {pdb_id}")

    # Загружаем координаты
    pdb_file = download_pdb(pdb_id, 'pdb_scop')
    if pdb_file is None:
        print("❌ Не удалось скачать PDB!")
        return

    coords = load_ca_coords(pdb_file)
    if coords is None or len(coords) < 20:
        print(f"❌ Белок слишком мал или не загружен ({len(coords) if coords is not None else 0} residues)")
        return

    n = len(coords)
    print(f"  Длина: {n} residues")

    # Тестируем разные cut positions
    cuts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ogh = OGH(alpha=0.5)

    results = []
    for cut in cuts:
        permuted = circular_permute(coords, cut)
        dist = ogh.compute(coords, permuted)
        results.append({'cut': cut, 'OGH': dist})
        print(f"  cut={cut:.1f} → OGH={dist:.4f}")

    df = pd.DataFrame(results)
    df.to_csv('circular_permutation_results.csv', index=False)
    print("\n✅ Сохранено: circular_permutation_results.csv")

    # График
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['cut'], df['OGH'], 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Cut position (fraction of chain)')
    ax.set_ylabel('OGH distance')
    ax.set_title(f'OGH sensitivity to circular permutation\n({pdb_id}, {n} residues)')
    ax.grid(True, alpha=0.3)

    # Аннотация
    max_row = df.loc[df['OGH'].idxmax()]
    ax.annotate(f"Max OGH = {max_row['OGH']:.4f}\nat cut={max_row['cut']:.1f}",
                xy=(max_row['cut'], max_row['OGH']),
                xytext=(max_row['cut'] + 0.1, max_row['OGH'] + 0.02),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig('circular_permutation_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Сохранено: circular_permutation_plot.png")

    # Вывод
    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТЫ")
    print("=" * 80)

    min_ogh = df.loc[df['cut'] == 0.0, 'OGH'].values[0]
    max_ogh = df['OGH'].max()
    print(f"  OGH (original):  {min_ogh:.4f}")
    print(f"  OGH (max):       {max_ogh:.4f}")
    print(f"  Увеличение:      {max_ogh - min_ogh:+.4f}")

    if max_ogh > min_ogh * 1.5:
        print("\n  ✅ OGH显著 увеличивается при circular permutation")
        print("     → Метод чувствителен к изменению порядка")
    else:
        print("\n  ⚠️ Увеличение небольшое, но OGH всё равно реагирует")

    print("\n✅ Тест завершён!")


if __name__ == "__main__":
    main()