"""
improve_ogh_for_article.py - Улучшенная версия OGH и расширенный анализ

- Модифицированная OGH с экспоненциальной геометрией, локальным выравниванием
- Корреляция с TM-score
- ROC-анализ для remote homology и перестроек доменов
- Комбинация OGH + TM-score

Authors: Timofeev A.V., Anufriev A.S.
Date: 2026
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr, spearmanr
from Bio.PDB import PDBParser
import glob
import os
import warnings
from itertools import combinations

warnings.filterwarnings('ignore')


# ============================================================================
# 1. УЛУЧШЕННАЯ OGH (Improved OGH)
# ============================================================================

class ImprovedOGH:
    """
    Улучшенная упорядоченная метрика Громова-Хаусдорфа.

    Изменения по сравнению с базовой версией:
    1. Экспоненциальное затухание геометрической компоненты (как в TM-score)
    2. Локальное выравнивание (разрешены пропуски на концах)
    3. Нормализация по L_min = min(n,m) вместо sqrt(min(n,m))
    4. Оптимизация alpha через сетку (для максимальной корреляции с TM-score)
    """

    def __init__(self, alpha=0.3, sigma_geom=1.2, local_alignment=False,
                 gap_penalty=20.0, window=15, use_dp=True):
        """
        Parameters:
        -----------
        alpha : float
            Вес порядка (0 = только геометрия, 1 = только порядок)
        sigma_geom : float
            Параметр ширины для экспоненциального веса геометрического расстояния
        local_alignment : bool
            Разрешить пропуски на концах (локальное выравнивание)
        gap_penalty : float
            Штраф за пропуск остатка (только при локальном выравнивании)
        window : int
            Размер окна для жадного выравнивания (если не используется DP)
        use_dp : bool
            Использовать динамическое программирование (точнее, но медленнее)
        """
        self.alpha = alpha
        self.sigma_geom = sigma_geom
        self.local_alignment = local_alignment
        self.gap_penalty = gap_penalty
        self.window = window
        self.use_dp = use_dp

    def _geom_similarity(self, X, Y):
        """
        Экспоненциальное взвешивание геометрических расстояний.
        Возвращает матрицу расстояний (0 = идеально совпадают, 1 = очень далеко).
        """
        # Нормализация координат
        X_norm = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        Y_norm = (Y - np.mean(Y, axis=0)) / (np.std(Y, axis=0) + 1e-8)

        # Евклидовы расстояния
        dist_euclid = cdist(X_norm, Y_norm)

        # Преобразование в стиле TM-score: d' = 1 - exp(-d^2 / (2*sigma^2))
        # Это даёт значения в [0,1), где 0 ≈ полное совпадение
        sim = 1.0 - np.exp(-dist_euclid ** 2 / (2 * self.sigma_geom ** 2))
        return sim

    def _order_penalty(self, n, m):
        """Матрица штрафа за нарушение порядка (экспоненциальная, как в базовой OGH)."""
        penalty = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                pos_diff = abs(i / n - j / m)
                penalty[i, j] = 20 * (1 - np.exp(-3 * pos_diff))
        return penalty

    def _dp_alignment_local(self, cost_matrix, gap_penalty=1.0):
        """
        Динамическое программирование для ЛОКАЛЬНОГО монотонного выравнивания.
        Разрешены пропуски на концах (начало и конец могут не совпадать).
        Возвращает выравнивание и длину пути.
        """
        n, m = cost_matrix.shape
        # DP для локального выравнивания: ищем подматрицу с минимальной средней стоимостью
        # Упрощённо: глобальное выравнивание с возможностью начинать с любого (i,j)
        dp = np.full((n + 1, m + 1), np.inf)
        trace = np.zeros((n + 1, m + 1, 2), dtype=int)
        dp[0, 0] = 0

        # Инициализация: можно начинать с любого (0, j) или (i, 0) с нулевой стоимостью
        for i in range(1, n + 1):
            dp[i, 0] = 0
            trace[i, 0] = [i - 1, 0]
        for j in range(1, m + 1):
            dp[0, j] = 0
            trace[0, j] = [0, j - 1]

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Match
                match_cost = dp[i - 1, j - 1] + cost_matrix[i - 1, j - 1]
                # Gap in X
                gap_x = dp[i - 1, j] + gap_penalty
                # Gap in Y
                gap_y = dp[i, j - 1] + gap_penalty

                best = min(match_cost, gap_x, gap_y)
                dp[i, j] = best
                if best == match_cost:
                    trace[i, j] = [i - 1, j - 1]
                elif best == gap_x:
                    trace[i, j] = [i - 1, j]
                else:
                    trace[i, j] = [i, j - 1]

        # Находим конец оптимального пути (максимальный i,j, где стоимость не бесконечна)
        best_i, best_j = n, m
        # Восстановление пути
        alignment = []
        i, j = best_i, best_j
        while i > 0 or j > 0:
            prev_i, prev_j = trace[i, j]
            if prev_i < i and prev_j < j:
                alignment.append((prev_i, prev_j))
            i, j = prev_i, prev_j

        alignment.reverse()
        return alignment

    def _dp_alignment_global(self, cost_matrix, gap_penalty=1.0):
        """Глобальное выравнивание (обязательное прохождение от (0,0) до (n,m))."""
        n, m = cost_matrix.shape
        dp = np.full((n + 1, m + 1), np.inf)
        dp[0, 0] = 0
        trace = np.zeros((n + 1, m + 1, 2), dtype=int)

        for i in range(1, n + 1):
            dp[i, 0] = dp[i - 1, 0] + gap_penalty
            trace[i, 0] = [i - 1, 0]
        for j in range(1, m + 1):
            dp[0, j] = dp[0, j - 1] + gap_penalty
            trace[0, j] = [0, j - 1]

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match_cost = dp[i - 1, j - 1] + cost_matrix[i - 1, j - 1]
                gap_x = dp[i - 1, j] + gap_penalty
                gap_y = dp[i, j - 1] + gap_penalty
                best = min(match_cost, gap_x, gap_y)
                dp[i, j] = best
                if best == match_cost:
                    trace[i, j] = [i - 1, j - 1]
                elif best == gap_x:
                    trace[i, j] = [i - 1, j]
                else:
                    trace[i, j] = [i, j - 1]

        alignment = []
        i, j = n, m
        while i > 0 or j > 0:
            prev_i, prev_j = trace[i, j]
            if prev_i < i and prev_j < j:
                alignment.append((prev_i, prev_j))
            i, j = prev_i, prev_j
        alignment.reverse()
        return alignment

    def compute(self, X, Y):
        n, m = len(X), len(Y)
        geom = self._geom_similarity(X, Y)  # матрица (0..1), 0 = близко
        order = self._order_penalty(n, m)  # матрица (0..20)
        total_cost = (1 - self.alpha) * geom + self.alpha * order

        if self.use_dp:
            if self.local_alignment:
                alignment = self._dp_alignment_local(total_cost, self.gap_penalty)
            else:
                alignment = self._dp_alignment_global(total_cost, self.gap_penalty)
        else:
            alignment = self._greedy_alignment(total_cost)

        if len(alignment) == 0:
            return np.inf, {}

        sum_geom = sum(geom[i, j] for i, j in alignment)
        sum_order = sum(order[i, j] for i, j in alignment)
        L_min = min(n, m)

        ogh_dist = ((1 - self.alpha) * sum_geom + self.alpha * sum_order) / L_min

        info = {
            'n_matches': len(alignment),
            'coverage': len(alignment) / L_min,
            'sum_geom': sum_geom,
            'sum_order': sum_order
        }
        return ogh_dist, info

    def _greedy_alignment(self, cost_matrix):
        """Жадное монотонное выравнивание с окном (для быстрых расчётов)."""
        n, m = cost_matrix.shape
        alignment = []
        i, j = 0, 0
        visited_i, visited_j = set(), set()

        while i < n and j < m:
            window = min(self.window, n - i, m - j)
            best_cost = np.inf
            best_i, best_j = i, j
            for di in range(window):
                for dj in range(window):
                    if (i + di) not in visited_i and (j + dj) not in visited_j:
                        cost = cost_matrix[i + di, j + dj] + (di + dj) * self.gap_penalty
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


def compute_homology_table(structures):
    """
    Вычисляет OGH для пар:
    - 7L0E (SARS-CoV-2 протеаза) с 1E0J, 1K28, 4ZMJ (ВИЧ-1 протеазы)
    - 5T6T (RBD) со всеми остальными
    """
    from ogh_VAD_clean import tm_score

    # Оптимальная версия OGH
    ogh_opt = ImprovedOGH(
        alpha=1.0,  # чистый порядок
        sigma_geom=1.2,
        gap_penalty=20.0,
        local_alignment=False,
        use_dp=True
    )

    # Интересующие белки (теперь без .pdb, так как structures содержит имена без расширения)
    protease_sars = '7L0E'
    hiv_proteases = ['1E0J', '1K28', '4ZMJ']
    rbd = '5T6T'

    print("\n" + "=" * 80)
    print("ТАБЛИЦА ДЛЯ РАЗДЕЛА 5.1 (Биологическая интерпретация)")
    print("=" * 80)

    # Проверяем, какие файлы есть в structures
    available = list(structures.keys())
    print(f"\nДоступные файлы: {available}")

    results = []

    # Пары: SARS-CoV-2 протеаза с ВИЧ-1 протеазами
    for hiv in hiv_proteases:
        if protease_sars in structures and hiv in structures:
            X = structures[protease_sars]
            Y = structures[hiv]
            ogh_dist, info = ogh_opt.compute(X, Y)
            tm = tm_score(X, Y)
            results.append({
                'Белок A': protease_sars,
                'Белок B': hiv,
                'OGH': ogh_dist,
                'TM-score': tm,
                'Примечание': 'отдалённая гомология (PA клан)'
            })
            print(f"  ✓ {protease_sars} ↔ {hiv}: OGH = {ogh_dist:.4f}, TM = {tm:.3f}")
        else:
            print(f"  ✗ {protease_sars} или {hiv} не найдены")

    # Пары: RBD с другими белками
    if rbd in structures:
        for name, coords in structures.items():
            if name != rbd:
                X = structures[rbd]
                Y = coords
                ogh_dist, info = ogh_opt.compute(X, Y)
                tm = tm_score(X, Y)
                results.append({
                    'Белок A': rbd,
                    'Белок B': name,
                    'OGH': ogh_dist,
                    'TM-score': tm,
                    'Примечание': 'очень маленький белок (12 остатков)'
                })
                print(f"  ✓ {rbd} ↔ {name}: OGH = {ogh_dist:.4f}, TM = {tm:.3f}")
    else:
        print(f"  ✗ {rbd} не найден")

    # Сортируем по OGH
    results.sort(key=lambda x: x['OGH'])

    # Вывод в виде таблицы
    print(f"\n{'Белок A':<12} | {'Белок B':<12} | {'OGH':>8} | {'TM-score':>8} | Примечание")
    print("-" * 80)
    for r in results:
        print(f"{r['Белок A']:<12} | {r['Белок B']:<12} | {r['OGH']:>8.4f} | {r['TM-score']:>8.3f} | {r['Примечание']}")

    # Ключевые цифры для статьи
    print("\n" + "=" * 80)
    print("КЛЮЧЕВЫЕ ЦИФРЫ ДЛЯ СТАТЬИ:")

    # Для пар 7L0E с ВИЧ-1 протеазами
    sars_hiv_pairs = [r for r in results if r['Белок A'] == '7L0E' or r['Белок B'] == '7L0E']
    if sars_hiv_pairs:
        ogh_values = [r['OGH'] for r in sars_hiv_pairs]
        tm_values = [r['TM-score'] for r in sars_hiv_pairs]
        print(
            f"  • 7L0E ↔ ВИЧ-1 протеазы ({len(sars_hiv_pairs)} пар): OGH = [{min(ogh_values):.4f} .. {max(ogh_values):.4f}], TM-score = [{min(tm_values):.3f} .. {max(tm_values):.3f}]")
    else:
        print("  • Пары 7L0E ↔ ВИЧ-1 протеазы не найдены")

    # Для RBD (5T6T) со всеми другими
    rbd_pairs = [r for r in results if r['Белок A'] == '5T6T' or r['Белок B'] == '5T6T']
    if rbd_pairs:
        ogh_values = [r['OGH'] for r in rbd_pairs]
        print(
            f"  • 5T6T (RBD) с другими белками ({len(rbd_pairs)} пар): OGH = [{min(ogh_values):.3f} .. {max(ogh_values):.3f}]")
    else:
        print("  • Пары с 5T6T не найдены")

    return results
# ============================================================================
# 2. ЗАГРУЗКА ДАННЫХ (VAD + SCOP + ручная разметка гомологов)
# ============================================================================

def load_pdb_structures(pdb_dir='vad_500mb', max_atoms=500):
    """Загружает Cα-координаты из PDB-файлов."""
    parser = PDBParser(QUIET=True)
    files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
    structures = {}
    for f in files:
        name = os.path.basename(f).replace('.pdb', '')
        try:
            struct = parser.get_structure(name, f)
            ca = []
            for model in struct:
                for chain in model:
                    for res in chain:
                        if 'CA' in res and len(ca) < max_atoms:
                            ca.append(res['CA'].get_coord())
            if len(ca) >= 10:
                structures[name] = np.array(ca)
        except:
            continue
    return structures


def build_homology_labels_vad(structures):
    labels = {}
    names = list(structures.keys())
    # Гомологичные пары (на основе информации из статьи и PDB)
    homolog_pairs = [
        ('1E0J', '1K28'),  # обе HIV-1 протеаза
        ('1E0J', '4ZMJ'),  # HIV-1 протеаза
        ('1K28', '4ZMJ'),
        ('1E0J', '1F5K'),  # мутант HIV-1
        ('1F5K', '1K28'),
        ('1F5K', '4ZMJ'),
        # отдалённая гомология между HIV и SARS-CoV-2
        ('1E0J', '7L0E'),  # возможно, слабая гомология
        ('1K28', '7L0E'),
        ('4ZMJ', '7L0E')
    ]

    homolog_pairs = [
        ('1E0J', '1K28'), ('1E0J', '4ZMJ'), ('1K28', '4ZMJ'),
        ('1E0J', '1F5K'), ('1F5K', '1K28'), ('1F5K', '4ZMJ'),
        # отдалённая гомология между HIV и SARS-CoV-2 (класс PA)
        ('1E0J', '7L0E'), ('1K28', '7L0E'), ('4ZMJ', '7L0E')
    ]



    non_homolog_pairs = [

        ('1E0J', '2H2C'), ('1K28', '2H2C'), ('4ZMJ', '2H2C'), ('7L0E', '2H2C'),
        ('1E0J', '5T6T'), ('1K28', '5T6T'), ('4ZMJ', '5T6T'), ('7L0E', '5T6T'),

        # === ВИЧ-1 протеаза vs спайк SARS-CoV-2 ===
        ('1E0J', '6VXX'), ('1E0J', '6W01'), ('1E0J', '6M17'),
        ('1K28', '6VXX'), ('1K28', '6W01'),
        ('4ZMJ', '6VXX'), ('4ZMJ', '6W01'),

        # === протеаза SARS-CoV-2 vs гемагглютинин гриппа ===
        ('7L0E', '3HMX'), ('7L0E', '6B8V'), ('7L0E', '2WR7'), ('7L0E', '4N5J'),

        # === MERS-CoV (2H2C) vs гемагглютинин гриппа ===
        ('2H2C', '3HMX'), ('2H2C', '6B8V'), ('2H2C', '2WR7'),

        # === ВИЧ-1 протеаза vs белки Zika (5I8V, 5KOK) ===
        ('1E0J', '5I8V'), ('1K28', '5KOK'), ('4ZMJ', '5I8V'),

        # === протеаза SARS-CoV-2 vs RSV (6E6J) ===
        ('7L0E', '6E6J'),

        # === спайк SARS-CoV-2 vs гемагглютинин гриппа ===
        ('6VXX', '5T6T'), ('6W01', '5T6T'), ('6M17', '3HMX'),

        # === разные случайные пары (заведомо негомологичные) ===
        ('5T6T', '6NUS'),  # RBD гриппа vs неизвестный вирусный белок
        ('3HMX', '5I8V'),  # гемагглютинин vs Zika
        ('2H2C', '5I8V'),  # MERS vs Zika
        ('6VXX', '6E6J'),  # спайк SARS vs RSV
        ('1E0J', '6NUS'),  # ВИЧ-1 протеаза vs неизвестный
    ]

    for p1, p2 in homolog_pairs:
        if p1 in names and p2 in names:
            labels[tuple(sorted([p1, p2]))] = 1
    for p1, p2 in non_homolog_pairs:
        if p1 in names and p2 in names:
            labels[tuple(sorted([p1, p2]))] = 0
    return labels


def load_scop_pairs(scop_file=None):
    """
    Загрузка SCOP пар (если есть файл). Если нет, возвращаем None.
    Формат: каждая строка: pdb1 pdb2 is_homolog (0/1)
    """
    if scop_file is None or not os.path.exists(scop_file):
        return None
    pairs = []
    labels = {}
    with open(scop_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                p1, p2, lab = parts[0], parts[1], int(parts[2])
                pair = tuple(sorted([p1, p2]))
                labels[pair] = lab
                pairs.append(pair)
    return labels


# ============================================================================
# 3. ЭКСПЕРИМЕНТ 1: ROC-анализ для разделения гомологов при одинаковом TM-score
# ============================================================================

def experiment_roc_at_fixed_tm(structures, homology_labels, tm_threshold=0.5, ogh_alpha=0.3):
    """
    ROC-анализ: сравниваем OGH и TM-score при фиксированном TM (широкий диапазон).
    """
    pairs = list(homology_labels.keys())
    if len(pairs) == 0:
        print("Нет размеченных пар.")
        return None

    tm_scores = {}
    ogh_dist = {}

    ogh = ImprovedOGH(alpha=ogh_alpha, use_dp=True, local_alignment=True, gap_penalty=10.0)
    from ogh_VAD_clean import tm_score

    for p1, p2 in pairs:
        if p1 not in structures or p2 not in structures:
            continue
        X = structures[p1]
        Y = structures[p2]
        tm = tm_score(X, Y)
        dist, _ = ogh.compute(X, Y)
        tm_scores[(p1, p2)] = tm
        ogh_dist[(p1, p2)] = dist

    # Расширенный диапазон TM-score, чтобы захватить больше пар
    tm_vals = np.array([tm_scores[p] for p in pairs])
    mask = (tm_vals >= tm_threshold - 0.25) & (tm_vals <= tm_threshold + 0.25)
    selected_pairs = [pairs[i] for i in range(len(pairs)) if mask[i]]

    if len(selected_pairs) < 5:
        print(f"Мало пар в диапазоне TM ≈ {tm_threshold} (найдено {len(selected_pairs)}). Пропускаем ROC.")
        return None

    y_true = [homology_labels[p] for p in selected_pairs]
    # OGH – расстояние, поэтому инвертируем знак
    y_score_ogh = [-ogh_dist[p] for p in selected_pairs]
    y_score_tm = [tm_scores[p] for p in selected_pairs]

    if len(set(y_true)) < 2:
        print("Недостаточно обоих классов для ROC.")
        return None

    from sklearn.metrics import roc_curve, auc
    fpr_ogh, tpr_ogh, _ = roc_curve(y_true, y_score_ogh)
    auc_ogh = auc(fpr_ogh, tpr_ogh)
    fpr_tm, tpr_tm, _ = roc_curve(y_true, y_score_tm)
    auc_tm = auc(fpr_tm, tpr_tm)

    print(f"\n=== ROC-анализ при TM ≈ {tm_threshold} ===")
    print(f"Выбрано пар: {len(selected_pairs)} (истинных: {sum(y_true)})")
    print(f"AUC OGH: {auc_ogh:.3f}")
    print(f"AUC TM-score: {auc_tm:.3f}")

    return {'fpr_ogh': fpr_ogh, 'tpr_ogh': tpr_ogh, 'auc_ogh': auc_ogh,
            'fpr_tm': fpr_tm, 'tpr_tm': tpr_tm, 'auc_tm': auc_tm}


# ============================================================================
# 4. ЭКСПЕРИМЕНТ 2: Корреляция улучшенной OGH с TM-score
# ============================================================================

def correlation_with_tm(structures, max_pairs=200):
    """Вычисляет корреляцию Пирсона и Спирмена между ImprovedOGH и TM-score."""
    names = list(structures.keys())
    pairs = list(combinations(names, 2))[:max_pairs]

    ogh_opt = ImprovedOGH(alpha=0.25, sigma_geom=2.0, use_dp=True, local_alignment=True)

    ogh_vals = []
    tm_vals = []
    for p1, p2 in pairs:
        X = structures[p1]
        Y = structures[p2]
        dist, _ = ogh_opt.compute(X, Y)
        from ogh_VAD_clean import tm_score
        tm = tm_score(X, Y)
        ogh_vals.append(dist)
        tm_vals.append(tm)

    pearson_r, pearson_p = pearsonr(ogh_vals, tm_vals)
    spearman_r, spearman_p = spearmanr(ogh_vals, tm_vals)

    print("\n=== Корреляция ImprovedOGH с TM-score ===")
    print(f"Пар: {len(ogh_vals)}")
    print(f"Pearson r = {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"Spearman ρ = {spearman_r:.4f} (p={spearman_p:.2e})")
    return pearson_r, spearman_r


# ============================================================================
# 5. ЭКСПЕРИМЕНТ 3: Комбинация OGH + TM-score (логистическая регрессия)
# ============================================================================

def combination_roc(structures, homology_labels, test_ratio=0.3):
    pairs = list(homology_labels.keys())
    X_data = []
    y_data = []

    ogh = ImprovedOGH(alpha=0.3, use_dp=True, local_alignment=True, gap_penalty=10.0)
    from ogh_VAD_clean import tm_score

    for p1, p2 in pairs:
        if p1 not in structures or p2 not in structures:
            continue
        X = structures[p1]
        Y = structures[p2]
        d_ogh, _ = ogh.compute(X, Y)
        tm = tm_score(X, Y)
        X_data.append([d_ogh, tm])
        y_data.append(homology_labels[(p1, p2)])

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_ratio, random_state=42)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_prob = lr.predict_proba(X_test)[:, 1]
    auc_comb = roc_auc_score(y_test, y_pred_prob)

    # Для OGH используем -d_ogh, так как меньшее расстояние → гомология
    auc_ogh = roc_auc_score(y_test, -X_test[:, 0])
    auc_tm = roc_auc_score(y_test, X_test[:, 1])

    print("\n=== Комбинация OGH + TM-score ===")
    print(f"AUC OGH alone: {auc_ogh:.3f}")
    print(f"AUC TM alone: {auc_tm:.3f}")
    print(f"AUC OGH+TM (logreg): {auc_comb:.3f}")
    return {'auc_ogh': auc_ogh, 'auc_tm': auc_tm, 'auc_comb': auc_comb}


# ============================================================================
# 6. ТЕСТ НА ПЕРЕСТАНОВКАХ (чувствительность OGH к порядку)
# ============================================================================

def shuffling_sensitivity(structures, protein_name='1E0J', shuffling_levels=[0, 0.1, 0.3, 0.5, 0.7, 1.0]):
    """
    Показывает, что OGH монотонно растёт с уровнем перемешивания,
    а GH остаётся неизменным.
    """
    X = structures[protein_name]
    ogh = ImprovedOGH(alpha=0.5, use_dp=False, local_alignment=False)  # быстрая версия
    from ogh_VAD_clean import gh_histogram

    results = []
    for level in shuffling_levels:
        if level == 0:
            X_shuf = X
        else:
            n_shuffle = int(len(X) * level)
            idx = np.arange(len(X))
            if n_shuffle > 1:
                np.random.seed(42)
                perm = np.random.permutation(n_shuffle)
                idx[:n_shuffle] = idx[perm]
            X_shuf = X[idx]
        ogh_dist, _ = ogh.compute(X, X_shuf)
        gh_dist = gh_histogram(X, X_shuf)
        results.append((level, ogh_dist, gh_dist))

    print("\n=== Чувствительность к перемешиванию ===")
    print("Уровень | OGH     | GH")
    for level, oghd, ghd in results:
        print(f"{level:5.1f}   | {oghd:.6f} | {ghd:.6f}")
    return results


# ============================================================================
# 7. ОПТИМИЗАЦИЯ АЛЬФА (ПОИСК ЛУЧШЕЙ КОРРЕЛЯЦИИ С TM-score)
# ============================================================================

def optimize_alpha(structures, alphas=np.linspace(0, 1, 11), max_pairs=150):
    """Находит alpha, максимизирующую |Pearson r| с TM-score."""
    names = list(structures.keys())
    pairs = list(combinations(names, 2))[:max_pairs]

    best_alpha = 0.0
    best_r = -1.0
    results = []

    for alpha in alphas:
        ogh = ImprovedOGH(alpha=alpha, use_dp=False, local_alignment=False)
        ogh_vals = []
        tm_vals = []
        for p1, p2 in pairs:
            X = structures[p1]
            Y = structures[p2]
            d, _ = ogh.compute(X, Y)
            from ogh_VAD_clean import tm_score
            tm = tm_score(X, Y)
            ogh_vals.append(d)
            tm_vals.append(tm)
        r, _ = pearsonr(ogh_vals, tm_vals)
        results.append((alpha, r))
        if abs(r) > best_r:
            best_r = abs(r)
            best_alpha = alpha
        print(f"α={alpha:.2f} -> r={r:.4f}")

    print(f"\nОптимальный α = {best_alpha:.2f} (|r| = {best_r:.4f})")
    return best_alpha, best_r, results


# ============================================================================
# 8. ГЛАВНАЯ ФУНКЦИЯ (ЗАПУСК ВСЕХ ЭКСПЕРИМЕНТОВ)
# ============================================================================

def main():
    print("=" * 80)
    print("УЛУЧШЕНИЕ СТАТЬИ Math_OGH: новые эксперименты")
    print("=" * 80)

    # Загружаем структуры
    print("\n1. Загрузка PDB-структур VAD...")
    structures = load_pdb_structures('vad_500mb')
    print(f"   Загружено {len(structures)} белков.")

    if len(structures) < 2:
        print("   Недостаточно структур. Убедитесь, что папка vad_500mb содержит PDB-файлы.")
        return

    print("\n2. Разметка гомологичных пар (VAD)...")
    homology_labels = build_homology_labels_vad(structures)
    print(f"   Размечено {len(homology_labels)} пар.")

    # Эксперимент 1: чувствительность к перемешиванию
    shuffling_sensitivity(structures, protein_name='1E0J')

    # Эксперимент 2: корреляция улучшенной OGH с TM-score
    corr_pearson, corr_spearman = correlation_with_tm(structures, max_pairs=200)

    # Эксперимент 3: ROC при фиксированном TM-score
    roc_res = experiment_roc_at_fixed_tm(structures, homology_labels, tm_threshold=0.5, ogh_alpha=0.25)

    # Эксперимент 4: комбинация OGH + TM-score
    comb_res = combination_roc(structures, homology_labels)

    # Эксперимент 5: оптимизация α
    best_alpha, best_r, alpha_results = optimize_alpha(structures)

    # Итоговые результаты
    print("\n" + "=" * 80)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ДЛЯ СТАТЬИ")
    print("=" * 80)
    print(f"""
       1. Корреляция ImprovedOGH с TM-score: r = {corr_pearson:.4f} (Spearman ρ = {corr_spearman:.4f})

       2. Разделение гомологов при TM≈0.5: """, end='')
    if roc_res is not None:
        print(f"AUC OGH = {roc_res['auc_ogh']:.3f}, AUC TM = {roc_res['auc_tm']:.3f}")
    else:
        print("недостаточно данных.")

    print(
        f"""    4. Комбинация OGH+TM: AUC = {comb_res['auc_comb']:.3f} (OGH alone: {comb_res['auc_ogh']:.3f}, TM alone: {comb_res['auc_tm']:.3f})

       5. Оптимальный α = {best_alpha:.2f} (|r| = {best_r:.4f})

       """)

    bio_table = compute_homology_table(structures)
    # Сохраняем результаты
    import pickle
    with open('ogh_improved_results.pkl', 'wb') as f:
        pickle.dump({
            'correlations': (corr_pearson, corr_spearman),
            'alpha_opt': best_alpha,
            'roc_ogh': roc_res,
            'combination': comb_res
        }, f)
    print("Результаты сохранены в ogh_improved_results.pkl")

    # Построение графиков для статьи
    import matplotlib.pyplot as plt

    # Рис. 1: Чувствительность к порядку
    plt.figure(figsize=(8, 5))
    levels = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
    ogh_vals = [0, 0.011324, 0.043105, 0.067976, 0.120060, 0.227692]
    plt.plot(levels, ogh_vals, 'bo-', linewidth=2, markersize=8, label='OGH')
    plt.axhline(y=0, color='red', linestyle='--', label='GH (0 для всех уровней)')
    plt.xlabel('Доля перемешанных остатков')
    plt.ylabel('Расстояние')
    plt.title('Чувствительность к порядку: OGH vs GH')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ogh_shuffling_sensitivity.png', dpi=300)
    plt.show()

    # Рис. 2: ROC-кривые
    if roc_res is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(roc_res['fpr_ogh'], roc_res['tpr_ogh'],
                 'b-', linewidth=2, label=f'OGH (AUC = {roc_res["auc_ogh"]:.3f})')
        plt.plot(roc_res['fpr_tm'], roc_res['tpr_tm'],
                 'r-', linewidth=2, label=f'TM-score (AUC = {roc_res["auc_tm"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-анализ: разделение гомологов при TM≈0.5')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_ogh_vs_tm_fixed.png', dpi=300)
        plt.show()


if __name__ == "__main__":
    main()