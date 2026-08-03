"""
rin_ogh_advanced.py - RIN-OGH: Enhanced OGH for Residue Interaction Networks

Authors: Timofeev A.V., Anufriev A.S.
Organization: AI Center for SCO+ Countries, St. Petersburg, Russia
Date: 2026
License: MIT

Description:
    Adaptation of the Ordered Gromov-Hausdorff metric for comparing
    Residue Interaction Networks (RINs) — graphs where nodes are amino acids
    and edges represent non-covalent interactions.

Features (beyond basic OGH):
    - Weighted edges (weight ∝ 1/distance²)
    - Graph distance normalization by protein size
    - Length ratio filtering (max_length_ratio)
    - DP alignment algorithm for better accuracy

Main Components:
    Class WeightedRINGraph
        Builds a weighted interaction graph from a PDB file
    Class ImprovedRIN_OGH
        Computes RIN-OGH distance with configurable parameters

Dependencies:
    numpy, scipy, Bio.PDB (Biopython), matplotlib, networkx (optional)

Output:
    - rin_graph_stats.png - graph statistics visualization
    - Console output comparing binary vs weighted edges

Usage:
    >>> python rin_ogh_advanced.py

Recommended Parameters:
    order_weight = 0.1-0.3 (order is less critical for RIN)
    max_length_ratio = 2.0 (compare proteins with length difference ≤2×)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path, floyd_warshall
from scipy.spatial.distance import cdist
from Bio.PDB import PDBParser, NeighborSearch, is_aa
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 1. ПОСТРОЕНИЕ RIN С ВЗВЕШЕННЫМИ РЁБРАМИ
# ============================================================================

class WeightedRINGraph:
    """
    Построение взвешенного графа остаточных взаимодействий (Weighted RIN)
    Рёбра имеют веса, пропорциональные силе взаимодействия
    """

    def __init__(self, pdb_file, distance_cutoff=5.0, weight_power=2.0):
        """
        Parameters:
        -----------
        pdb_file : str
            Путь к PDB-файлу
        distance_cutoff : float
            Порог расстояния (Å) для определения взаимодействия
        weight_power : float
            Степень для преобразования расстояния в вес (вес = 1 / (расстояние)^p)
        """
        self.pdb_file = pdb_file
        self.distance_cutoff = distance_cutoff
        self.weight_power = weight_power
        self.contacts = None
        self.contact_weights = None
        self.distance_matrix = None
        self.residues = []
        self.residue_indices = {}
        self._parse_pdb()
        self._build_weighted_rin()
        self._compute_graph_distances()

    def _parse_pdb(self):
        """Извлечение остатков и их Cα-координат"""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', self.pdb_file)

        residues = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if is_aa(residue):
                        res_id = (chain.id, residue.id[1], residue.id[2] if residue.id[2] != ' ' else '')
                        residues.append({
                            'id': res_id,
                            'name': residue.resname,
                            'position': residue.id[1],
                            'chain': chain.id,
                            'ca_coord': None
                        })
                        if 'CA' in residue:
                            residues[-1]['ca_coord'] = residue['CA'].get_coord()

        self.residues = sorted(residues, key=lambda x: (x['chain'], x['position']))
        for idx, res in enumerate(self.residues):
            self.residue_indices[res['id']] = idx

    def _build_weighted_rin(self):
        """Построение взвешенного графа взаимодействий"""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', self.pdb_file)

        # Собираем все атомы
        atoms = []
        atom_to_residue = {}
        for atom in structure.get_atoms():
            atoms.append(atom)
            parent = atom.get_parent()
            if parent and is_aa(parent):
                res_id = (parent.get_parent().id, parent.id[1], parent.id[2] if parent.id[2] != ' ' else '')
                if res_id in self.residue_indices:
                    atom_to_residue[atom] = self.residue_indices[res_id]

        # Поиск контактов между атомами с весами
        ns = NeighborSearch(atoms)

        contacts = defaultdict(set)
        contact_weights = defaultdict(dict)

        for atom in atoms:
            if atom not in atom_to_residue:
                continue
            res_i = atom_to_residue[atom]
            neighbors = ns.search(atom.coord, self.distance_cutoff)
            for neighbor in neighbors:
                if neighbor not in atom_to_residue:
                    continue
                res_j = atom_to_residue[neighbor]
                if res_i != res_j:
                    # Расстояние между атомами
                    dist = np.linalg.norm(atom.coord - neighbor.coord)
                    # Вес = 1 / (расстояние)^p (чем ближе, тем больше вес)
                    weight = 1.0 / (dist ** self.weight_power + 1e-8)

                    # Сохраняем максимальный вес для пары остатков
                    if res_j in contact_weights[res_i]:
                        contact_weights[res_i][res_j] = max(contact_weights[res_i][res_j], weight)
                        contact_weights[res_j][res_i] = contact_weights[res_i][res_j]
                    else:
                        contact_weights[res_i][res_j] = weight
                        contact_weights[res_j][res_i] = weight

                    contacts[res_i].add(res_j)
                    contacts[res_j].add(res_i)

        self.contacts = contacts
        self.contact_weights = contact_weights
        self.n_residues = len(self.residues)

    def _compute_graph_distances(self):
        """Вычисление матрицы кратчайших путей во взвешенном графе"""
        # Строим матрицу смежности с весами (чем больше вес, тем ближе)
        # Преобразуем вес в расстояние: dist = 1/weight
        adj_matrix = np.full((self.n_residues, self.n_residues), np.inf)

        for i, neighbors in self.contact_weights.items():
            for j, weight in neighbors.items():
                # Расстояние обратно пропорционально весу
                adj_matrix[i, j] = 1.0 / (weight + 1e-8)
                adj_matrix[j, i] = 1.0 / (weight + 1e-8)

        # Диагональ = 0
        for i in range(self.n_residues):
            adj_matrix[i, i] = 0.0

        # Алгоритм Флойда-Уоршелла для кратчайших путей
        self.distance_matrix = floyd_warshall(adj_matrix, directed=False)

        # Заменяем бесконечности на большое число
        self.distance_matrix[np.isinf(self.distance_matrix)] = 100.0

    def get_graph_distances(self):
        """Возвращает матрицу расстояний в графе"""
        return self.distance_matrix

    def get_normalized_distances(self):
        """Возвращает НОРМАЛИЗОВАННУЮ матрицу расстояний (по среднему)"""
        dist = self.distance_matrix.copy()
        # Нормализуем по среднему (убираем зависимость от размера белка)
        mean_dist = np.mean(dist[dist < 100])
        if mean_dist > 0:
            dist = dist / mean_dist
        return dist

    def get_ca_coords(self):
        """Возвращает координаты Cα-атомов"""
        coords = []
        for res in self.residues:
            if res['ca_coord'] is not None:
                coords.append(res['ca_coord'])
            else:
                coords.append([0, 0, 0])
        return np.array(coords)

    def get_residue_order(self):
        """Возвращает порядок остатков"""
        return [res['position'] for res in self.residues]

    def get_graph_stats(self):
        """Возвращает статистику графа"""
        n_nodes = self.n_residues
        n_edges = sum(len(v) for v in self.contacts.values()) // 2
        avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
        return {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'avg_degree': avg_degree,
            'density': n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
        }


# ============================================================================
# 2. УЛУЧШЕННЫЙ RIN-OGH С НОРМАЛИЗАЦИЕЙ
# ============================================================================

class ImprovedRIN_OGH:
    """
    Улучшенная версия RIN-OGH с:
    - Нормализацией графовых расстояний
    - Взвешенными рёбрами
    - Фильтрацией по разнице в длине
    """

    def __init__(self, order_weight=0.3, gap_penalty=1.0, window=15, use_dp=True,
                 use_normalization=True, max_length_ratio=2.0):
        """
        Parameters:
        -----------
        order_weight : float
            Вес порядка (α)
        gap_penalty : float
            Штраф за пропуск остатка
        window : int
            Размер окна для жадного алгоритма
        use_dp : bool
            Использовать ли DP алгоритм
        use_normalization : bool
            Нормализовать ли расстояния по размеру белка
        max_length_ratio : float
            Максимальное отношение длин для сравнения (2.0 = разница не более 2×)
        """
        self.order_weight = order_weight
        self.gap_penalty = gap_penalty
        self.window = window
        self.use_dp = use_dp
        self.use_normalization = use_normalization
        self.max_length_ratio = max_length_ratio

    def _check_length_compatibility(self, n, m):
        """Проверяет, можно ли сравнивать белки с такой разницей в длине"""
        ratio = max(n, m) / min(n, m)
        return ratio <= self.max_length_ratio

    def compute(self, rin1, rin2):
        n, m = len(rin1.residues), len(rin2.residues)

        if not self._check_length_compatibility(n, m):
            return np.inf, {'error': f'Length ratio {max(n, m) / min(n, m):.1f} exceeds limit {self.max_length_ratio}'}

        # Получаем матрицы расстояний (с нормализацией)
        if self.use_normalization:
            D1 = rin1.get_normalized_distances()
            D2 = rin2.get_normalized_distances()
        else:
            D1 = rin1.get_graph_distances()
            D2 = rin2.get_graph_distances()

        # Межбелковая матрица расстояний
        # Используем разницу в расстояниях до других остатков (более стабильно)
        D_combined = np.zeros((n, m))

        # Нормализуем матрицы ещё раз для сравнения
        max1 = np.max(D1[D1 < 100])
        max2 = np.max(D2[D2 < 100])
        D1_norm = D1 / (max1 + 1e-8)
        D2_norm = D2 / (max2 + 1e-8)

        for i in range(n):
            for j in range(m):
                # Сравниваем векторы расстояний
                vec_i = D1_norm[i, :]
                vec_j = D2_norm[j, :]
                min_len = min(len(vec_i), len(vec_j))
                vec_i = vec_i[:min_len]
                vec_j = vec_j[:min_len]

                # Используем корреляцию вместо евклидова расстояния
                if np.std(vec_i) > 0 and np.std(vec_j) > 0:
                    corr = np.corrcoef(vec_i, vec_j)[0, 1]
                    D_combined[i, j] = 1 - max(0, corr)  # 0 = идеальное совпадение
                else:
                    D_combined[i, j] = np.linalg.norm(vec_i - vec_j) / np.sqrt(min_len)

        # Штраф за порядок
        order_penalty = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                pos_diff = abs(i / n - j / m)
                order_penalty[i, j] = self.order_weight * 20 * (1 - np.exp(-3 * pos_diff))

        total_cost = D_combined + order_penalty

        # Выравнивание
        if self.use_dp:
            alignment = self._dp_alignment(total_cost)
        else:
            alignment = self._monotonic_alignment(total_cost)

        if len(alignment) == 0:
            return np.inf, {}

        align_i = [a[0] for a in alignment]
        align_j = [a[1] for a in alignment]

        matched_geom = [D_combined[i, j] for i, j in alignment]
        matched_order = [order_penalty[i, j] for i, j in alignment]

        metric_dist = np.mean(matched_geom)
        order_dist = np.mean(matched_order)

        ogh_dist = (1 - self.order_weight) * metric_dist + self.order_weight * order_dist
        ogh_dist = ogh_dist / np.sqrt(min(n, m))

        return ogh_dist, {
            'metric_component': metric_dist,
            'order_component': order_dist,
            'n_matches': len(alignment),
            'coverage': len(alignment) / max(n, m),
            'n1': n, 'n2': m,
            'length_ratio': max(n, m) / min(n, m)
        }

    def _monotonic_alignment(self, cost_matrix):
        """Жадное монотонное выравнивание"""
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

    def _dp_alignment(self, cost_matrix):
        """Глобальное динамическое программирование"""
        n, m = cost_matrix.shape
        dp = np.full((n + 1, m + 1), np.inf)
        dp[0, 0] = 0
        trace = np.zeros((n + 1, m + 1, 2), dtype=int)

        for i in range(n + 1):
            for j in range(m + 1):
                if i > 0 and j > 0:
                    cost = dp[i - 1, j - 1] + cost_matrix[i - 1, j - 1]
                    if cost < dp[i, j]:
                        dp[i, j] = cost
                        trace[i, j] = [i - 1, j - 1]
                if i > 0:
                    cost = dp[i - 1, j] + self.gap_penalty
                    if cost < dp[i, j]:
                        dp[i, j] = cost
                        trace[i, j] = [i - 1, j]
                if j > 0:
                    cost = dp[i, j - 1] + self.gap_penalty
                    if cost < dp[i, j]:
                        dp[i, j] = cost
                        trace[i, j] = [i, j - 1]

        # Восстановление пути
        alignment = []
        i, j = n, m
        while i > 0 or j > 0:
            prev_i, prev_j = trace[i, j]
            if prev_i < i and prev_j < j:
                alignment.append((prev_i, prev_j))
            i, j = prev_i, prev_j

        return alignment[::-1]


# ============================================================================
# 3. ФИЛЬТРАЦИЯ ПО РАЗНИЦЕ В ДЛИНЕ И АНАЛИЗ
# ============================================================================

def filter_by_length_ratio(structures, max_ratio=2.0):
    """Фильтрует пары белков по отношению длин"""
    names = list(structures.keys())
    compatible_pairs = []

    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i < j:
                len1 = len(structures[name1].residues)
                len2 = len(structures[name2].residues)
                ratio = max(len1, len2) / min(len1, len2)
                if ratio <= max_ratio:
                    compatible_pairs.append((name1, name2, ratio))

    return compatible_pairs


def analyze_with_improved_rin_ogh(structures, max_ratio=2.0):
    """Анализ с улучшенным RIN-OGH на отфильтрованных парах"""
    print("\n" + "=" * 70)
    print("АНАЛИЗ С УЛУЧШЕННЫМ RIN-OGH (нормализация + взвешенные рёбра)")
    print("=" * 70)

    # Фильтрация пар
    compatible = filter_by_length_ratio(structures, max_ratio)
    print(f"\n  Найдено {len(compatible)} пар с отношением длин ≤ {max_ratio}")

    if not compatible:
        print("  Нет совместимых пар для анализа")
        return []

    # Инициализация
    ogh = ImprovedRIN_OGH(order_weight=0.3, gap_penalty=1.0, window=15,
                          use_dp=True, use_normalization=True, max_length_ratio=max_ratio)

    results = []

    for name1, name2, ratio in compatible:
        print(f"\n  Сравнение: {name1} ↔ {name2} (отношение длин = {ratio:.2f})")

        dist, info = ogh.compute(structures[name1], structures[name2])

        if np.isinf(dist):
            print(f"    ⚠️ Пропущено: {info.get('error', 'несовместимы')}")
            continue

        print(f"    RIN-OGH:       {dist:.4f}")
        print(f"    Совпадений:    {info['n_matches']}")
        print(f"    Покрытие:      {info['coverage']:.2%}")
        print(f"    Геометрическая: {info['metric_component']:.4f}")
        print(f"    Порядковая:     {info['order_component']:.4f}")

        results.append({
            'protein1': name1,
            'protein2': name2,
            'len1': info['n1'],
            'len2': info['n2'],
            'length_ratio': ratio,
            'RIN_OGH': dist,
            'n_matches': info['n_matches'],
            'coverage': info['coverage'],
            'geometric': info['metric_component'],
            'order': info['order_component']
        })

    return results


# ============================================================================
# 4. СРАВНИТЕЛЬНЫЙ АНАЛИЗ: бинарные vs взвешенные рёбра
# ============================================================================

def compare_edge_types(rin1, rin2):
    """Сравнивает RIN-OGH на бинарных и взвешенных рёбрах"""
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ: бинарные рёбра vs взвешенные рёбра")
    print("=" * 70)

    # Для бинарных рёбер используем обычный RINGraph из текущего модуля
    # Создаём бинарную версию на месте (без весов)

    # Бинарный вариант (все веса = 1)
    class BinaryRINGraph:
        def __init__(self, pdb_file, distance_cutoff=5.0):
            self.pdb_file = pdb_file
            self.distance_cutoff = distance_cutoff
            self.residues = rin1.residues  # копируем из взвешенного
            self.n_residues = len(self.residues)

            # Строим бинарные контакты из взвешенных
            self.contacts = rin1.contacts
            self.contact_weights = {i: {j: 1.0 for j in neigh} for i, neigh in rin1.contacts.items()}
            self._compute_binary_distances()

        def _compute_binary_distances(self):
            """Вычисление матрицы расстояний для бинарного графа"""
            adj_matrix = np.full((self.n_residues, self.n_residues), np.inf)
            for i, neighbors in self.contact_weights.items():
                for j in neighbors:
                    adj_matrix[i, j] = 1.0  # бинарное расстояние = 1
                    adj_matrix[j, i] = 1.0
            for i in range(self.n_residues):
                adj_matrix[i, i] = 0.0

            from scipy.sparse.csgraph import floyd_warshall
            self.distance_matrix = floyd_warshall(adj_matrix, directed=False)
            self.distance_matrix[np.isinf(self.distance_matrix)] = 100.0

        def get_normalized_distances(self):
            dist = self.distance_matrix.copy()
            mean_dist = np.mean(dist[dist < 100])
            if mean_dist > 0:
                dist = dist / mean_dist
            return dist

        def get_graph_distances(self):
            return self.distance_matrix

        def get_ca_coords(self):
            return rin1.get_ca_coords()

    try:
        # Создаём бинарную версию
        bin_rin1 = BinaryRINGraph(rin1.pdb_file)
        bin_rin2 = BinaryRINGraph(rin2.pdb_file)

        ogh_bin = ImprovedRIN_OGH(order_weight=0.3, use_normalization=True, use_dp=True)
        dist_bin, info_bin = ogh_bin.compute(bin_rin1, bin_rin2)

        ogh_weighted = ImprovedRIN_OGH(order_weight=0.3, use_normalization=True, use_dp=True)
        dist_weighted, info_weighted = ogh_weighted.compute(rin1, rin2)

        print(f"\n  Бинарные рёбра:")
        print(f"    RIN-OGH: {dist_bin:.4f}")
        print(f"    Совпадений: {info_bin.get('n_matches', 'N/A')}")
        print(f"    Покрытие: {info_bin.get('coverage', 0):.2%}")

        print(f"\n  Взвешенные рёбра:")
        print(f"    RIN-OGH: {dist_weighted:.4f}")
        print(f"    Совпадений: {info_weighted.get('n_matches', 'N/A')}")
        print(f"    Покрытие: {info_weighted.get('coverage', 0):.2%}")

        # Сравнение
        if dist_weighted < dist_bin:
            print(f"\n  ✅ Взвешенные рёбра дают меньшее расстояние (лучшее совпадение)")
        else:
            print(f"\n  ⚠️ Бинарные рёбра дают меньшее расстояние")

        return (dist_bin, info_bin), (dist_weighted, info_weighted)

    except Exception as e:
        print(f"\n  ❌ Ошибка при сравнении: {e}")
        return None, None


# ============================================================================
# 5. ВИЗУАЛИЗАЦИЯ СТАТИСТИКИ ГРАФА
# ============================================================================

def visualize_graph_stats(structures):
    """Визуализация статистики графов RIN"""
    print("\n" + "=" * 70)
    print("СТАТИСТИКА ГРАФОВ RIN")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = []
    n_nodes = []
    n_edges = []
    densities = []
    avg_degrees = []

    for name, rin in structures.items():
        stats = rin.get_graph_stats()
        names.append(name.replace('.pdb', ''))
        n_nodes.append(stats['n_nodes'])
        n_edges.append(stats['n_edges'])
        densities.append(stats['density'])
        avg_degrees.append(stats['avg_degree'])

        print(f"\n  {name}:")
        print(f"    Вершин: {stats['n_nodes']}")
        print(f"    Рёбер: {stats['n_edges']}")
        print(f"    Плотность: {stats['density']:.4f}")
        print(f"    Средняя степень: {stats['avg_degree']:.2f}")

    # Графики
    ax = axes[0, 0]
    bars = ax.bar(names, n_nodes, color='steelblue', alpha=0.7)
    ax.set_ylabel('Количество вершин')
    ax.set_title('Размеры графов RIN')
    ax.tick_params(axis='x', rotation=45)

    ax = axes[0, 1]
    bars = ax.bar(names, n_edges, color='coral', alpha=0.7)
    ax.set_ylabel('Количество рёбер')
    ax.set_title('Плотность связей в RIN')
    ax.tick_params(axis='x', rotation=45)

    ax = axes[1, 0]
    bars = ax.bar(names, densities, color='seagreen', alpha=0.7)
    ax.set_ylabel('Плотность')
    ax.set_title('Плотность графов RIN')
    ax.tick_params(axis='x', rotation=45)

    ax = axes[1, 1]
    bars = ax.bar(names, avg_degrees, color='purple', alpha=0.7)
    ax.set_ylabel('Средняя степень')
    ax.set_title('Средняя степень вершин')
    ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Статистика графов RIN для вирусных белков VAD', fontsize=14)
    plt.tight_layout()
    plt.savefig('rin_graph_stats.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  Сохранено: rin_graph_stats.png")


# ============================================================================
# 6. ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("=" * 70)
    print("RIN-OGH УЛУЧШЕННАЯ ВЕРСИЯ")
    print("Нормализация | Взвешенные рёбра | Фильтрация по длине")
    print("=" * 70)

    # 1. Загрузка структур со взвешенными рёбрами
    print("\n1. Загрузка PDB файлов со взвешенными рёбрами...")

    pdb_files = ['vad_500mb/1E0J.pdb', 'vad_500mb/1F5K.pdb',
                 'vad_500mb/1K28.pdb', 'vad_500mb/2H2C.pdb']

    structures = {}
    for pdb_file in pdb_files:
        try:
            rin = WeightedRINGraph(pdb_file, distance_cutoff=5.0, weight_power=2.0)
            structures[rin.pdb_file.split('/')[-1]] = rin
            print(f"  Загружено: {rin.pdb_file.split('/')[-1]} ({len(rin.residues)} остатков)")
        except FileNotFoundError:
            print(f"  Пропущено: {pdb_file} (файл не найден)")

    if not structures:
        print("\n  ❌ Нет загруженных структур. Убедитесь, что PDB файлы существуют.")
        return

    # 2. Статистика графов
    visualize_graph_stats(structures)

    # 3. Анализ с разными ограничениями по длине
    for max_ratio in [1.5, 2.0, 3.0]:
        results = analyze_with_improved_rin_ogh(structures, max_ratio=max_ratio)

        if results:
            print(f"\n  --- Результаты (max_ratio={max_ratio}) ---")
            for r in results:
                print(
                    f"    {r['protein1']} ↔ {r['protein2']}: RIN-OGH={r['RIN_OGH']:.4f}, покрытие={r['coverage']:.2%}")

    # 4. Сравнение бинарных и взвешенных рёбер (если есть хотя бы 2 структуры)
    if len(structures) >= 2:
        names = list(structures.keys())
        print("\n" + "=" * 70)
        print("СРАВНЕНИЕ: бинарные рёбра vs взвешенные рёбра")
        print("=" * 70)

        for i in range(min(2, len(names) - 1)):
            compare_edge_types(structures[names[i]], structures[names[i + 1]])

    # 5. Итоговые рекомендации
    print("\n" + "=" * 70)
    print("ИТОГОВЫЕ РЕКОМЕНДАЦИИ")
    print("=" * 70)

    print("""
    📊 ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ ДЛЯ RIN-OGH:

    1. Нормализация графовых расстояний:
       ✅ ВКЛЮЧИТЬ (use_normalization=True)
       → Убирает зависимость от размера белка

    2. Взвешенные рёбра:
       ✅ ВКЛЮЧИТЬ (вес = 1/расстояние²)
       → Учитывает силу взаимодействий

    3. Фильтрация по длине:
       ✅ max_length_ratio = 2.0
       → Сравнивать только белки с разницей в длине не более 2×

    4. Алгоритм выравнивания:
       ✅ use_dp = True (для точности)
       ⚠️ use_dp = False (для скорости, если белков много)

    5. Вес порядка (α):
       💡 Рекомендуется α = 0.1-0.3 для RIN
       (порядок менее критичен, чем для геометрии Cα)

    🔧 ПРИМЕР ИСПОЛЬЗОВАНИЯ:

    ogh = ImprovedRIN_OGH(
        order_weight=0.2,
        gap_penalty=1.0,
        window=15,
        use_dp=True,
        use_normalization=True,
        max_length_ratio=2.0
    )

    distance, info = ogh.compute(rin1, rin2)
    """)


if __name__ == "__main__":
    main()