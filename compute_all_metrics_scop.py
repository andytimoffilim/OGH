"""
compute_all_metrics_scop.py — Compute ALL metrics on SCOP/ASTRAL dataset
For Tables 2 and 3
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr, spearmanr, kendalltau, wasserstein_distance, ks_2samp
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mutual_info_score
import warnings
import os
import requests
from Bio.PDB import PDBParser

warnings.filterwarnings('ignore')


# ============================================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
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


def seq_aligned_rmsd(X, Y):
    min_len = min(len(X), len(Y))
    X_sub = X[:min_len]
    Y_sub = Y[:min_len]
    X_centered = X_sub - np.mean(X_sub, axis=0)
    Y_centered = Y_sub - np.mean(Y_sub, axis=0)
    return np.sqrt(np.mean(np.sum((X_centered - Y_centered) ** 2, axis=1)))


# ============================================================================
# 2. OGH КЛАСС
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
        return ogh_dist, {}

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
# 3. ЗАГРУЗКА ДАННЫХ С ЗАГРУЗКОЙ КООРДИНАТ ИЗ PDB
# ============================================================================

def load_pairs_with_coords():
    """Загружает пары и загружает координаты из PDB файлов."""

    if not os.path.exists('homology_real_ogh_results.csv'):
        print("❌ homology_real_ogh_results.csv не найден!")
        return None

    df_main = pd.read_csv('homology_real_ogh_results.csv')
    print(f"✅ Загружено {len(df_main)} пар из SCOP/ASTRAL")

    # Получаем уникальные ID
    all_ids = set(df_main['protein1'].tolist() + df_main['protein2'].tolist())
    print(f"   Уникальных белков: {len(all_ids)}")

    # Загружаем координаты для каждого белка
    print("\n📥 Загрузка координат из PDB...")
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

    # Добавляем координаты в датафрейм
    df_main['coords1'] = df_main['protein1'].apply(lambda x: coords_dict.get(x))
    df_main['coords2'] = df_main['protein2'].apply(lambda x: coords_dict.get(x))

    # Удаляем пары без координат
    df_pairs = df_main[df_main['coords1'].notna() & df_main['coords2'].notna()].copy()
    print(f"   Пар с координатами: {len(df_pairs)}")

    return df_pairs


# ============================================================================
# 4. ВЫЧИСЛЕНИЕ МЕТРИК
# ============================================================================

def compute_all_metrics(df_pairs):
    """Вычисляет все метрики для всех пар."""

    print("\n📊 Вычисление метрик...")

    ogh = OrderedGromovHausdorff(alpha=0.5)
    results = []

    for idx, row in df_pairs.iterrows():
        X = row['coords1']
        Y = row['coords2']

        # OGH
        ogh_dist, _ = ogh.compute(X, Y)

        # GH histogram
        gh_hist = gh_histogram(X, Y)

        # GH KS
        gh_ks_val = gh_ks(X, Y)

        # RMSD
        rmsd_val = rmsd_with_alignment(X, Y)

        # Seq-RMSD
        seq_rmsd_val = seq_aligned_rmsd(X, Y)

        results.append({
            'is_homolog': row['is_homolog'],
            'TM_score': row['TM_score'],
            'OGH': ogh_dist,
            'GH_histogram': gh_hist,
            'GH_KS': gh_ks_val,
            'RMSD': rmsd_val,
            'Seq_RMSD': seq_rmsd_val
        })

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(df_pairs)} пар...")

    return pd.DataFrame(results)


# ============================================================================
# 5. СТАТИСТИКИ ДЛЯ TABLES 2 & 3
# ============================================================================

def compute_stats(df_metrics):
    """Вычисляет статистики для Table 2."""

    print("\n📊 Вычисление статистик для Table 2...")

    metrics = ['OGH', 'GH_histogram', 'GH_KS', 'RMSD', 'Seq_RMSD']
    results = []

    for metric in metrics:
        vals = df_metrics[metric].replace([np.inf, -np.inf], np.nan).dropna()

        if len(vals) == 0:
            continue

        mean_val = vals.mean()
        std_val = vals.std()

        valid = df_metrics[metric].notna() & df_metrics['TM_score'].notna()
        valid = valid & (df_metrics[metric] < 1e6)

        if valid.sum() > 2:
            pearson_r, pearson_p = pearsonr(df_metrics.loc[valid, metric], df_metrics.loc[valid, 'TM_score'])
            spearman_r, spearman_p = spearmanr(df_metrics.loc[valid, metric], df_metrics.loc[valid, 'TM_score'])
            kendall_r, kendall_p = kendalltau(df_metrics.loc[valid, metric], df_metrics.loc[valid, 'TM_score'])
        else:
            pearson_r, pearson_p = np.nan, np.nan
            spearman_r, spearman_p = np.nan, np.nan
            kendall_r, kendall_p = np.nan, np.nan

        results.append({
            'metric': metric,
            'mean': mean_val,
            'std': std_val,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'kendall_r': kendall_r,
            'kendall_p': kendall_p,
            'n_pairs': valid.sum()
        })

    return pd.DataFrame(results)


def compute_stratified_correlations(df_metrics):
    """Вычисляет корреляции по зонам для Table 3."""

    print("\n📊 Вычисление стратифицированных корреляций для Table 3...")

    zones = [
        ('Diverse (TM ≤ 0.3)', df_metrics['TM_score'] <= 0.3),
        ('Twilight (0.3 < TM ≤ 0.5)', (df_metrics['TM_score'] > 0.3) & (df_metrics['TM_score'] <= 0.5)),
        ('Homologous (TM > 0.5)', df_metrics['TM_score'] > 0.5)
    ]

    results = []

    for zone_name, mask in zones:
        zone_df = df_metrics[mask]

        if len(zone_df) < 5:
            continue

        r, p_r = pearsonr(zone_df['OGH'], zone_df['TM_score'])
        rho, p_rho = spearmanr(zone_df['OGH'], zone_df['TM_score'])
        tau, p_tau = kendalltau(zone_df['OGH'], zone_df['TM_score'])

        try:
            ogh_bins = pd.qcut(zone_df['OGH'], q=10, duplicates='drop')
            tm_bins = pd.qcut(zone_df['TM_score'], q=10, duplicates='drop')
            mi = mutual_info_score(ogh_bins.astype(str), tm_bins.astype(str))
            nmi = mi / (0.5 * (np.log2(len(ogh_bins.unique())) + np.log2(len(tm_bins.unique()))))
        except:
            mi, nmi = np.nan, np.nan

        results.append({
            'zone': zone_name,
            'N': len(zone_df),
            'pearson_r': r,
            'pearson_p': p_r,
            'spearman_rho': rho,
            'spearman_p': p_rho,
            'kendall_tau': tau,
            'kendall_p': p_tau,
            'MI': mi,
            'NMI': nmi
        })

    return pd.DataFrame(results)


# ============================================================================
# 6. MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("📊 ВЫЧИСЛЕНИЕ ВСЕХ МЕТРИК НА SCOP/ASTRAL")
    print("=" * 80)

    df_pairs = load_pairs_with_coords()
    if df_pairs is None or len(df_pairs) == 0:
        print("\n❌ Нет пар с координатами!")
        return

    # Вычисляем все метрики
    df_metrics = compute_all_metrics(df_pairs)

    # Сохраняем
    df_metrics.to_csv('scop_all_metrics.csv', index=False)
    print(f"\n✅ Сохранено: scop_all_metrics.csv")

    # Table 2
    df_stats = compute_stats(df_metrics)
    df_stats.to_csv('table2_scop.csv', index=False)
    print(f"\n✅ Table 2 сохранена: table2_scop.csv")

    # Table 3
    df_strat = compute_stratified_correlations(df_metrics)
    df_strat.to_csv('table3_scop.csv', index=False)
    print(f"\n✅ Table 3 сохранена: table3_scop.csv")

    # Вывод
    print("\n" + "=" * 80)
    print("📋 TABLE 2 (SCOP/ASTRAL)")
    print("=" * 80)
    print(df_stats.to_string(index=False))

    print("\n" + "=" * 80)
    print("📋 TABLE 3 (SCOP/ASTRAL)")
    print("=" * 80)
    print(df_strat.to_string(index=False))

    print("\n✅ Готово!")


if __name__ == "__main__":
    main()