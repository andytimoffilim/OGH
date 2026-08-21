"""
BASIC version
TM-SCORE via Biopython (VALIDATED IMPLEMENTATION)
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
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.Polypeptide import PPBuilder

warnings.filterwarnings('ignore')

# ============================================================================
# 0. FIX PROTEINBUILDER ERROR
# ============================================================================

from Bio.PDB import Structure, Model, Chain, Residue, Atom


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
    """
    TM-score via Biopython Superimposer.
    """
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


def tm_score_verified(X, Y):
    """Validated TM-score implementation."""
    try:
        return tm_score_biopython(X, Y)
    except Exception as e:
        print(f"⚠️  Biopython error: {e}, using fallback")
        return tm_score_fallback(X, Y)


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
    return np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / L_min


# ============================================================================
# 1. EXPERIMENT PARAMETERS
# ============================================================================

MAX_PAIRS = 700
MAX_FAMILIES = 20
ORDER_WEIGHT = 0.5
WINDOW_SIZE = 15
GAP_PENALTY = 3.0
SIGMA = 3.0

WINDOW_SIZES = [10, 15, 20, 25]
GAP_PENALTIES = [1.0, 2.0, 3.0]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

print(f"""
⚙️  EXPERIMENT PARAMETERS:
   MAX_PAIRS      = {MAX_PAIRS} (total pairs: {MAX_PAIRS * 2})
   MAX_FAMILIES   = {MAX_FAMILIES}
   ORDER_WEIGHT (α)= {ORDER_WEIGHT}
   WINDOW_SIZE (w)= {WINDOW_SIZE}
   GAP_PENALTY (γ)= {GAP_PENALTY}
   SIGMA (σ)      = {SIGMA}
""")


# ============================================================================
# 2. OGH CLASS
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

        # VECTORIZED
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
# 3. PDB LOADING AND PAIR BUILDING
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


def build_pairs_with_pdb(df):
    print(f"\nBuilding pairs from {len(df)} records...")
    df = df[df['sequence'].notna() & (df['sequence'] != '')].copy()
    print(f"With sequences: {len(df)}")

    families = df[df['family'] != '']['family'].unique()
    print(f"Families: {len(families)}")

    pairs = []
    homolog_count = 0

    for family in families[:MAX_FAMILIES]:
        proteins = df[df['family'] == family]
        ids = proteins['id'].tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pdb1 = download_pdb_for_id(ids[i], 'pdb_scop')
                pdb2 = download_pdb_for_id(ids[j], 'pdb_scop')
                if pdb1 and pdb2:
                    coords1 = load_pdb_coords(pdb1)
                    coords2 = load_pdb_coords(pdb2)
                    if coords1 is not None and coords2 is not None:
                        pairs.append({
                            'protein1': ids[i], 'protein2': ids[j],
                            'coords1': coords1, 'coords2': coords2,
                            'is_homolog': 1, 'family': family
                        })
                        homolog_count += 1
                        if len(pairs) >= MAX_PAIRS:
                            break
            if len(pairs) >= MAX_PAIRS:
                break
        if len(pairs) >= MAX_PAIRS:
            break

    print(f"Homologous pairs: {homolog_count}")

    non_homolog_count = 0
    np.random.seed(2)
    all_ids = df['id'].tolist()
    attempts = 0
    max_attempts = 20000

    while non_homolog_count < homolog_count and len(pairs) < MAX_PAIRS * 2 and attempts < max_attempts:
        attempts += 1
        p1, p2 = np.random.choice(all_ids, 2, replace=False)
        f1 = df[df['id'] == p1]['family'].iloc[0] if not df[df['id'] == p1].empty else ''
        f2 = df[df['id'] == p2]['family'].iloc[0] if not df[df['id'] == p2].empty else ''

        if f1 != f2 and f1 != '' and f2 != '':
            pdb1 = download_pdb_for_id(p1, 'pdb_scop')
            pdb2 = download_pdb_for_id(p2, 'pdb_scop')
            if pdb1 and pdb2:
                coords1 = load_pdb_coords(pdb1)
                coords2 = load_pdb_coords(pdb2)
                if coords1 is not None and coords2 is not None:
                    pairs.append({
                        'protein1': p1, 'protein2': p2,
                        'coords1': coords1, 'coords2': coords2,
                        'is_homolog': 0, 'family': f"{f1} vs {f2}"
                    })
                    non_homolog_count += 1

    df_pairs = pd.DataFrame(pairs)
    print(f"Total pairs: {len(df_pairs)}")
    print(f"  Homologs: {df_pairs['is_homolog'].sum()}")
    print(f"  Non-homologs: {len(df_pairs) - df_pairs['is_homolog'].sum()}")
    return df_pairs


# ============================================================================
# 4. METRIC COMPUTATION
# ============================================================================

def compute_metrics(df_pairs, alpha=0.5, window_size=15, gap_penalty=2.0):
    print(f"\nComputing metrics (α={alpha}, w={window_size}, γ={gap_penalty})...")

    ogh = OrderedGromovHausdorff(
        alpha=alpha,
        align_sigma=SIGMA,
        align_order_weight=1.0,
        window_size=window_size,
        gap_penalty=gap_penalty,
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
            'alpha': alpha,
            'window_size': window_size,
            'gap_penalty': gap_penalty
        })

        if (idx + 1) % 20 == 0 or idx == len(df_pairs) - 1:
            print(f"  Processed {idx + 1}/{len(df_pairs)} pairs...")

    return pd.DataFrame(results)


# ============================================================================
# 5. ZONE-SPECIFIC ANALYSIS (DIVERSE, TWILIGHT, HOMOLOGOUS)
# ============================================================================

def run_zone_analysis(df_metrics):
    """
    ROC analysis stratified by TM-score zones:
    - Diverse: TM ≤ 0.3
    - Twilight: 0.3 < TM ≤ 0.5
    - Homologous: TM > 0.5
    """
    print("\n" + "=" * 80)
    print("📊 ZONE-SPECIFIC ANALYSIS (TM-score strata)")
    print("=" * 80)

    zones = [
        ('Diverse (TM ≤ 0.3)', 'TM_score <= 0.3'),
        ('Twilight (0.3 < TM ≤ 0.5)', '(TM_score > 0.3) & (TM_score <= 0.5)'),
        ('Homologous (TM > 0.5)', 'TM_score > 0.5')
    ]

    zone_results = []

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (zone_name, condition) in enumerate(zones):
        mask = df_metrics.eval(condition)
        zone_df = df_metrics[mask]

        if len(zone_df) < 5:
            print(f"\n{zone_name}: insufficient data ({len(zone_df)} pairs)")
            continue

        y_true = zone_df['is_homolog'].values

        # OGH (invert: smaller = closer = more homologous)
        fpr_ogh, tpr_ogh, _ = roc_curve(y_true, -zone_df['d_hat_OGH'].values)
        auc_ogh = auc(fpr_ogh, tpr_ogh)

        # TM-score (larger = closer)
        fpr_tm, tpr_tm, _ = roc_curve(y_true, zone_df['TM_score'].values)
        auc_tm = auc(fpr_tm, tpr_tm)

        # Combined
        ogh_norm = 1.0 - (zone_df['d_hat_OGH'] - zone_df['d_hat_OGH'].min()) / \
                         (zone_df['d_hat_OGH'].max() - zone_df['d_hat_OGH'].min() + 1e-8)
        tm_norm = (zone_df['TM_score'] - zone_df['TM_score'].min()) / \
                  (zone_df['TM_score'].max() - zone_df['TM_score'].min() + 1e-8)
        combined = 0.5 * ogh_norm + 0.5 * tm_norm
        fpr_comb, tpr_comb, _ = roc_curve(y_true, combined.values)
        auc_comb = auc(fpr_comb, tpr_comb)

        # Length ratio (control)
        length_ratio = zone_df['len1'] / (zone_df['len2'] + 1)
        fpr_len, tpr_len, _ = roc_curve(y_true, length_ratio.values)
        auc_len = auc(fpr_len, tpr_len)

        print(f"\n{zone_name}:")
        print(f"  Pairs: {len(zone_df)}")
        print(f"  AUC \hat{{d}}_OGH:  {auc_ogh:.3f}")
        print(f"  AUC TM-score:     {auc_tm:.3f}")
        print(f"  AUC Combined:     {auc_comb:.3f}")
        print(f"  AUC Length:       {auc_len:.3f} (control)")

        zone_results.append({
            'zone': zone_name,
            'n_pairs': len(zone_df),
            'auc_ogh': auc_ogh,
            'auc_tm': auc_tm,
            'auc_combined': auc_comb,
            'auc_len': auc_len
        })

        # Plot ROC curves for each zone
        ax = axes[idx]
        ax.plot(fpr_ogh, tpr_ogh, 'r-', linewidth=2.5, label=f'OGH (AUC={auc_ogh:.3f})')
        ax.plot(fpr_tm, tpr_tm, 'b-', linewidth=2.5, label=f'TM-score (AUC={auc_tm:.3f})')
        ax.plot(fpr_comb, tpr_comb, 'm-', linewidth=2.5, label=f'Combined (AUC={auc_comb:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{zone_name}\n(n={len(zone_df)} pairs)')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('roc_zone_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Zone ROC plot saved: roc_zone_analysis.png")
    plt.show()

    return pd.DataFrame(zone_results)


# ============================================================================
# 6. AUC ANALYSIS
# ============================================================================

def run_auc_analysis(df_metrics, suffix=''):
    print("\n" + "=" * 80)
    print("📊 AUC ANALYSIS: \hat{d}_OGH vs TM-score")
    print("=" * 80)

    y_true = df_metrics['is_homolog'].values

    fpr_ogh, tpr_ogh, _ = roc_curve(y_true, -df_metrics['d_hat_OGH'].values)
    auc_ogh = auc(fpr_ogh, tpr_ogh)

    fpr_tm, tpr_tm, _ = roc_curve(y_true, df_metrics['TM_score'].values)
    auc_tm = auc(fpr_tm, tpr_tm)

    ogh_norm = 1.0 - (df_metrics['d_hat_OGH'] - df_metrics['d_hat_OGH'].min()) / \
                     (df_metrics['d_hat_OGH'].max() - df_metrics['d_hat_OGH'].min() + 1e-8)
    tm_norm = (df_metrics['TM_score'] - df_metrics['TM_score'].min()) / \
              (df_metrics['TM_score'].max() - df_metrics['TM_score'].min() + 1e-8)
    combined = 0.5 * ogh_norm + 0.5 * tm_norm
    fpr_comb, tpr_comb, _ = roc_curve(y_true, combined.values)
    auc_comb = auc(fpr_comb, tpr_comb)

    length_ratio = df_metrics['len1'] / (df_metrics['len2'] + 1)
    fpr_len, tpr_len, _ = roc_curve(y_true, length_ratio.values)
    auc_len = auc(fpr_len, tpr_len)

    print(f"\n📈 AUC RESULTS:")
    print(f"  \hat{{d}}_OGH:  {auc_ogh:.3f}")
    print(f"  TM-score:     {auc_tm:.3f}")
    print(f"  Combined:     {auc_comb:.3f}")
    print(f"  Length:       {auc_len:.3f} (control)")

    plt.figure(figsize=(10, 8))
    plt.plot(fpr_ogh, tpr_ogh, 'r-', linewidth=3, label=f'OGH (AUC={auc_ogh:.3f})')
    plt.plot(fpr_tm, tpr_tm, 'b-', linewidth=2, label=f'TM-score (AUC={auc_tm:.3f})')
    plt.plot(fpr_comb, tpr_comb, 'm-', linewidth=2, label=f'Combined (AUC={auc_comb:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    fname = f'homology_auc_real_ogh{suffix}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved: {fname}")
    plt.show()

    return {'auc_ogh': auc_ogh, 'auc_tm': auc_tm, 'auc_combined': auc_comb, 'auc_len': auc_len}


# ============================================================================
# 7. SENSITIVITY ANALYSIS
# ============================================================================

def run_sensitivity_analysis(df_pairs):
    print("\n" + "=" * 80)
    print("🔬 SENSITIVITY ANALYSIS")
    print("=" * 80)

    records = []

    for w in WINDOW_SIZES:
        for g in GAP_PENALTIES:
            df_res = compute_metrics(df_pairs, alpha=0.5, window_size=w, gap_penalty=g)
            y_true = df_res['is_homolog'].values
            fpr, tpr, _ = roc_curve(y_true, -df_res['d_hat_OGH'].values)
            a = auc(fpr, tpr)
            records.append({'window_size': w, 'gap_penalty': g, 'alpha': 0.5, 'AUC': a})
            print(f"  w={w:2d}, γ={g:.1f} → AUC = {a:.3f}")

    for alpha in ALPHAS:
        df_res = compute_metrics(df_pairs, alpha=alpha, window_size=15, gap_penalty=2.0)
        y_true = df_res['is_homolog'].values
        fpr, tpr, _ = roc_curve(y_true, -df_res['d_hat_OGH'].values)
        a = auc(fpr, tpr)
        records.append({'window_size': 15, 'gap_penalty': 2.0, 'alpha': alpha, 'AUC': a})
        print(f"  α={alpha:.2f} → AUC = {a:.3f}")

    df_sens = pd.DataFrame(records)
    df_sens.to_csv('sensitivity_analysis.csv', index=False)
    print("\n📋 sensitivity_analysis.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df_wg = df_sens[df_sens['alpha'] == 0.5].drop_duplicates(subset=['window_size', 'gap_penalty'])
    pivot = df_wg.pivot(index='gap_penalty', columns='window_size', values='AUC')

    im = axes[0].imshow(pivot.values, cmap='viridis', aspect='auto')
    axes[0].set_xticks(range(len(pivot.columns)))
    axes[0].set_xticklabels(pivot.columns)
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_yticklabels([f'{x:.1f}' for x in pivot.index])
    axes[0].set_xlabel('Window size (w)')
    axes[0].set_ylabel('Gap penalty (γ)')
    axes[0].set_title('AUC: w vs γ')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            axes[0].text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", color="w")
    plt.colorbar(im, ax=axes[0])

    alpha_data = df_sens[(df_sens['window_size'] == 15) & (df_sens['gap_penalty'] == 2.0)]
    axes[1].plot(alpha_data['alpha'], alpha_data['AUC'], 'o-', linewidth=2, markersize=8)
    axes[1].set_xlabel('α (order weight)')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Sensitivity to α (w=15, γ=2.0)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 sensitivity_analysis.png")
    plt.show()

    return df_sens


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("🔬 VALIDATION OF \hat{d}_OGH FOR HOMOLOGY DETECTION")
    print("=" * 80)

    print("\n1. Loading homologs_database_clean.csv...")
    try:
        df = pd.read_csv('homologs_database_clean.csv')
        print(f"   Loaded {len(df)} records")
    except FileNotFoundError:
        print("   ❌ File not found!")
        return

    print("\n2. Building pairs...")
    df_pairs = build_pairs_with_pdb(df)

    if len(df_pairs) == 0:
        print("❌ No pairs with PDB")
        return

    print("\n3. Main experiment...")
    df_metrics = compute_metrics(df_pairs, alpha=0.5, window_size=15, gap_penalty=2.0)
    df_metrics.to_csv('homology_real_ogh_results.csv', index=False)
    print(f"\n✅ Results saved: homology_real_ogh_results.csv")

    # 4. Global AUC analysis
    results = run_auc_analysis(df_metrics, suffix='_main')

    # 5. Zone-specific analysis (Diverse, Twilight, Homologous)
    print("\n5. Zone-specific analysis...")
    zone_results = run_zone_analysis(df_metrics)
    zone_results.to_csv('zone_auc_results.csv', index=False)
    print(f"\n✅ Zone results saved: zone_auc_results.csv")



    # 6. Sensitivity analysis
    #print("\n6. Sensitivity analysis...")
    #df_sens = run_sensitivity_analysis(df_pairs)

    # 7. Summary
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    print(f"   AUC \hat{{d}}_OGH (global):      {results['auc_ogh']:.3f}")
    print(f"   AUC TM-score (global):         {results['auc_tm']:.3f}")
    print(f"   AUC Combined (global):         {results['auc_combined']:.3f}")
    print(f"   Improvement:                   {results['auc_combined'] - max(results['auc_ogh'], results['auc_tm']):.3f}")

    if not zone_results.empty:
        print(f"\n   ZONE-SPECIFIC RESULTS:")
        for _, row in zone_results.iterrows():
            print(f"     {row['zone']}: OGH={row['auc_ogh']:.3f}, TM={row['auc_tm']:.3f}, Combined={row['auc_combined']:.3f}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()