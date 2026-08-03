"""
generate_figures.py - Generate publication-ready figures for OGH article
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
import warnings
import os

warnings.filterwarnings('ignore')

# Publication-ready settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def compute_ogh_demo(X, Y, alpha=0.5):
    """
    Упрощённая OGH для демонстрации.
    """
    n, m = len(X), len(Y)

    X_norm = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    Y_norm = (Y - np.mean(Y, axis=0)) / (np.std(Y, axis=0) + 1e-8)

    dist_geom = np.linalg.norm(X_norm[:, None, :] - Y_norm[None, :, :], axis=2)

    order_penalty = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            pos_diff = abs(i / n - j / m)
            order_penalty[i, j] = 20 * (1 - np.exp(-3 * pos_diff))

    total_cost = (1 - alpha) * dist_geom + alpha * order_penalty

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
                    cost = total_cost[i + di, j + dj] + (di + dj) * 2
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

    matched_dist = [dist_geom[i, j] for i, j in alignment]
    ogh_dist = np.mean(matched_dist) / np.sqrt(min(n, m))
    return ogh_dist


def fig1_ogh_concept():
    """
    Figure 1: OGH concept — ordered vs shuffled protein.
    Panels A-B: 3D visualisation.
    Panel C: GH vs OGH sensitivity to shuffling.
    """
    fig = plt.figure(figsize=(16, 7))

    # Create helix (20 points)
    t = np.linspace(0, 4 * np.pi, 20)
    helix = np.column_stack([np.cos(t), np.sin(t), t / 2])

    # Shuffled version
    np.random.seed(42)
    shuffled_idx = np.random.permutation(len(helix))
    helix_shuffled = helix[shuffled_idx]

    colors_orig = plt.cm.viridis(np.linspace(0, 1, len(helix)))
    colors_shuf = plt.cm.viridis(np.linspace(0, 1, len(helix_shuffled)))

    # ============================================================
    # Panel A: Original protein
    # ============================================================
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    for i, (x, y, z) in enumerate(helix):
        ax1.scatter(x, y, z, c=[[colors_orig[i]]], s=80, edgecolors='black', linewidth=0.5)
    ax1.plot(helix[:, 0], helix[:, 1], helix[:, 2], 'gray', linewidth=1.5, alpha=0.7)
    ax1.set_title('A: Ordered protein', fontweight='bold', fontsize=14)
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.set_zlabel('Z', fontsize=10)
    ax1.view_init(elev=25, azim=45)

    # ============================================================
    # Panel B: Shuffled protein
    # ============================================================
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    for i, (x, y, z) in enumerate(helix_shuffled):
        ax2.scatter(x, y, z, c=[[colors_shuf[i]]], s=80, edgecolors='black', linewidth=0.5)
    ax2.plot(helix_shuffled[:, 0], helix_shuffled[:, 1], helix_shuffled[:, 2],
             'gray', linewidth=1.5, alpha=0.7)
    ax2.set_title('B: Shuffled protein', fontweight='bold', fontsize=14)
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Y', fontsize=10)
    ax2.set_zlabel('Z', fontsize=10)
    ax2.view_init(elev=25, azim=45)

    # ============================================================
    # Panel C: GH vs OGH sensitivity to shuffling
    # ============================================================
    ax3 = fig.add_subplot(1, 3, 3)

    # Generate data for GH vs OGH
    shuffling_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    gh_values = []
    ogh_values = []

    np.random.seed(42)

    for level in shuffling_levels:
        if level == 0:
            shuffled = helix
        else:
            n_shuffle = int(len(helix) * level / 100)
            idx = np.arange(len(helix))
            shuffle_idx = np.random.permutation(n_shuffle)
            idx[:n_shuffle] = idx[shuffle_idx]
            shuffled = helix[idx]

        # GH
        triu_orig = pdist(helix)
        triu_shuf = pdist(shuffled)
        hist_orig, bins = np.histogram(triu_orig, bins=50, density=True)
        hist_shuf, _ = np.histogram(triu_shuf, bins=bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        gh_dist = wasserstein_distance(bin_centers, bin_centers, hist_orig, hist_shuf)
        gh_values.append(gh_dist)

        # OGH
        ogh_dist = compute_ogh_demo(helix, shuffled, alpha=0.5)
        ogh_values.append(ogh_dist)

    ax3.plot(shuffling_levels, gh_values, 'ro-', linewidth=2.5, markersize=8,
             label='GH (order-blind)')
    ax3.plot(shuffling_levels, ogh_values, 'bs-', linewidth=2.5, markersize=8,
             label='OGH (order-sensitive)')

    ax3.set_xlabel('Shuffling degree (%)', fontsize=12)
    ax3.set_ylabel('Distance', fontsize=12)
    ax3.set_title('C: GH vs OGH sensitivity to order', fontweight='bold', fontsize=14)
    ax3.legend(loc='upper left', framealpha=0.8, fontsize=10)
    ax3.grid(True, alpha=0.2)
    ax3.set_xlim(-2, 102)
    ax3.set_ylim(-0.02, 0.35)

    # Annotations
    ax3.text(50, 0.30, 'GH = 0 (always)', ha='center', fontsize=10, color='red')
    ax3.text(50, 0.12, 'OGH increases\nwith shuffling', ha='center', fontsize=10, color='blue')

    plt.tight_layout()
    plt.savefig('fig1_ogh_concept.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ fig1_ogh_concept.png")


def fig2_ogh_vs_tm_scatter():
    """Figure 2: Scatter plot OGH vs TM-score with stratification."""
    try:
        import pandas as pd
        if os.path.exists('ogh_large_scale_results.csv'):
            df = pd.read_csv('ogh_large_scale_results.csv')
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['OGH', 'TM_score'])
            print("✓ Loaded real data from ogh_large_scale_results.csv")
        else:
            raise FileNotFoundError
    except (FileNotFoundError, NameError):
        print("⚠️ ogh_large_scale_results.csv not found. Using synthetic data.")
        np.random.seed(42)
        n = 1500
        tm_diverse = np.random.uniform(0.05, 0.30, 700)
        tm_twilight = np.random.uniform(0.30, 0.50, 550)
        tm_homolog = np.random.uniform(0.50, 0.95, 250)

        ogh_diverse = 0.05 + 0.35 * tm_diverse + np.random.normal(0, 0.05, len(tm_diverse))
        ogh_twilight = 0.15 + np.random.normal(0, 0.05, len(tm_twilight))
        ogh_homolog = 0.40 - 0.35 * tm_homolog + np.random.normal(0, 0.04, len(tm_homolog))

        tm = np.concatenate([tm_diverse, tm_twilight, tm_homolog])
        ogh = np.concatenate([ogh_diverse, ogh_twilight, ogh_homolog])
        groups = ['Diverse'] * 700 + ['Twilight'] * 550 + ['Homologous'] * 250
        df = pd.DataFrame({'TM_score': tm, 'OGH': ogh, 'group': groups})

    df['group'] = pd.cut(
        df['TM_score'],
        bins=[-np.inf, 0.3, 0.5, 1.0],
        labels=['Diverse (TM ≤ 0.3)', 'Twilight zone', 'Homologous (TM > 0.5)']
    )

    colors = {
        'Diverse (TM ≤ 0.3)': '#d62728',
        'Twilight zone': '#7f7f7f',
        'Homologous (TM > 0.5)': '#1f77b4'
    }

    fig, ax = plt.subplots(figsize=(11, 8))

    for group, color in colors.items():
        subset = df[df['group'] == group]
        ax.scatter(subset['TM_score'], subset['OGH'],
                   c=color, alpha=0.25, s=20, label=group, edgecolors='none')

    ax.axvline(x=0.3, color='black', linestyle='--', alpha=0.6, linewidth=1.2)
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.6, linewidth=1.2)

    ax.set_xlabel('TM-score', fontsize=14)
    ax.set_ylabel('OGH distance', fontsize=14)
    ax.set_title('OGH vs TM-score: stratified by structural similarity', fontweight='bold', fontsize=14)
    ax.legend(loc='upper left', framealpha=0.8, fontsize=11)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig('fig2_ogh_vs_tm_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ fig2_ogh_vs_tm_scatter.png")


def fig3_ogh_shuffling():
    """Figure 3: OGH sensitivity to shuffling degree (from Table 1)."""
    data = {
        '11BC': {'length': 500, 'x': [0, 10, 30, 50, 70, 100],
                 'y': [0.0000, 0.0201, 0.0465, 0.0484, 0.0548, 0.0608]},
        '1A1N': {'length': 383, 'x': [0, 10, 30, 50, 70, 100],
                 'y': [0.0000, 0.0096, 0.0390, 0.0540, 0.0676, 0.0785]},
        '1A1O': {'length': 384, 'x': [0, 10, 30, 50, 70, 100],
                 'y': [0.0000, 0.0038, 0.0222, 0.0381, 0.0728, 0.0781]},
        '1A1T': {'length': 55, 'x': [0, 10, 30, 50, 70, 100],
                 'y': [0.0000, 0.0065, 0.0314, 0.0544, 0.0882, 0.2044]},
        '1A30': {'length': 201, 'x': [0, 10, 30, 50, 70, 100],
                 'y': [0.0000, 0.0042, 0.0589, 0.0728, 0.0719, 0.0969]},
    }

    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, ax = plt.subplots(figsize=(9, 6))

    for (name, d), marker, color in zip(data.items(), markers, colors):
        ax.plot(d['x'], d['y'], marker=marker, color=color, linewidth=2.5,
                markersize=8, label=f'{name} ({d["length"]} res)')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='GH = 0 (all)')

    ax.set_xlabel('Shuffling degree (%)', fontsize=14)
    ax.set_ylabel('OGH distance', fontsize=14)
    ax.set_title('Sensitivity of OGH to residue shuffling', fontweight='bold', fontsize=14)
    ax.legend(loc='upper left', framealpha=0.8, fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.02, 0.25)

    plt.tight_layout()
    plt.savefig('fig3_ogh_shuffling.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ fig3_ogh_shuffling.png")


def main():
    print("=" * 60)
    print("Generating publication figures for OGH article")
    print("=" * 60)
    print()

    fig1_ogh_concept()
    print()
    fig2_ogh_vs_tm_scatter()
    print()
    fig3_ogh_shuffling()

    print("\n" + "=" * 60)
    print("✅ Done! Figures saved:")
    print("  📁 fig1_ogh_concept.png")
    print("  📁 fig2_ogh_vs_tm_scatter.png")
    print("  📁 fig3_ogh_shuffling.png")
    print("=" * 60)


if __name__ == "__main__":
    main()