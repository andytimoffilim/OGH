#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
order_sensitivity_vs_C.py

Empirical test of the "degeneracy at small C" thesis for the OGH
saturation constant C in w_ord(i,j) = C * (1 - exp(-sigma * |i/n - j/m|)).

For C in {2, 5, 10, 20, 50}:
  1) Shuffling test on 5 proteins (d1faki_, d1i39a_, d1di1a_, d1edg__, d1b1ya_):
     OGH(native, shuffled) at 0/10/30/50/70/100% shuffling.
  2) Circular permutation test on d1faki_ (full chain, 597 residues):
     OGH(native, permuted) at cut 0.0/0.1/.../0.9.
  3) Relative contribution of the order term to the final OGH score
     on the SCOP/ASTRAL benchmark (1400 pairs):
         frac_order = alpha * mean(w_ord) /
                      [ (1-alpha)*mean(d_geom) + alpha*mean(w_ord) ]

Outputs:
  - order_sensitivity_vs_C.csv
  - order_sensitivity_vs_C.png
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Reuse infrastructure from the existing pipeline
from parameter_sensitivity_analysis_new import (
    OrderedGromovHausdorff,
    download_pdb,
    load_pdb_coords,
    load_pairs_with_coords,
)


# ============================================================================
# 1. OGH WITH PARAMETRIZED SATURATION CONSTANT + ORDER-TERM DIAGNOSTICS
# ============================================================================

class OGHWithSaturation(OrderedGromovHausdorff):
    """Same as OrderedGromovHausdorff, but C in w_ord is a parameter.

    Additionally returns the relative contribution of the order term
    to the final score, which is the quantity we use to test degeneracy.
    """

    def __init__(self, saturation_constant=20.0, **kwargs):
        super().__init__(**kwargs)
        self.saturation_constant = saturation_constant

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

        # <-- the only change: C instead of hardcoded 20.0
        w_ord = self.saturation_constant * (
            1.0 - np.exp(-self.align_sigma * np.abs(i_norm - j_norm))
        )

        cost_align = d_geom + self.align_order_weight * w_ord
        alignment = self._greedy_monotone_alignment(cost_align)

        if len(alignment) == 0:
            return np.inf, {}

        align_i, align_j = zip(*alignment)
        matched_geom = d_geom[align_i, align_j]
        matched_ord = w_ord[align_i, align_j]
        k = len(alignment)

        metric_dist = float(np.mean(matched_geom))
        order_dist = float(np.mean(matched_ord))

        ogh_dist = ((1.0 - self.alpha) * metric_dist
                    + self.alpha * order_dist) / np.sqrt(L)

        # Relative contribution of the order term to the weighted sum
        denom = (1.0 - self.alpha) * metric_dist + self.alpha * order_dist
        frac_order = (self.alpha * order_dist / denom) if denom > 0 else np.nan

        return ogh_dist, {
            'metric_component': metric_dist,
            'order_component': order_dist,
            'frac_order': frac_order,
            'n_matches': k,
            'coverage': k / max(n, m),
            'alpha': self.alpha,
            'saturation_constant': self.saturation_constant,
            'len1': n,
            'len2': m,
        }


# ============================================================================
# 2. SHUFFLING HELPERS
# ============================================================================

def shuffle_coords(coords, fraction, rng):
    """Randomly permute `fraction` of residues (keeping the rest fixed)."""
    n = len(coords)
    k = int(round(fraction * n))
    if k == 0:
        return coords.copy()
    idx = rng.choice(n, size=k, replace=False)
    permuted = coords.copy()
    permuted[idx] = coords[rng.permutation(idx)]
    return permuted


def load_five_proteins(pdb_ids=('d1faki_', 'd1i39a_', 'd1di1a_',
                                'd1edg__', 'd1b1ya_'),
                       pdb_dir='pdb_scop'):
    """Download and load the 5 proteins used in Table 1 of OGH.tex."""
    out = {}
    for pid in pdb_ids:
        f = download_pdb(pid, pdb_dir)
        if f is None:
            print(f"  WARNING: could not download {pid}")
            continue
        coords = load_pdb_coords(f)
        if coords is None or len(coords) < 10:
            print(f"  WARNING: could not load {pid}")
            continue
        out[pid] = coords
        print(f"  loaded {pid}: {len(coords)} residues")
    return out


# ============================================================================
# 3. TEST 1: SHUFFLING vs C
# ============================================================================

def run_shuffling_vs_C(proteins, C_grid, fractions=(0.0, 0.1, 0.3, 0.5, 0.7, 1.0),
                       n_repeats=5, base_kwargs=None, seed=42):
    """For each C, for each protein, average OGH(native, shuffled) over repeats."""
    if base_kwargs is None:
        base_kwargs = dict(alpha=0.5, align_sigma=3.0,
                           align_order_weight=1.0,
                           window_size=15, gap_penalty=2.0,
                           use_normalization=True)

    rows = []
    rng = np.random.default_rng(seed)

    for C in C_grid:
        ogh = OGHWithSaturation(saturation_constant=C, **base_kwargs)
        for pid, coords in proteins.items():
            for frac in fractions:
                vals = []
                for _ in range(n_repeats if frac > 0 else 1):
                    shuf = shuffle_coords(coords, frac, rng)
                    d, _ = ogh.compute(coords, shuf)
                    vals.append(d)
                rows.append({
                    'C': C,
                    'protein': pid,
                    'length': len(coords),
                    'shuffle_fraction': frac,
                    'OGH_mean': float(np.mean(vals)),
                    'OGH_std': float(np.std(vals)),
                    'n_repeats': len(vals),
                })
        print(f"  C={C:>4}: shuffling done for {len(proteins)} proteins")

    return pd.DataFrame(rows)


# ============================================================================
# 4. TEST 2: CIRCULAR PERMUTATION vs C
# ============================================================================

def circular_permute(coords, cut_fraction):
    """Circular permutation: swap N- and C-terminal fragments."""
    n = len(coords)
    cut = int(round(cut_fraction * n)) % n
    if cut == 0:
        return coords.copy()
    return np.vstack([coords[cut:], coords[:cut]])


def run_circular_permutation_vs_C(protein_coords, C_grid,
                                  cuts=(0.0, 0.1, 0.2, 0.3, 0.4,
                                        0.5, 0.6, 0.7, 0.8, 0.9),
                                  base_kwargs=None):
    """For each C, compute OGH(native, circularly permuted) across cuts."""
    if base_kwargs is None:
        base_kwargs = dict(alpha=0.5, align_sigma=3.0,
                           align_order_weight=1.0,
                           window_size=15, gap_penalty=2.0,
                           use_normalization=True)

    rows = []
    for C in C_grid:
        ogh = OGHWithSaturation(saturation_constant=C, **base_kwargs)
        for cut in cuts:
            perm = circular_permute(protein_coords, cut)
            d, _ = ogh.compute(protein_coords, perm)
            rows.append({
                'C': C,
                'cut_fraction': cut,
                'OGH': d,
            })
        print(f"  C={C:>4}: circular permutation done")
    return pd.DataFrame(rows)


# ============================================================================
# 5. TEST 3: RELATIVE ORDER-TERM CONTRIBUTION vs C
# ============================================================================

def run_frac_order_vs_C(df_pairs, C_grid, base_kwargs=None):
    """For each C, average frac_order over all SCOP/ASTRAL pairs."""
    if base_kwargs is None:
        base_kwargs = dict(alpha=0.5, align_sigma=3.0,
                           align_order_weight=1.0,
                           window_size=15, gap_penalty=2.0,
                           use_normalization=True)

    rows = []
    for C in C_grid:
        ogh = OGHWithSaturation(saturation_constant=C, **base_kwargs)
        fracs, geoms, ords = [], [], []
        for row in df_pairs.itertuples(index=False):
            d, info = ogh.compute(row.coords1, row.coords2)
            if not info:
                continue
            fracs.append(info['frac_order'])
            geoms.append(info['metric_component'])
            ords.append(info['order_component'])
        rows.append({
            'C': C,
            'frac_order_mean': float(np.nanmean(fracs)),
            'frac_order_median': float(np.nanmedian(fracs)),
            'frac_order_p10': float(np.nanpercentile(fracs, 10)),
            'frac_order_p90': float(np.nanpercentile(fracs, 90)),
            'metric_component_mean': float(np.nanmean(geoms)),
            'order_component_mean': float(np.nanmean(ords)),
            'n_pairs': len(fracs),
        })
        print(f"  C={C:>4}: frac_order mean = {rows[-1]['frac_order_mean']:.4f}")
    return pd.DataFrame(rows)


# ============================================================================
# 6. PLOTTING
# ============================================================================

def plot_all(df_shuf, df_circ, df_frac, out_png='order_sensitivity_vs_C.png'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: shuffling — OGH at 100% shuffling vs C, per protein
    ax = axes[0]
    df_full = df_shuf[df_shuf['shuffle_fraction'] == 1.0]
    for pid, g in df_full.groupby('protein'):
        g = g.sort_values('C')
        ax.plot(g['C'], g['OGH_mean'], 'o-', label=f"{pid} (n={g['length'].iloc[0]})")
    ax.set_xscale('log')
    ax.set_xlabel('Saturation constant C')
    ax.set_ylabel('OGH at 100% shuffling')
    ax.set_title('A. Order sensitivity to shuffling')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ax.axvline(20, color='green', ls=':', lw=1.5)

    # Panel B: circular permutation — OGH at cut=0.7 vs C
    ax = axes[1]
    df_c = df_circ[df_circ['cut_fraction'] == 0.7].sort_values('C')
    ax.plot(df_c['C'], df_c['OGH'], 'o-', color='tab:red',
            label='d1faki_ (cut=0.7)')
    df_c0 = df_circ[df_circ['cut_fraction'] == 0.0].sort_values('C')
    ax.plot(df_c0['C'], df_c0['OGH'], 's--', color='gray',
            label='cut=0 (native)')
    ax.set_xscale('log')
    ax.set_xlabel('Saturation constant C')
    ax.set_ylabel('OGH')
    ax.set_title('B. Order sensitivity to circular permutation')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.axvline(20, color='green', ls=':', lw=1.5)

    # Panel C: relative order-term contribution vs C
    ax = axes[2]
    ax.plot(df_frac['C'], df_frac['frac_order_mean'],
            'o-', color='tab:purple', label='mean frac_order')
    ax.fill_between(df_frac['C'],
                    df_frac['frac_order_p10'],
                    df_frac['frac_order_p90'],
                    color='tab:purple', alpha=0.2, label='10–90%')
    ax.set_xscale('log')
    ax.set_xlabel('Saturation constant C')
    ax.set_ylabel('Order-term share of OGH score')
    ax.set_title('C. Relative contribution of the order term')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.axvline(20, color='green', ls=':', lw=1.5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"  Saved: {out_png}")
    plt.close(fig)


# ============================================================================
# 7. MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("ORDER-SENSITIVITY vs SATURATION CONSTANT C")
    print("=" * 80)

    C_grid = (2, 5, 10, 20, 50)

    # ---- Load 5 proteins for shuffling / circular permutation ----
    print("\n[1/3] Loading 5 proteins for shuffling/circular permutation...")
    proteins = load_five_proteins()
    if not proteins:
        print("ERROR: no proteins loaded.")
        return

    print("\n[2/3] Shuffling test vs C ...")
    df_shuf = run_shuffling_vs_C(proteins, C_grid)

    print("\n[3/3] Circular permutation test vs C (d1faki_) ...")
    if 'd1faki_' in proteins:
        df_circ = run_circular_permutation_vs_C(proteins['d1faki_'], C_grid)
    else:
        df_circ = pd.DataFrame()

    print("\n[4/4] Relative order-term contribution on SCOP/ASTRAL ...")
    df_pairs = load_pairs_with_coords()
    if len(df_pairs) == 0:
        print("WARNING: no pairs with coords; skipping frac_order analysis.")
        df_frac = pd.DataFrame()
    else:
        df_frac = run_frac_order_vs_C(df_pairs, C_grid)

    # ---- Save ----
    df_shuf.to_csv('order_sensitivity_shuffling_vs_C.csv', index=False)
    if not df_circ.empty:
        df_circ.to_csv('order_sensitivity_circular_vs_C.csv', index=False)
    if not df_frac.empty:
        df_frac.to_csv('order_sensitivity_frac_vs_C.csv', index=False)

    # Combined tidy table
    combined = []
    for C in C_grid:
        row = {'C': C}
        s = df_shuf[(df_shuf['C'] == C) & (df_shuf['shuffle_fraction'] == 1.0)]
        if not s.empty:
            row['shuffling_OGH_100pct_mean'] = s['OGH_mean'].mean()
        if not df_circ.empty:
            c7 = df_circ[(df_circ['C'] == C) & (df_circ['cut_fraction'] == 0.7)]
            if not c7.empty:
                row['circular_perm_OGH_cut07'] = float(c7['OGH'].iloc[0])
        if not df_frac.empty:
            f = df_frac[df_frac['C'] == C]
            if not f.empty:
                row['frac_order_mean'] = float(f['frac_order_mean'].iloc[0])
                row['frac_order_median'] = float(f['frac_order_median'].iloc[0])
        combined.append(row)
    df_combined = pd.DataFrame(combined)
    df_combined.to_csv('order_sensitivity_vs_C.csv', index=False)
    print("\nSaved: order_sensitivity_vs_C.csv")
    print("\nCombined summary:")
    print(df_combined.to_string(index=False))

    # ---- Plot ----
    if not df_circ.empty and not df_frac.empty:
        plot_all(df_shuf, df_circ, df_frac)

    # ---- Headline diagnostics ----
    print("\n" + "=" * 80)
    print("HEADLINE DIAGNOSTICS (degeneracy test)")
    print("=" * 80)
    if 'shuffling_OGH_100pct_mean' in df_combined.columns:
        base = df_combined[df_combined['C'] == 20].iloc[0]
        for _, r in df_combined.iterrows():
            if r['C'] == 20:
                continue
            print(f"  C={r['C']:>4}: "
                  f"dOGH_shuf={r['shuffling_OGH_100pct_mean'] - base['shuffling_OGH_100pct_mean']:+.4f}  "
                  f"dOGH_circ={r.get('circular_perm_OGH_cut07', np.nan) - base.get('circular_perm_OGH_cut07', np.nan):+.4f}  "
                  f"dfrac_order={r.get('frac_order_mean', np.nan) - base.get('frac_order_mean', np.nan):+.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()