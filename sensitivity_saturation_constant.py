#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sensitivity_saturation_constant.py

Sensitivity analysis of the saturation constant C = 20 in the OGH order
penalty w_ord(i,j) = C * (1 - exp(-sigma * |i/n - j/m|)).

Reuses the OrderedGromovHausdorff class and data loading from
parameter_sensitivity_analysis_new.py, but exposes C as a parameter.

Outputs:
  - sensitivity_saturation_constant.csv   (per-C: AUC overall + by TM zone
                                            + correlations with TM-score)
  - sensitivity_saturation_constant.png   (AUC vs C, overall + by zone)
  - prints a summary of stability (min/max/range of AUC across C)
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Reuse the existing infrastructure
from parameter_sensitivity_analysis_new import (
    OrderedGromovHausdorff,
    load_pairs_with_coords,
)


# ============================================================================
# 1. OGH WITH PARAMETRIZED SATURATION CONSTANT
# ============================================================================

class OGHWithSaturation(OrderedGromovHausdorff):
    """Same as OrderedGromovHausdorff, but C in w_ord is a parameter."""

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

        metric_dist = np.mean(matched_geom)
        order_dist = np.mean(matched_ord)

        ogh_dist = ((1.0 - self.alpha) * metric_dist
                    + self.alpha * order_dist) / np.sqrt(L)

        return ogh_dist, {
            'metric_component': metric_dist,
            'order_component': order_dist,
            'n_matches': k,
            'coverage': k / max(n, m),
            'alpha': self.alpha,
            'saturation_constant': self.saturation_constant,
            'len1': n,
            'len2': m,
        }


# ============================================================================
# 2. HELPERS
# ============================================================================

def tm_zone(tm):
    if tm <= 0.3:
        return 'diverse'
    elif tm <= 0.5:
        return 'twilight'
    else:
        return 'homologous'


def safe_auc(y_true, scores):
    """AUC for homology detection; higher score = more homologous."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, scores)
    return auc(fpr, tpr)


def compute_ogh_vector(df_pairs, C, base_kwargs):
    """Compute OGH for every pair at a given saturation constant C."""
    ogh = OGHWithSaturation(saturation_constant=C, **base_kwargs)
    vals = np.empty(len(df_pairs), dtype=float)
    for idx, row in enumerate(df_pairs.itertuples(index=False)):
        d, _ = ogh.compute(row.coords1, row.coords2)
        vals[idx] = d
    return vals


# ============================================================================
# 3. MAIN ANALYSIS
# ============================================================================

def run_saturation_analysis(df_pairs,
                            C_grid=(2, 5, 10, 15, 20, 30, 40, 50, 100),
                            base_kwargs=None):
    """
    For each C in C_grid:
      - compute OGH for all pairs
      - overall AUC (homology detection, OGH inverted: lower OGH -> homolog)
      - AUC within each TM-score zone
      - Pearson/Spearman/Kendall correlation between OGH and TM-score
        (overall and within zones)
    Returns a tidy DataFrame.
    """
    if base_kwargs is None:
        base_kwargs = dict(alpha=0.5, align_sigma=3.0,
                           align_order_weight=1.0,
                           window_size=15, gap_penalty=2.0,
                           use_normalization=True)

    y_true = df_pairs['is_homolog'].values
    tm = df_pairs['tm_score'].values if 'tm_score' in df_pairs.columns \
         else df_pairs['TM_score'].values
    zones = np.array([tm_zone(t) for t in tm])

    rows = []
    for C in C_grid:
        ogh_vals = compute_ogh_vector(df_pairs, C, base_kwargs)
        # For homology detection, lower OGH = more homologous -> use -OGH
        auc_overall = safe_auc(y_true, -ogh_vals)

        # Correlations with TM-score (expect negative: higher TM -> lower OGH)
        r_p, p_p = pearsonr(ogh_vals, tm)
        r_s, p_s = spearmanr(ogh_vals, tm)
        r_k, p_k = kendalltau(ogh_vals, tm)

        row = {
            'C': C,
            'AUC_overall': auc_overall,
            'pearson_r': r_p, 'pearson_p': p_p,
            'spearman_rho': r_s, 'spearman_p': p_s,
            'kendall_tau': r_k, 'kendall_p': p_k,
        }

        # Per-zone AUC and correlations
        for zone in ('diverse', 'twilight', 'homologous'):
            mask = zones == zone
            if mask.sum() == 0:
                continue
            y_z = y_true[mask]
            ogh_z = ogh_vals[mask]
            tm_z = tm[mask]
            row[f'AUC_{zone}'] = safe_auc(y_z, -ogh_z)
            if len(np.unique(ogh_z)) > 1 and len(np.unique(tm_z)) > 1:
                row[f'pearson_r_{zone}'] = pearsonr(ogh_z, tm_z)[0]
                row[f'spearman_rho_{zone}'] = spearmanr(ogh_z, tm_z)[0]
            else:
                row[f'pearson_r_{zone}'] = np.nan
                row[f'spearman_rho_{zone}'] = np.nan

        rows.append(row)
        print(f"  C={C:>4}: AUC_overall={auc_overall:.4f}  "
              f"r(Pearson)={r_p:+.4f}  rho(Spearman)={r_s:+.4f}")

    return pd.DataFrame(rows)


# ============================================================================
# 4. PLOTTING
# ============================================================================

def plot_saturation(df_res, out_png='sensitivity_saturation_constant.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: AUC vs C (overall + by zone)
    ax = axes[0]
    ax.plot(df_res['C'], df_res['AUC_overall'],
            'o-', lw=2, ms=7, color='black', label='Overall')
    for zone, color in (('diverse', 'tab:red'),
                        ('twilight', 'tab:gray'),
                        ('homologous', 'tab:blue')):
        col = f'AUC_{zone}'
        if col in df_res.columns:
            ax.plot(df_res['C'], df_res[col],
                    's--', lw=1.5, ms=5, color=color,
                    label=f'{zone} (TM zone)')
    ax.axvline(20, color='green', ls=':', lw=1.5, label='C=20 (default)')
    ax.set_xscale('log')
    ax.set_xlabel('Saturation constant C')
    ax.set_ylabel('AUC (homology detection)')
    ax.set_title('AUC vs C')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel B: correlation with TM-score vs C
    ax = axes[1]
    ax.plot(df_res['C'], df_res['pearson_r'],
            'o-', lw=2, ms=7, color='tab:purple', label='Pearson r')
    ax.plot(df_res['C'], df_res['spearman_rho'],
            's--', lw=1.5, ms=5, color='tab:orange', label='Spearman ρ')
    ax.axvline(20, color='green', ls=':', lw=1.5, label='C=20 (default)')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('Saturation constant C')
    ax.set_ylabel('Correlation with TM-score')
    ax.set_title('Correlation(OGH, TM-score) vs C')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"  Saved: {out_png}")
    plt.close(fig)


# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("SENSITIVITY ANALYSIS OF SATURATION CONSTANT C (default 20)")
    print("=" * 80)

    df_pairs = load_pairs_with_coords()
    if len(df_pairs) == 0:
        print("ERROR: no pairs with coordinates loaded.")
        return

    # Make sure TM-score column is present and named consistently
    if 'tm_score' not in df_pairs.columns and 'TM_score' in df_pairs.columns:
        df_pairs = df_pairs.rename(columns={'TM_score': 'tm_score'})
    if 'tm_score' not in df_pairs.columns:
        raise KeyError("TM-score column not found in homology_real_ogh_results.csv")

    C_grid = (2, 5, 10, 15, 20, 30, 40, 50, 100)
    print(f"\nTesting C in {C_grid}")
    print(f"Pairs: {len(df_pairs)}\n")

    df_res = run_saturation_analysis(df_pairs, C_grid=C_grid)

    out_csv = 'sensitivity_saturation_constant.csv'
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    plot_saturation(df_res)

    # ---- Stability summary ----
    print("\n" + "=" * 80)
    print("STABILITY SUMMARY")
    print("=" * 80)

    for col in ['AUC_overall', 'AUC_diverse', 'AUC_twilight', 'AUC_homologous',
                'pearson_r', 'spearman_rho']:
        if col not in df_res.columns:
            continue
        v = df_res[col].dropna().values
        if len(v) == 0:
            continue
        print(f"{col:>18}: min={v.min():.4f}  max={v.max():.4f}  "
              f"range={v.max()-v.min():.4f}  std={v.std():.4f}")

    # Compare C=20 against the rest
    if 20 in df_res['C'].values:
        base = df_res[df_res['C'] == 20].iloc[0]
        print("\nDeviation from C=20 (default):")
        for _, row in df_res.iterrows():
            if row['C'] == 20:
                continue
            print(f"  C={row['C']:>4}: "
                  f"dAUC_overall={row['AUC_overall']-base['AUC_overall']:+.4f}  "
                  f"dPearson={row['pearson_r']-base['pearson_r']:+.4f}  "
                  f"dSpearman={row['spearman_rho']-base['spearman_rho']:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()