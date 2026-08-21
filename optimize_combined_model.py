#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
optimize_combined_model.py — Optimize combined OGH + TM-score model

Description:
    Tests two advanced combination strategies for integrating OGH and TM-score:
    1. Weighted sum (learn optimal weights via grid search on validation set)
    2. Logistic Regression (learn decision boundary with train/validation/test split)

    Uses train/validation/test split (60/20/20) with stratification to ensure
    unbiased evaluation of the combined predictor.

Input:
    homology_real_ogh_results.csv — Results from SCOP/ASTRAL validation
        Required columns: 'd_hat_OGH', 'TM_score', 'is_homolog'

Output:
    combined_optimization_results.csv — All metrics for each method (AUC, accuracy, F1)
    combined_optimization_plot.png — Comparison bar plot and ROC curves

Usage:
    python optimize_combined_model.py

Dependencies:
    numpy, pandas, scikit-learn, matplotlib

Author: Andrey V. Timofeev, Alexandr S. Anufriev
Date: 2026
License: MIT
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, f1_score
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')


# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_data():
    """Load the SCOP/ASTRAL validation results."""
    if not os.path.exists('homology_real_ogh_results.csv'):
        print("❌ File not found: homology_real_ogh_results.csv")
        return None

    df = pd.read_csv('homology_real_ogh_results.csv')
    print(f"✅ Loaded {len(df)} pairs from SCOP/ASTRAL")

    # Check required columns
    required = ['d_hat_OGH', 'TM_score', 'is_homolog']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        return None

    return df


# ============================================================================
# 2. FEATURE NORMALIZATION
# ============================================================================

def normalize_features(df):
    """
    Normalize features to [0, 1] interval.

    OGH is inverted (lower is better → higher is better after normalization).
    TM-score is normalized directly (higher is better).
    """
    df = df.copy()

    # OGH: invert (smaller distance → better)
    ogh_min, ogh_max = df['d_hat_OGH'].min(), df['d_hat_OGH'].max()
    df['OGH_norm'] = 1.0 - (df['d_hat_OGH'] - ogh_min) / (ogh_max - ogh_min + 1e-8)

    # TM-score: direct normalization
    tm_min, tm_max = df['TM_score'].min(), df['TM_score'].max()
    df['TM_norm'] = (df['TM_score'] - tm_min) / (tm_max - tm_min + 1e-8)

    return df


# ============================================================================
# 3. COMBINATION METHODS
# ============================================================================

def simple_average(ogh_norm, tm_norm):
    """Simple arithmetic mean (baseline)."""
    return 0.5 * ogh_norm + 0.5 * tm_norm


def weighted_sum(ogh_norm, tm_norm, w):
    """Weighted sum with weight w for OGH and (1-w) for TM-score."""
    return w * ogh_norm + (1 - w) * tm_norm


def logistic_regression_predict(X_train, y_train, X_val, X_test):
    """
    Train logistic regression on train set, evaluate on val and test sets.

    Returns:
        y_val_prob: predicted probabilities on validation set
        y_test_prob: predicted probabilities on test set
        model: trained logistic regression model
        scaler: fitted StandardScaler
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    # Predict probabilities
    y_val_prob = lr.predict_proba(X_val_scaled)[:, 1]
    y_test_prob = lr.predict_proba(X_test_scaled)[:, 1]

    return y_val_prob, y_test_prob, lr, scaler


# ============================================================================
# 4. EVALUATION
# ============================================================================

def evaluate_predictions(y_true, y_pred_prob, name):
    """Compute evaluation metrics for a given predictor."""
    auc_score = roc_auc_score(y_true, y_pred_prob)

    # For accuracy and F1, use threshold 0.5
    y_pred_binary = (y_pred_prob >= 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)

    return {
        'method': name,
        'AUC': auc_score,
        'accuracy': accuracy,
        'f1_score': f1
    }


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def plot_results(df_results, y_test, y_test_simple, y_test_weighted, y_test_lr):
    """Generate comparison bar plot and ROC curves."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ============================================================
    # Panel 1: AUC comparison
    # ============================================================
    ax1 = axes[0]
    methods = df_results['method'].tolist()
    aucs = df_results['AUC'].tolist()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax1.bar(methods, aucs, color=colors, alpha=0.7, edgecolor='black')

    # Add values on top of bars
    for bar, auc_val in zip(bars, aucs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f'{auc_val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title('Comparison of combination methods (Test set)', fontsize=13)
    ax1.set_ylim(0.70, 0.80)
    ax1.grid(True, alpha=0.3, axis='y')

    # Baseline reference line
    baseline_auc = df_results[df_results['method'].str.contains('Simple')]['AUC'].values[0]
    ax1.axhline(y=baseline_auc, color='gray', linestyle='--', alpha=0.5,
                label=f'Baseline: {baseline_auc:.3f}')
    ax1.legend(loc='lower right')

    # ============================================================
    # Panel 2: ROC curves
    # ============================================================
    ax2 = axes[1]

    # Simple Average
    fpr_simple, tpr_simple, _ = roc_curve(y_test, y_test_simple)
    ax2.plot(fpr_simple, tpr_simple, 'b-', linewidth=2.5,
             label=f'Simple Average (AUC={roc_auc_score(y_test, y_test_simple):.3f})')

    # Weighted Sum
    fpr_weighted, tpr_weighted, _ = roc_curve(y_test, y_test_weighted)
    ax2.plot(fpr_weighted, tpr_weighted, 'orange', linewidth=2.5,
             label=f'Weighted Sum (AUC={roc_auc_score(y_test, y_test_weighted):.3f})')

    # Logistic Regression
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_test_lr)
    ax2.plot(fpr_lr, tpr_lr, 'g-', linewidth=2.5,
             label=f'Logistic Regression (AUC={roc_auc_score(y_test, y_test_lr):.3f})')

    # Random classifier
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.500)')

    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curves on Test Set', fontsize=13)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('combined_optimization_plot.png', dpi=300, bbox_inches='tight')
    print("   ✅ combined_optimization_plot.png")
    plt.show()


# ============================================================================
# 6. MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("🔬 OPTIMIZING COMBINED OGH + TM-SCORE MODEL")
    print("   Comparison: Simple Average vs Weighted Sum vs Logistic Regression")
    print("=" * 80)

    # 1. Load data
    df = load_data()
    if df is None:
        return

    # 2. Normalize features
    df = normalize_features(df)
    print(f"\n📊 Normalization complete")
    print(f"   OGH: min={df['d_hat_OGH'].min():.3f}, max={df['d_hat_OGH'].max():.3f}")
    print(f"   TM-score: min={df['TM_score'].min():.3f}, max={df['TM_score'].max():.3f}")

    # 3. Train/validation/test split (60/20/20) with stratification
    X = df[['OGH_norm', 'TM_norm']].values
    y = df['is_homolog'].values

    # First, separate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Then split remaining into train (75% of 80% = 60% of total) and val (25% of 80% = 20% of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"\n📋 DATA SPLIT:")
    print(f"   Train: {len(X_train)} pairs ({len(X_train) / len(df) * 100:.1f}%)")
    print(f"   Val:   {len(X_val)} pairs ({len(X_val) / len(df) * 100:.1f}%)")
    print(f"   Test:  {len(X_test)} pairs ({len(X_test) / len(df) * 100:.1f}%)")

    # Check class balance
    print(f"\n   Class balance in Train: {y_train.sum()}/{len(y_train)} homologs")
    print(f"   Class balance in Test:  {y_test.sum()}/{len(y_test)} homologs")

    # Extract normalized features
    ogh_train, tm_train = X_train[:, 0], X_train[:, 1]
    ogh_val, tm_val = X_val[:, 0], X_val[:, 1]
    ogh_test, tm_test = X_test[:, 0], X_test[:, 1]

    # ============================================================
    # METHOD 1: SIMPLE AVERAGE (BASELINE)
    # ============================================================
    print("\n" + "=" * 80)
    print("📊 METHOD 1: SIMPLE AVERAGE (BASELINE)")
    print("=" * 80)

    y_val_simple = simple_average(ogh_val, tm_val)
    y_test_simple = simple_average(ogh_test, tm_test)

    val_simple = evaluate_predictions(y_val, y_val_simple, "Simple Average (Val)")
    test_simple = evaluate_predictions(y_test, y_test_simple, "Simple Average (Test)")

    print(f"   Validation AUC: {val_simple['AUC']:.4f}")
    print(f"   Test AUC:       {test_simple['AUC']:.4f}")

    # ============================================================
    # METHOD 2: WEIGHTED SUM (GRID SEARCH)
    # ============================================================
    print("\n" + "=" * 80)
    print("📊 METHOD 2: WEIGHTED SUM (GRID SEARCH)")
    print("=" * 80)

    best_w = 0.5
    best_auc = 0
    best_y_val_weighted = None

    # Search over weights from 0 to 1
    weights = np.linspace(0, 1, 101)

    for w in weights:
        y_pred = weighted_sum(ogh_val, tm_val, w)
        auc_score = roc_auc_score(y_val, y_pred)
        if auc_score > best_auc:
            best_auc = auc_score
            best_w = w
            best_y_val_weighted = y_pred

    # Apply best weight to test set
    y_test_weighted = weighted_sum(ogh_test, tm_test, best_w)

    val_weighted = evaluate_predictions(y_val, best_y_val_weighted,
                                        f"Weighted Sum (Val, w={best_w:.2f})")
    test_weighted = evaluate_predictions(y_test, y_test_weighted,
                                         f"Weighted Sum (Test, w={best_w:.2f})")

    print(f"   Best weight w (OGH): {best_w:.2f}")
    print(f"   TM-score weight:     {1 - best_w:.2f}")
    print(f"   Validation AUC:      {val_weighted['AUC']:.4f}")
    print(f"   Test AUC:            {test_weighted['AUC']:.4f}")

    # ============================================================
    # METHOD 3: LOGISTIC REGRESSION
    # ============================================================
    print("\n" + "=" * 80)
    print("📊 METHOD 3: LOGISTIC REGRESSION")
    print("=" * 80)

    y_val_lr, y_test_lr, lr_model, scaler_model = logistic_regression_predict(
        X_train, y_train, X_val, X_test
    )

    val_lr = evaluate_predictions(y_val, y_val_lr, "Logistic Regression (Val)")
    test_lr = evaluate_predictions(y_test, y_test_lr, "Logistic Regression (Test)")

    # Model coefficients
    coef_ogh = lr_model.coef_[0][0]
    coef_tm = lr_model.coef_[0][1]
    intercept = lr_model.intercept_[0]

    print(f"   OGH coefficient:     {coef_ogh:.4f}")
    print(f"   TM-score coefficient: {coef_tm:.4f}")
    print(f"   Intercept:           {intercept:.4f}")
    print(f"   Validation AUC:      {val_lr['AUC']:.4f}")
    print(f"   Test AUC:            {test_lr['AUC']:.4f}")

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("\n" + "=" * 80)
    print("📋 SUMMARY TABLE (TEST SET)")
    print("=" * 80)

    results = [test_simple, test_weighted, test_lr]
    df_results = pd.DataFrame(results)
    df_results['method'] = [
        'Simple Average (baseline)',
        f'Weighted Sum (w={best_w:.2f})',
        'Logistic Regression'
    ]
    df_results = df_results[['method', 'AUC', 'accuracy', 'f1_score']]

    print(df_results.to_string(index=False))

    # Save results
    df_results.to_csv('combined_optimization_results.csv', index=False)
    print(f"\n✅ Saved: combined_optimization_results.csv")

    # ============================================================
    # VISUALIZATION
    # ============================================================
    print("\n📊 Generating plots...")
    plot_results(df_results, y_test, y_test_simple, y_test_weighted, y_test_lr)

    # ============================================================
    # CONCLUSIONS
    # ============================================================
    print("\n" + "=" * 80)
    print("📌 CONCLUSIONS")
    print("=" * 80)

    best_method = df_results.loc[df_results['AUC'].idxmax()]
    improvement = (best_method['AUC'] - test_simple['AUC']) * 100

    print(f"\n   Best method by AUC: {best_method['method']}")
    print(f"   AUC: {best_method['AUC']:.4f}")
    print(f"   Improvement over Simple Average: +{improvement:.2f}%")
    print(f"   Accuracy: {best_method['accuracy']:.4f}")
    print(f"   F1-score: {best_method['f1_score']:.4f}")

    if 'Logistic Regression' in best_method['method']:
        print(f"\n   Logistic Regression coefficients:")
        print(f"     OGH:       {coef_ogh:.4f}")
        print(f"     TM-score:  {coef_tm:.4f}")
        print(f"     Intercept: {intercept:.4f}")
        print(f"\n   Decision function:")
        print(f"     P(homolog) = 1 / (1 + exp(-({coef_ogh:.2f}*OGH_norm + {coef_tm:.2f}*TM_norm + {intercept:.2f})))")

    print("\n✅ Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()