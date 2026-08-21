#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_supplementary_table_s2.py — Generate Supplementary Table S2

Description:
    Creates Supplementary Table S2 containing the complete list of 1,400
    protein pairs used for homology detection validation in the OGH study.
    The table includes protein IDs, SCOP family assignments, homology labels,
    and computed metrics (OGH, TM-score).

Input:
    homology_real_ogh_results.csv — Results from homology_validation_real_ogh_v2_eng.p.py

Output:
    supplementary_table_s2.csv — Complete list of 1,400 pairs with all metrics
    supplementary_table_s2.xlsx — Excel version with formatting

Format:
    Columns:
    - Pair_ID: Unique pair identifier (e.g., P0001)
    - Protein1: SCOP/ASTRAL ID of first protein
    - Protein2: SCOP/ASTRAL ID of second protein
    - Family1: SCOP family of first protein
    - Family2: SCOP family of second protein
    - Is_Homolog: 1 (same family) or 0 (different families)
    - OGH: \hat{d}_OGH distance (α=0.5)
    - TM_score: TM-score value
    - Coverage: Alignment coverage
    - Len1: Length of first protein
    - Len2: Length of second protein
    - TM_zone: Diverse / Twilight / Homologous

Dependencies:
    pandas, openpyxl (for Excel export)

Usage:
    python generate_supplementary_table_s2.py

Author: Andrey V. Timofeev, Alexandr S. Anufriev
Date: 2026
License: MIT
"""

import pandas as pd
import numpy as np
from pathlib import Path


def assign_tm_zone(tm_score):
    """Assign TM-score zone based on Zhang-Skolnick thresholds."""
    if tm_score <= 0.3:
        return 'Diverse (TM ≤ 0.3)'
    elif tm_score <= 0.5:
        return 'Twilight (0.3 < TM ≤ 0.5)'
    else:
        return 'Homologous (TM > 0.5)'


def generate_supplementary_table_s2(input_csv, output_csv, output_xlsx=None):
    """
    Generate Supplementary Table S2 from validation results.

    Parameters:
    -----------
    input_csv : str or Path
        Path to homology_real_ogh_results.csv
    output_csv : str or Path
        Path to save the supplementary table (CSV)
    output_xlsx : str or Path, optional
        Path to save the supplementary table (Excel)

    Returns:
    --------
    pd.DataFrame : Complete table with all columns
    """
    print("=" * 80)
    print("📋 GENERATING SUPPLEMENTARY TABLE S2")
    print("=" * 80)

    # Load results
    print(f"\n1. Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"   Loaded {len(df)} pairs")

    # Create supplementary table
    print("\n2. Creating supplementary table...")

    # Add Pair_ID (unique identifier)
    df['Pair_ID'] = [f"P{str(i + 1).zfill(4)}" for i in range(len(df))]

    # Extract family information from existing columns
    # The family column contains either a single family (homologs) or "fam1 vs fam2" (non-homologs)
    def extract_families(row):
        if row['is_homolog'] == 1:
            return row['family'], row['family']
        else:
            # Split "fam1 vs fam2"
            parts = row['family'].split(' vs ')
            if len(parts) == 2:
                return parts[0], parts[1]
            else:
                return row['family'], ''

    df['Family1'], df['Family2'] = zip(*df.apply(extract_families, axis=1))

    # Add TM zone
    df['TM_zone'] = df['TM_score'].apply(assign_tm_zone)

    # Reorder columns for the supplementary table
    columns_order = [
        'Pair_ID',
        'protein1',
        'protein2',
        'Family1',
        'Family2',
        'is_homolog',
        'TM_score',
        'd_hat_OGH',
        'coverage',
        'n_matches',
        'len1',
        'len2',
        'TM_zone'
    ]

    # Rename columns to match paper conventions
    df_supp = df[columns_order].copy()
    df_supp.columns = [
        'Pair ID',
        'Protein 1 (SCOP ID)',
        'Protein 2 (SCOP ID)',
        'Family 1',
        'Family 2',
        'Is Homolog (1=yes, 0=no)',
        'TM-score',
        'd_hat_OGH',
        'Alignment coverage',
        'Number of matched residues',
        'Length 1',
        'Length 2',
        'TM-score zone'
    ]

    # Sort by Pair ID
    df_supp = df_supp.sort_values('Pair ID').reset_index(drop=True)

    print(f"   Created table with {len(df_supp)} rows and {len(df_supp.columns)} columns")

    # Statistics summary
    print("\n3. Statistics summary:")
    print(f"   Total pairs: {len(df_supp)}")
    print(f"   Homologous pairs: {df_supp['Is Homolog (1=yes, 0=no)'].sum()}")
    print(f"   Non-homologous pairs: {len(df_supp) - df_supp['Is Homolog (1=yes, 0=no)'].sum()}")
    print(f"\n   Distribution by TM-score zone:")
    for zone in ['Diverse (TM ≤ 0.3)', 'Twilight (0.3 < TM ≤ 0.5)', 'Homologous (TM > 0.5)']:
        count = (df_supp['TM-score zone'] == zone).sum()
        if count > 0:
            print(f"     {zone}: {count} pairs")

    # Save CSV
    print(f"\n4. Saving CSV: {output_csv}")
    df_supp.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"   ✅ CSV saved")

    # Save Excel (if requested)
    if output_xlsx:
        print(f"\n5. Saving Excel: {output_xlsx}")
        try:
            # Use openpyxl for better Excel formatting
            with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
                # Main table
                df_supp.to_excel(writer, sheet_name='Table S2', index=False)

                # Statistics sheet
                stats_data = {
                    'Statistic': ['Total pairs', 'Homologous pairs', 'Non-homologous pairs'],
                    'Value': [len(df_supp), df_supp['Is Homolog (1=yes, 0=no)'].sum(),
                              len(df_supp) - df_supp['Is Homolog (1=yes, 0=no)'].sum()]
                }
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            print(f"   ✅ Excel saved")
        except Exception as e:
            print(f"   ⚠️ Could not save Excel: {e}")
            print("   Install openpyxl: pip install openpyxl")

    print("\n" + "=" * 80)
    print("✅ SUPPLEMENTARY TABLE S2 GENERATED")
    print("=" * 80)

    return df_supp


def generate_latex_table(df_supp, output_file='supplementary_table_s2.tex'):
    """
    Generate LaTeX table for the supplementary materials.

    Parameters:
    -----------
    df_supp : pd.DataFrame
        Supplementary table data
    output_file : str
        Path to save LaTeX file
    """
    print(f"\n6. Generating LaTeX table: {output_file}")

    # Select first 10 rows as example (full table would be too long)
    df_example = df_supp.head(10)

    # Generate LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append(
        "\\caption{Complete list of 1,400 protein pairs used for homology detection validation (first 10 rows shown).}")
    latex_lines.append("\\label{tab:s2}")
    latex_lines.append("\\begin{tabular}{llllrrrrrrr}")
    latex_lines.append("\\toprule")

    # Column headers
    headers = [
        'Pair ID', 'Protein 1', 'Protein 2', 'Family 1', 'Family 2',
        'Homolog', 'TM-score', 'd_hat_OGH', 'Coverage', 'Len1', 'Len2'
    ]
    latex_lines.append(" & ".join(headers) + " \\\\")
    latex_lines.append("\\midrule")

    # Data rows (first 10)
    for _, row in df_example.iterrows():
        # ✅ FIX: compute value outside f-string
        ogh_value = row['d_hat_OGH']
        row_data = [
            row['Pair ID'],
            row['Protein 1 (SCOP ID)'],
            row['Protein 2 (SCOP ID)'],
            row['Family 1'],
            row['Family 2'],
            str(row['Is Homolog (1=yes, 0=no)']),
            f"{row['TM-score']:.3f}",
            f"{ogh_value:.4f}",
            f"{row['Alignment coverage']:.3f}",
            str(row['Length 1']),
            str(row['Length 2'])
        ]
        latex_lines.append(" & ".join(row_data) + " \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\begin{tablenotes}")
    latex_lines.append("\\item[1] \\textit{Complete table available in the supplementary CSV file.}")
    latex_lines.append("\\item[2] \\textit{Homolog = 1 indicates proteins from the same SCOP family.}")
    latex_lines.append("\\end{tablenotes}")
    latex_lines.append("\\end{table}")

    # Save LaTeX
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))
    print(f"   ✅ LaTeX saved: {output_file}")

    return latex_lines


def main():
    # Paths
    project_root = Path(__file__).parent
    input_csv = project_root / 'homology_real_ogh_results.csv'
    output_csv = project_root / 'supplementary_table_s2.csv'
    output_xlsx = project_root / 'supplementary_table_s2.xlsx'
    output_latex = project_root / 'supplementary_table_s2.tex'

    # Check if input file exists
    if not input_csv.exists():
        print(f"❌ Input file not found: {input_csv}")
        print("Please run homology_validation_real_ogh_v2_eng.p.py first.")
        return

    # Generate supplementary table
    df_supp = generate_supplementary_table_s2(input_csv, output_csv, output_xlsx)

    # Generate LaTeX table (optional)
    try:
        generate_latex_table(df_supp, output_latex)
    except Exception as e:
        print(f"   ⚠️ Could not generate LaTeX: {e}")

    print("\n" + "=" * 80)
    print("📋 FILES GENERATED:")
    print(f"   CSV:  {output_csv}")
    if output_xlsx.exists():
        print(f"   Excel: {output_xlsx}")
    if Path(output_latex).exists():
        print(f"   LaTeX: {output_latex}")
    print("=" * 80)

    # Show first 5 rows as preview
    print("\n📝 PREVIEW (first 5 rows):")
    print(df_supp.head().to_string(index=False))

    print("\n✅ Done! Supplementary Table S2 is ready for the manuscript.")


if __name__ == "__main__":
    main()