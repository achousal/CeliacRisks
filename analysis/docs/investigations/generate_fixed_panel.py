#!/usr/bin/env python3
"""
Generate fixed 100-protein panel for factorial experiment.

This ensures all configurations use the same features,
isolating sampling effects (prevalent_frac × case:control).
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from ced_ml.data.schema import TARGET_COL, CASE_LABELS


def compute_univariate_auc(X, y):
    """Compute univariate AUROC for each protein."""
    y_binary = np.isin(y, CASE_LABELS).astype(int)
    aucs = {}

    for col in X.columns:
        try:
            auc = roc_auc_score(y_binary, X[col])
            # Store absolute distance from 0.5 (discriminative power)
            aucs[col] = abs(auc - 0.5)
        except:
            aucs[col] = 0.0

    return aucs


def main():
    parser = argparse.ArgumentParser(
        description='Generate fixed 100-protein panel for experiment'
    )
    parser.add_argument(
        '--infile',
        type=Path,
        required=True,
        help='Input data file (parquet)'
    )
    parser.add_argument(
        '--outfile',
        type=Path,
        default=Path('top100_panel.csv'),
        help='Output panel file (default: top100_panel.csv)'
    )
    parser.add_argument(
        '--final-k',
        type=int,
        default=100,
        help='Final number of features (default: 100)'
    )
    args = parser.parse_args()

    print(f"Loading data from {args.infile}...")
    df = pd.read_parquet(args.infile)

    # Get protein columns (all columns ending with _resid)
    protein_cols = [col for col in df.columns if col.endswith('_resid')]
    X = df[protein_cols]
    y = df[TARGET_COL]

    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} proteins")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # Compute univariate AUCs
    print(f"\nComputing univariate AUROC for all proteins...")
    aucs = compute_univariate_auc(X, y)

    # Sort by discriminative power
    sorted_proteins = sorted(aucs.items(), key=lambda x: x[1], reverse=True)
    top_features = [p[0] for p in sorted_proteins[:args.final_k]]

    print(f"Selected top {len(top_features)} proteins by AUROC")
    print(f"AUROC range: {sorted_proteins[0][1]:.4f} (best) to {sorted_proteins[args.final_k-1][1]:.4f} (k={args.final_k})")

    # Save panel (no header - just protein names, one per line)
    args.outfile.write_text('\n'.join(top_features) + '\n')

    print(f"\nPanel saved to: {args.outfile}")
    print(f"Proteins: {len(top_features)} (no header)")
    print(f"First 10 proteins: {top_features[:10]}")


if __name__ == '__main__':
    main()
