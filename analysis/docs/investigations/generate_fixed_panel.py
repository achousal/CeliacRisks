#!/usr/bin/env python3
"""
Generate fixed 100-protein panel for factorial experiment.

This ensures all configurations use the same features,
isolating sampling effects (prevalent_frac × case:control).
"""

import argparse
from pathlib import Path

import pandas as pd

from ced_ml.data.io import load_data
from ced_ml.data.schema import CeliacDataSchema
from ced_ml.features.screening import screen_features
from ced_ml.features.kbest import select_k_best


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
        '--screen-method',
        default='mannwhitney',
        choices=['mannwhitney', 'ttest'],
        help='Screening method (default: mannwhitney)'
    )
    parser.add_argument(
        '--screen-top-n',
        type=int,
        default=1000,
        help='Number of features after screening (default: 1000)'
    )
    parser.add_argument(
        '--final-k',
        type=int,
        default=100,
        help='Final number of features (default: 100)'
    )
    args = parser.parse_args()

    print(f"Loading data from {args.infile}...")
    df = load_data(args.infile)

    # Initialize schema
    schema = CeliacDataSchema()
    X = df[schema.protein_cols]
    y = df[schema.target_col]

    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} proteins")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # Screen to top N
    print(f"\nScreening to top {args.screen_top_n} using {args.screen_method}...")
    screened_features = screen_features(
        X, y,
        method=args.screen_method,
        top_n=args.screen_top_n
    )
    print(f"Screened features: {len(screened_features)}")

    # Select top K
    print(f"\nSelecting top {args.final_k} by univariate AUC...")
    X_screened = X[screened_features]
    top_features = select_k_best(X_screened, y, k=args.final_k)
    print(f"Final panel: {len(top_features)} proteins")

    # Save panel
    panel_df = pd.DataFrame({'protein': top_features})
    panel_df.to_csv(args.outfile, index=False)

    print(f"\nPanel saved to: {args.outfile}")
    print(f"First 10 proteins: {top_features[:10]}")


if __name__ == '__main__':
    main()
