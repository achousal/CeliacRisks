#!/usr/bin/env python3
"""
validate_calibration_counts.py

Diagnostic script to validate sample count calculations in calibration plots.
Identifies discrepancies between probability-space and logit-space bin counts
for SINGLE-SPLIT plots.

Usage:
    python scripts/validate_calibration_counts.py
"""

import numpy as np
import sys


def validate_single_split_counts():
    """Validate bin count logic for single-split calibration plots.

    This is the case when split_ids is None or has only one unique value.

    Returns True if logic is consistent, False if discrepancy found.
    """
    n_samples = 1000
    n_bins = 10
    min_bin_size_logit = 1  # Updated default in _binned_logits (was 30)

    print(f"Simulating SINGLE SPLIT with {n_samples} samples")
    print(f"Number of bins: {n_bins}")
    print(f"Logit min_bin_size threshold: {min_bin_size_logit}")
    print("")

    # Generate skewed predictions (like real risk scores - most near 0)
    np.random.seed(42)
    # Beta distribution to simulate typical risk scores (most low, few high)
    p = np.random.beta(0.5, 5, n_samples)
    y = (np.random.random(n_samples) < p).astype(int)  # Labels based on probs

    bins_uniform = np.linspace(0, 1, n_bins + 1)

    # === PROBABILITY SPACE - Single Split (Panel 2) ===
    print("=" * 60)
    print("PROBABILITY SPACE - Uniform Binning (Single Split)")
    print("=" * 60)
    print("")
    print("Code from _plot_prob_calibration_panel (lines 175-193):")
    print("  for i in range(actual_n_bins):")
    print("      m = bin_idx == i")
    print("      if m.sum() == 0:")
    print("          sizes.append(0)")
    print("      else:")
    print("          sizes.append(int(m.sum()))  # <-- ALL bins included")
    print("")

    bin_idx = np.digitize(p, bins_uniform) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    prob_sizes = []
    for i in range(n_bins):
        m = bin_idx == i
        prob_sizes.append(int(m.sum()))

    prob_sizes = np.array(prob_sizes)
    valid_prob = prob_sizes > 0

    print(f"Bin edges: {bins_uniform}")
    print(f"Samples per bin: {prob_sizes.tolist()}")
    print(f"Total samples: {prob_sizes.sum()}")
    print(f"Bins with data: {valid_prob.sum()} / {n_bins}")
    print(f"Legend would show sizes: {prob_sizes[valid_prob].tolist()}")

    # === LOGIT SPACE - Single Split (Panel 4) ===
    print("")
    print("=" * 60)
    print("LOGIT SPACE - Uniform Binning (Single Split)")
    print("=" * 60)
    print("")
    print("Code from _binned_logits (lines 354-360):")
    print("  for i in range(len(bins) - 1):")
    print("      mask_bin = bin_idx == i")
    print("      if mask_bin.sum() < min_bin_size and merge_tail:")
    print("          continue  # <-- SKIPS bins with < 30 samples!")
    print("      ...")
    print("      sizes_list.append(n)")
    print("")

    # Replicate _binned_logits logic
    logit_sizes = []
    logit_bin_indices = []
    for i in range(n_bins):
        mask_bin = bin_idx == i
        count = mask_bin.sum()
        if count < min_bin_size_logit:
            continue  # Skip small bins (merge_tail=True default)
        logit_sizes.append(int(count))
        logit_bin_indices.append(i)

    logit_sizes = np.array(logit_sizes) if logit_sizes else np.array([])

    print(f"Bins KEPT (>= {min_bin_size_logit} samples): {logit_bin_indices}")
    print(f"Samples per kept bin: {logit_sizes.tolist()}")
    print(f"Total samples in kept bins: {logit_sizes.sum() if len(logit_sizes) > 0 else 0}")
    print(f"Bins shown: {len(logit_sizes)} / {n_bins}")
    print(f"Legend would show sizes: {logit_sizes.tolist()}")

    # === DISCREPANCY ANALYSIS ===
    print("")
    print("=" * 60)
    print("DISCREPANCY ANALYSIS")
    print("=" * 60)

    skipped_bins = [i for i in range(n_bins) if prob_sizes[i] < min_bin_size_logit and prob_sizes[i] > 0]
    skipped_samples = sum(prob_sizes[i] for i in skipped_bins)

    print(f"\nProbability space shows ALL {valid_prob.sum()} non-empty bins")
    print(f"Logit space shows only {len(logit_sizes)} bins (with >= {min_bin_size_logit} samples)")
    print("")
    print(f"Bins skipped in logit: {skipped_bins}")
    print(f"Samples in skipped bins: {skipped_samples}")
    print("")

    if len(logit_sizes) > 0 and valid_prob.sum() > 0:
        prob_total = prob_sizes[valid_prob].sum()
        logit_total = logit_sizes.sum()
        print(f"Probability space total: {prob_total}")
        print(f"Logit space total: {logit_total}")
        print(f"Difference: {prob_total - logit_total} samples 'missing' from logit")

    print("")
    print("=" * 60)
    print("ROOT CAUSE")
    print("=" * 60)
    print("")
    print("The logit-space panel uses _binned_logits() which has:")
    print("  - min_bin_size=30 (default)")
    print("  - merge_tail=True (default)")
    print("")
    print("This SKIPS bins with fewer than 30 samples, so:")
    print("  1. Fewer bins are shown in logit space")
    print("  2. Total sample count appears lower")
    print("")
    print("The probability-space panel includes ALL non-empty bins.")

    # Check if this is the actual issue
    has_discrepancy = len(logit_sizes) < valid_prob.sum()

    return not has_discrepancy


def validate_multi_split_counts():
    """Validate bin count logic for multi-split aggregated plots."""
    print("")
    print("=" * 60)
    print("MULTI-SPLIT AGGREGATION (for reference)")
    print("=" * 60)
    print("")
    print("Additional issue in multi-split mode:")
    print("  - Probability space uses: np.nansum(counts_all, axis=0)")
    print("  - Logit space uses: np.nanmean(bin_sizes_per_split, axis=0)")
    print("")
    print("This means logit shows ~1/n_splits of the probability-space counts.")


def check_code_locations():
    """Print code locations for reference."""
    print("")
    print("=" * 60)
    print("CODE LOCATION REFERENCE")
    print("=" * 60)
    print("")
    print("File: analysis/src/ced_ml/plotting/calibration.py")
    print("")
    print("Single-split probability-space (lines 175-193):")
    print("  - Iterates ALL bins, includes any with samples")
    print("  - No minimum bin size threshold")
    print("")
    print("Single-split logit-space via _binned_logits (lines 354-360):")
    print("  - Skips bins with < min_bin_size (default 30) samples")
    print("  - This reduces visible bins and total count")
    print("")
    print("FIX OPTIONS:")
    print("  A) Remove min_bin_size filter in _binned_logits for display")
    print("  B) Add min_bin_size filter to probability-space for consistency")
    print("  C) Keep both, but document the difference in panel titles")


if __name__ == "__main__":
    print("=" * 60)
    print("Calibration Plot Sample Count Validation")
    print("=" * 60)
    print("")

    is_consistent = validate_single_split_counts()
    validate_multi_split_counts()
    check_code_locations()

    print("")
    print("=" * 60)
    if is_consistent:
        print("VALIDATION RESULT: OK - Logic is consistent")
        sys.exit(0)
    else:
        print("VALIDATION RESULT: DISCREPANCY FOUND")
        print("See above for root cause and fix options.")
        sys.exit(1)
