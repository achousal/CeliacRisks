"""Biomarker panel building with correlation pruning.

Constructs clinical biomarker panels from feature selection results by:
1. Building raw panels from selection frequencies (topN or frequency threshold)
2. Pruning highly correlated proteins using connected components
3. Refilling to target size from ranked candidate pool

Design:
- Pure functions operating on DataFrames and dictionaries
- No side effects: all outputs explicit
- Supports multiple panel sizes and correlation thresholds
- Graph-based correlation component detection
"""

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def compute_univariate_strength(
    df: pd.DataFrame,
    y: np.ndarray,
    proteins: List[str],
) -> Dict[str, Tuple[float, float]]:
    """Compute univariate Mann-Whitney p-values and effect sizes for proteins.

    Used for tie-breaking when multiple proteins have equal selection frequency.

    Args:
        df: DataFrame containing protein columns
        y: Binary outcome array (0/1)
        proteins: List of protein column names to evaluate

    Returns:
        Dict mapping protein -> (p_value, abs_mean_diff)
        - p_value: Mann-Whitney U test p-value (smaller = stronger)
        - abs_mean_diff: Absolute difference in means between cases/controls

    Example:
        >>> df = pd.DataFrame({'PROT_A': [1, 2, 3, 4], 'PROT_B': [2, 3, 4, 5]})
        >>> y = np.array([0, 0, 1, 1])
        >>> compute_univariate_strength(df, y, ['PROT_A', 'PROT_B'])
        {'PROT_A': (0.33, 2.0), 'PROT_B': (0.33, 2.0)}
    """
    y_int = np.asarray(y, dtype=int)
    result: Dict[str, Tuple[float, float]] = {}

    for protein in proteins:
        if protein not in df.columns:
            continue

        # Convert to numeric and filter missing
        x = pd.to_numeric(df[protein], errors="coerce")
        valid = x.notna().to_numpy()

        # Require minimum sample size and both classes present
        if valid.sum() < 30 or len(np.unique(y_int[valid])) < 2:
            continue

        x_valid = x[valid].to_numpy(dtype=float)
        y_valid = y_int[valid]

        # Split by class
        x_control = x_valid[y_valid == 0]
        x_case = x_valid[y_valid == 1]

        # Require minimum per-class sample size
        if len(x_control) < 5 or len(x_case) < 5:
            continue

        # Compute Mann-Whitney U test
        try:
            try:
                # Try with method parameter (scipy >= 1.7)
                _, p_value = stats.mannwhitneyu(
                    x_case, x_control, alternative="two-sided", method="asymptotic"
                )
            except TypeError:
                # Fallback for older scipy
                _, p_value = stats.mannwhitneyu(
                    x_case, x_control, alternative="two-sided"
                )
            p_value = float(p_value)
        except Exception:
            p_value = np.nan

        # Compute effect size (mean difference)
        mean_diff = float(np.nanmean(x_case) - np.nanmean(x_control))

        result[protein] = (p_value, abs(mean_diff))

    return result


def prune_correlated_proteins(
    df: pd.DataFrame,
    y: Optional[np.ndarray],
    proteins: List[str],
    selection_freq: Optional[Dict[str, float]] = None,
    corr_threshold: float = 0.80,
    corr_method: Literal["pearson", "spearman"] = "pearson",
    tiebreak_method: Literal["freq", "freq_then_univariate"] = "freq",
) -> Tuple[pd.DataFrame, List[str]]:
    """Prune correlated proteins using connected components graph algorithm.

    Builds correlation graph where edges connect proteins with |corr| >= threshold.
    Each connected component represents a cluster of correlated proteins.
    Keeps one representative per component based on selection frequency (and
    optionally univariate strength for tie-breaking).

    Args:
        df: DataFrame containing protein columns (typically TRAIN set)
        y: Binary outcome array (required if tiebreak_method="freq_then_univariate")
        proteins: List of protein column names to evaluate
        selection_freq: Dict mapping protein -> selection frequency (for tie-breaking)
        corr_threshold: Correlation threshold for pruning (default: 0.80)
        corr_method: Correlation method ("pearson" or "spearman")
        tiebreak_method: How to select representative from component:
            - "freq": Highest selection frequency wins (default)
            - "freq_then_univariate": Use univariate p-value if frequencies tied

    Returns:
        (component_map, kept_proteins)
        - component_map: DataFrame with columns:
            [component_id, protein, selection_freq, kept, rep_protein, component_size]
        - kept_proteins: List of representative proteins (one per component)

    Example:
        >>> df = pd.DataFrame({
        ...     'A': [1, 2, 3, 4],
        ...     'B': [1.1, 2.1, 3.1, 4.1],  # Highly correlated with A
        ...     'C': [4, 3, 2, 1]  # Negatively correlated with A/B
        ... })
        >>> freqs = {'A': 0.9, 'B': 0.8, 'C': 0.7}
        >>> component_map, kept = prune_correlated_proteins(
        ...     df, None, ['A', 'B', 'C'], freqs, corr_threshold=0.9
        ... )
        >>> kept  # A and B form component, A kept (higher freq)
        ['A', 'C']
    """
    # Filter to proteins present in df
    available = [p for p in proteins if p in df.columns]
    if len(available) == 0:
        empty_df = pd.DataFrame(
            columns=[
                "component_id",
                "protein",
                "selection_freq",
                "kept",
                "rep_protein",
                "component_size",
            ]
        )
        return empty_df, []

    # Prepare numeric data
    X = df[available].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.median(axis=0, skipna=True))

    # Validate correlation method
    if corr_method not in ("pearson", "spearman"):
        corr_method = "pearson"

    # Compute correlation matrix
    corr_matrix = X.corr(method=corr_method).abs().fillna(0.0)

    # Build adjacency list for correlation graph
    adjacency = {p: set() for p in available}
    for i, p1 in enumerate(available):
        for j in range(i + 1, len(available)):
            p2 = available[j]
            if float(corr_matrix.loc[p1, p2]) >= float(corr_threshold):
                adjacency[p1].add(p2)
                adjacency[p2].add(p1)

    # Find connected components using DFS
    visited = set()
    components = []

    for protein in available:
        if protein in visited:
            continue

        # Depth-first search to find component
        stack = [protein]
        component = []
        visited.add(protein)

        while stack:
            node = stack.pop()
            component.append(node)

            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        components.append(sorted(component))

    # Compute univariate strengths if needed for tie-breaking
    univariate_map: Dict[str, Tuple[float, float]] = {}
    if tiebreak_method == "freq_then_univariate" and y is not None:
        univariate_map = compute_univariate_strength(df, y, available)

    # Select representative from each component
    kept_proteins = []
    component_rows = []

    for component_id, component in enumerate(components, start=1):
        # Define sorting key for selecting representative
        def representative_key(protein: str) -> tuple:
            # Primary: Higher selection frequency (negate for descending)
            freq = selection_freq.get(protein, np.nan) if selection_freq else np.nan
            freq_val = freq if np.isfinite(freq) else 0.0
            primary = -freq_val

            # Secondary: Univariate strength (if enabled and tied on frequency)
            if tiebreak_method == "freq_then_univariate":
                p_value, abs_delta = univariate_map.get(protein, (np.nan, np.nan))
                p_val = p_value if np.isfinite(p_value) else 1.0
                delta_val = abs_delta if np.isfinite(abs_delta) else 0.0
                # Sort by: freq DESC, p_value ASC, abs_delta DESC, name ASC
                return (primary, p_val, -delta_val, protein)
            else:
                # Sort by: freq DESC, name ASC
                return (primary, protein)

        # Select representative (first after sorting)
        representative = sorted(component, key=representative_key)[0]
        kept_proteins.append(representative)

        # Record mapping for all proteins in component
        for protein in component:
            freq = selection_freq.get(protein, np.nan) if selection_freq else np.nan
            component_rows.append(
                {
                    "component_id": component_id,
                    "protein": protein,
                    "selection_freq": freq,
                    "kept": (protein == representative),
                    "rep_protein": representative,
                    "component_size": len(component),
                }
            )

    # Build component map DataFrame
    component_map = (
        pd.DataFrame(component_rows)
        .sort_values(
            ["kept", "selection_freq", "protein"],
            ascending=[False, False, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    # Sort kept proteins by frequency
    kept_proteins = sorted(
        set(kept_proteins),
        key=lambda p: (-(selection_freq.get(p, 0.0) if selection_freq else 0.0), p),
    )

    return component_map, kept_proteins


def prune_and_refill_panel(
    df: pd.DataFrame,
    y: Optional[np.ndarray],
    ranked_proteins: List[str],
    selection_freq: Dict[str, float],
    target_size: int,
    corr_threshold: float,
    pool_limit: int,
    corr_method: Literal["pearson", "spearman"] = "pearson",
    tiebreak_method: Literal["freq", "freq_then_univariate"] = "freq",
) -> Tuple[pd.DataFrame, List[str]]:
    """Build correlation-pruned panel of fixed size with backfill.

    Algorithm:
    1. Take top N proteins from ranked list
    2. Prune correlated proteins (keep one per component)
    3. If pruned size < N, backfill from ranked list (skipping correlated candidates)

    Args:
        df: DataFrame containing protein columns (typically TRAIN set)
        y: Binary outcome array (required if tiebreak_method="freq_then_univariate")
        ranked_proteins: Proteins ranked by selection frequency (descending)
        selection_freq: Dict mapping protein -> selection frequency
        target_size: Desired final panel size
        corr_threshold: Correlation threshold for pruning
        pool_limit: Maximum number of candidates to consider for backfill
        corr_method: Correlation method ("pearson" or "spearman")
        tiebreak_method: How to select representative ("freq" or "freq_then_univariate")

    Returns:
        (component_map, final_panel)
        - component_map: DataFrame documenting pruning decisions
        - final_panel: List of proteins in final panel (length = target_size)

    Example:
        >>> df = pd.DataFrame({
        ...     'A': [1, 2, 3, 4],
        ...     'B': [1.1, 2.1, 3.1, 4.1],  # Correlated with A
        ...     'C': [4, 3, 2, 1],
        ...     'D': [2, 4, 1, 3]
        ... })
        >>> ranked = ['A', 'B', 'C', 'D']  # A > B > C > D by frequency
        >>> freqs = {'A': 0.9, 'B': 0.85, 'C': 0.7, 'D': 0.6}
        >>> component_map, panel = prune_and_refill_panel(
        ...     df, None, ranked, freqs, target_size=3,
        ...     corr_threshold=0.9, pool_limit=10
        ... )
        >>> panel  # Should be ['A', 'C', 'D'] (B pruned due to corr with A)
        ['A', 'C', 'D']
    """
    # Filter to available proteins
    available = [p for p in ranked_proteins if p in df.columns]

    # Take top N from ranked list
    top_n = available[: min(target_size, len(available))]

    # Prune correlated proteins
    component_map, kept = prune_correlated_proteins(
        df=df,
        y=y,
        proteins=top_n,
        selection_freq=selection_freq,
        corr_threshold=corr_threshold,
        corr_method=corr_method,
        tiebreak_method=tiebreak_method,
    )

    # Add metadata columns to component map
    if not component_map.empty:
        component_map["representative_flag"] = component_map["kept"].astype(bool)
        component_map["removed_due_to_corr_with"] = np.where(
            component_map["kept"], "", component_map["rep_protein"]
        )

    # If already at target size, done
    if len(kept) >= target_size:
        return component_map, kept[:target_size]

    # Backfill from pool of candidates
    pool = available[: min(pool_limit, len(available))]
    if not pool:
        return component_map, kept

    # Compute correlation matrix on pool for backfill checks
    X_pool = df[pool].apply(pd.to_numeric, errors="coerce")
    if X_pool.isna().any().any():
        X_pool = X_pool.fillna(X_pool.median(axis=0, skipna=True))

    if corr_method not in ("pearson", "spearman"):
        corr_method = "pearson"

    corr_matrix = X_pool.corr(method=corr_method).abs().fillna(0.0)

    # Greedily add candidates that aren't too correlated with existing panel
    final_panel = list(kept)
    kept_set = set(final_panel)

    for candidate in pool:
        if len(final_panel) >= target_size:
            break

        if candidate in kept_set:
            continue

        # Check if candidate is too correlated with any existing panel member
        too_correlated = False
        for existing in final_panel:
            if (
                candidate in corr_matrix.index
                and existing in corr_matrix.columns
                and float(corr_matrix.loc[candidate, existing]) >= corr_threshold
            ):
                too_correlated = True
                break

        if too_correlated:
            continue

        final_panel.append(candidate)
        kept_set.add(candidate)

    # Document backfilled proteins in component map
    if not component_map.empty and len(final_panel) > len(kept):
        max_component_id = int(component_map["component_id"].max())

        backfill_rows = []
        for i, protein in enumerate(final_panel[len(kept) :], start=1):
            backfill_rows.append(
                {
                    "component_id": max_component_id + i,
                    "protein": protein,
                    "selection_freq": float(selection_freq.get(protein, np.nan)),
                    "kept": True,
                    "rep_protein": protein,
                    "component_size": 1,
                    "representative_flag": True,
                    "removed_due_to_corr_with": "",
                }
            )

        component_map = pd.concat(
            [component_map, pd.DataFrame(backfill_rows)], ignore_index=True
        )

    return component_map, final_panel


def build_multi_size_panels(
    df: pd.DataFrame,
    y: Optional[np.ndarray],
    selection_freq: Dict[str, float],
    panel_sizes: List[int],
    corr_threshold: float = 0.80,
    pool_limit: int = 1000,
    corr_method: Literal["pearson", "spearman"] = "pearson",
    tiebreak_method: Literal["freq", "freq_then_univariate"] = "freq",
) -> Dict[int, Tuple[pd.DataFrame, List[str]]]:
    """Build multiple panels of different sizes with correlation pruning.

    Convenience wrapper around prune_and_refill_panel for building
    nested panels (e.g., 10, 25, 50, 100, 200 proteins).

    Args:
        df: DataFrame containing protein columns (typically TRAIN set)
        y: Binary outcome array (required if tiebreak_method="freq_then_univariate")
        selection_freq: Dict mapping protein -> selection frequency
        panel_sizes: List of target panel sizes (e.g., [10, 25, 50, 100])
        corr_threshold: Correlation threshold for pruning
        pool_limit: Maximum number of candidates to consider
        corr_method: Correlation method ("pearson" or "spearman")
        tiebreak_method: How to select representative ("freq" or "freq_then_univariate")

    Returns:
        Dict mapping panel_size -> (component_map, final_panel)

    Example:
        >>> df = pd.DataFrame(...)  # TRAIN set
        >>> freqs = {'A': 0.9, 'B': 0.8, ...}  # 100 proteins
        >>> panels = build_multi_size_panels(
        ...     df, y, freqs, panel_sizes=[10, 25, 50],
        ...     corr_threshold=0.80
        ... )
        >>> panels[10][1]  # 10-protein panel
        ['A', 'C', 'D', ...]
        >>> panels[50][1]  # 50-protein panel
        ['A', 'C', 'D', ..., 'Z']
    """
    # Rank proteins by selection frequency
    ranked = sorted(selection_freq.keys(), key=lambda p: (-selection_freq[p], p))

    results = {}
    for size in sorted(panel_sizes):
        component_map, panel = prune_and_refill_panel(
            df=df,
            y=y,
            ranked_proteins=ranked,
            selection_freq=selection_freq,
            target_size=size,
            corr_threshold=corr_threshold,
            pool_limit=pool_limit,
            corr_method=corr_method,
            tiebreak_method=tiebreak_method,
        )
        results[size] = (component_map, panel)

    return results
