#!/usr/bin/env python3
"""
save_splits.py

Purpose-built splitter for the CeD modeling workflow:
  1. Creates a never-touched holdout set (default 30%).
  2. Builds repeated stratified TRAIN/TEST splits for each scenario.
  3. Supports IncidentOnly and IncidentPlusPrevalent case definitions.
  4. Uses a 30% test fraction plus demographic-aware strata selection.
  5. Enforces shared row filtering so downstream training sees identical rows.

Holdout indices are stored using the global (post-filter) row numbers so that
training/evaluation code can drop them before any reset_index calls. In
contrast, TRAIN/TEST index CSVs are saved in the dev-local coordinate space
expected by celiacML_faith.py after the holdout rows are removed.
"""

import os
import sys
import json
import hashlib
import argparse
import warnings
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import shared row filter from celiacML_faith.py
# This ensures both scripts use identical filtering logic
try:
    from celiacML_faith import (
        apply_row_filters,
        TARGET_COL,
        CONTROL_LABEL,
        CED_DATE_COL,
        META_NUM_COLS,
        temporal_order_indices,
    )
    # Alias for backward compatibility with existing code
    META_NUM = META_NUM_COLS

    # Backwards-compatible alias for callers/tests that expect
    # `apply_training_row_filters` from earlier versions.
    apply_training_row_filters = apply_row_filters
except ImportError:
    # Fallback for standalone testing - will raise error if actually used
    warnings.warn(
        "Could not import from celiacML_faith.py. "
        "Row filtering may be inconsistent. "
        "Ensure celiacML_faith.py is in the same directory.",
        UserWarning
    )
    # Define minimal constants for error messages
    TARGET_COL = "CeD_comparison"
    CONTROL_LABEL = "Controls"
    CED_DATE_COL = "CeD_date"
    META_NUM = ["age", "BMI"]

    def apply_row_filters(*args, **kwargs):
        raise ImportError(
            "apply_row_filters must be imported from celiacML_faith.py "
            "to ensure consistent filtering between split generation and training."
        )

    def temporal_order_indices(*args, **kwargs):
        raise ImportError(
            "temporal_order_indices requires celiacML_faith.py. Ensure the scripts reside together."
        )

    apply_training_row_filters = apply_row_filters
# ---------------- constants ----------------
ID_COL = "eid"
INCIDENT_LABEL = "Incident"
PREVALENT_LABEL = "Prevalent"

META_CAT = ["sex", "Genetic ethnic grouping"]


def log(msg: str) -> None:
    print(f"[save_splits] {msg}", flush=True)


def compute_split_id(indices: np.ndarray) -> str:
    """Generate reproducible hash of split indices."""
    sorted_idx = np.sort(indices)
    hash_obj = hashlib.md5(sorted_idx.tobytes())
    return hash_obj.hexdigest()[:12]


def _downsample_controls(
    idx_set: np.ndarray,
    df: pd.DataFrame,
    case_labels: Optional[List[str]],
    controls_per_case: Optional[float],
    rng: np.random.RandomState,
    label: str,
) -> np.ndarray:
    if controls_per_case is None or controls_per_case <= 0:
        return np.sort(idx_set.astype(int))

    idx_set = np.asarray(idx_set, dtype=int)
    if idx_set.size == 0:
        return idx_set

    labels = df.loc[idx_set, TARGET_COL].astype(str)
    if case_labels is None:
        case_labels = []
    if isinstance(case_labels, str):
        case_labels = [case_labels]
    idx_cases = idx_set[labels.isin(case_labels)]
    idx_controls = idx_set[labels == CONTROL_LABEL]

    n_cases = int(idx_cases.size)
    n_controls = int(idx_controls.size)
    if n_cases == 0 or n_controls == 0:
        log(f"  [{label}] skip control downsample (cases={n_cases}, controls={n_controls})")
        return np.sort(idx_set.astype(int))

    target_controls = int(round(n_cases * float(controls_per_case)))
    if target_controls >= n_controls:
        log(f"  [{label}] keep all controls ({n_controls}); target={target_controls}")
        return np.sort(idx_set.astype(int))

    keep_controls = rng.choice(idx_controls, size=target_controls, replace=False)
    kept = np.sort(np.concatenate([idx_cases, keep_controls]).astype(int))
    log(f"  [{label}] downsample controls: {n_controls} -> {target_controls} (cases={n_cases})")
    return kept


# ---------------- stratification helpers ----------------
def _age_bins(age: pd.Series, scheme: str) -> pd.Series:
    age = age.fillna(age.median())
    if scheme == "age3":
        return pd.cut(age, bins=[0, 40, 60, 150], labels=["young", "middle", "old"]).astype(str)
    elif scheme == "age2":
        return pd.cut(age, bins=[0, 60, 150], labels=["lt60", "ge60"]).astype(str)
    else:
        raise ValueError(f"Unknown age scheme: {scheme}")


def _make_strata(df: pd.DataFrame, scheme: str) -> pd.Series:
    """
    Schemes:
      - "outcome+sex+age3"
      - "outcome+sex+age2"
      - "outcome+age3"
      - "outcome+sex"
      - "outcome"
    """
    outcome = df[TARGET_COL].astype(str).fillna("UnknownOutcome")
    sex = df["sex"].astype(str).fillna("UnknownSex")

    if scheme == "outcome+sex+age3":
        ageb = _age_bins(df["age"], "age3")
        return (outcome + "_" + sex + "_" + ageb).astype(str)

    if scheme == "outcome+sex+age2":
        ageb = _age_bins(df["age"], "age2")
        return (outcome + "_" + sex + "_" + ageb).astype(str)

    if scheme == "outcome+age3":
        ageb = _age_bins(df["age"], "age3")
        return (outcome + "_" + ageb).astype(str)

    if scheme == "outcome+sex":
        return (outcome + "_" + sex).astype(str)

    if scheme == "outcome":
        return outcome.astype(str)

    raise ValueError(f"Unknown stratification scheme: {scheme}")


def _collapse_rare_strata(df: pd.DataFrame, strata: pd.Series, min_count: int) -> pd.Series:
    vc = strata.value_counts(dropna=False)
    rare = set(vc[vc < min_count].index.tolist())
    if not rare:
        return strata

    outcome = df[TARGET_COL].astype(str).fillna("UnknownOutcome")
    collapsed = strata.copy()
    mask_rare = collapsed.isin(rare)
    collapsed.loc[mask_rare] = outcome.loc[mask_rare] + "_RARE"
    return collapsed.astype(str)


def _validate_strata_for_split(strata: pd.Series) -> Tuple[bool, str]:
    vc = strata.value_counts(dropna=False)
    minc = int(vc.min()) if len(vc) else 0
    if minc < 2:
        return False, f"min stratum count is {minc} (<2)"
    return True, "ok"


def build_working_strata(df_work: pd.DataFrame, min_count: int = 2) -> Tuple[pd.Series, str]:
    schemes = [
        "outcome+sex+age3",
        "outcome+sex+age2",
        "outcome+age3",
        "outcome+sex",
        "outcome",
    ]
    last_reason: Optional[str] = None

    for sch in schemes:
        strata = _make_strata(df_work, sch)
        strata = _collapse_rare_strata(df_work, strata, min_count=min_count)
        ok, reason = _validate_strata_for_split(strata)
        if ok:
            return strata, sch
        last_reason = f"{sch}: {reason}"

    strata = _make_strata(df_work, "outcome")
    strata = _collapse_rare_strata(df_work, strata, min_count=min_count)
    return strata, f"outcome (fallback; last failure: {last_reason})"


# ---------------- metadata ----------------
def save_split_metadata(
    outdir: str,
    scenario: str,
    seed: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    val_idx: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    split_type: str = "development",
    strat_scheme: Optional[str] = None,
    row_filter_stats: Optional[Dict[str, Any]] = None,
    index_space: str = "full",  # "full" or "dev"
) -> None:
    meta: Dict[str, Any] = {
        "scenario": scenario,
        "seed": int(seed),
        "split_type": split_type,
        "index_space": index_space,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_train_pos": int(y_train.sum()),
        "n_test_pos": int(y_test.sum()),
        "prevalence_train": float(y_train.mean()),
        "prevalence_test": float(y_test.mean()),
        "split_id_train": compute_split_id(train_idx),
        "split_id_test": compute_split_id(test_idx),
    }
    if val_idx is not None and y_val is not None and len(val_idx) > 0:
        meta.update({
            "n_val": int(len(val_idx)),
            "n_val_pos": int(y_val.sum()),
            "prevalence_val": float(y_val.mean()),
            "split_id_val": compute_split_id(val_idx),
        })
    if strat_scheme is not None:
        meta["stratification_scheme"] = strat_scheme
    if row_filter_stats is not None:
        meta["row_filters"] = row_filter_stats

    meta_path = os.path.join(outdir, f"{scenario}_split_meta_seed{seed}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"  Saved metadata: {meta_path}")


# ---------------- scenario definitions ----------------
SCENARIO_DEFINITIONS = {
    "IncidentOnly": {
        "positives": [INCIDENT_LABEL],
        "description": "Controls + Incident (prospective prediction, recommended)",
        "warning": None,
    },
    "IncidentPlusPrevalent": {
        "positives": [INCIDENT_LABEL, PREVALENT_LABEL],
        "description": "Controls + Incident + Prevalent",
        "warning": None,
    },
}


def _generate_scenario_splits(
    df: pd.DataFrame,
    scenario: str,
    outdir: str,
    mode: str,
    n_splits: int,
    test_size: float,
    val_size: float,
    holdout_size: float,
    seed_start: int,
    overwrite: bool,
    temporal_split: bool,
    temporal_col: str,
    train_control_per_case: Optional[float],
    eval_control_per_case: Optional[float],
    train_controls_incident_only: bool,
    prevalent_train_frac: float,
    prevalent_train_only: bool,
) -> None:
    """Generate splits for a single scenario."""
    scenario_def = SCENARIO_DEFINITIONS[scenario]
    positives = scenario_def["positives"]
    eval_case_labels = positives
    if prevalent_train_only and PREVALENT_LABEL in positives:
        eval_case_labels = [INCIDENT_LABEL]
    train_case_labels = positives
    if train_controls_incident_only and PREVALENT_LABEL in positives:
        train_case_labels = [INCIDENT_LABEL]

    log(f"\n{'='*60}")
    log(f"=== Preparing {scenario} scenario ({scenario_def['description']}) ===")
    log(f"{'='*60}")

    # Show warning for IncidentPlusPrevalent
    if scenario_def["warning"]:
        log(scenario_def["warning"])
    if prevalent_train_only and PREVALENT_LABEL in positives:
        log(f"  Prevalent handling: TRAIN only (frac={prevalent_train_frac:.2f})")

    # Filter to scenario
    keep_labels = [CONTROL_LABEL] + positives
    mask = df[TARGET_COL].isin(keep_labels)
    df_scenario_raw = df[mask].copy()
    log(f"  {scenario} (raw, pre row-filters): {len(df_scenario_raw):,} samples")

    # Apply training row filters (using shared helper from celiacML_faith.py)
    df_scenario, rf_stats = apply_row_filters(
        df_scenario_raw,
    )
    log("  Row-filter alignment (to match training dataset):")
    log("    - drop_uncertain_controls=True")
    log("    - dropna_meta_num=True")
    log(f"    - removed_uncertain_controls={rf_stats['n_removed_uncertain_controls']:,}")
    log(f"    - removed_dropna_meta_num={rf_stats['n_removed_dropna_meta_num']:,}")
    log(f"  {scenario} (post row-filters): {len(df_scenario):,} samples")

    if temporal_split:
        log(f"  Temporal ordering enabled (column={temporal_col})")
        order_idx = temporal_order_indices(df_scenario, temporal_col)
        df_scenario = df_scenario.iloc[order_idx].reset_index(drop=True)
        if len(df_scenario) > 0 and temporal_col in df_scenario.columns:
            log(f"    Earliest value: {df_scenario[temporal_col].iloc[0]}")
            log(f"    Latest value:   {df_scenario[temporal_col].iloc[-1]}")

    # Create outcome variable (1 for any positive label)
    y_full = df_scenario[TARGET_COL].isin(positives).astype(int).to_numpy()
    n_controls = (y_full == 0).sum()
    n_cases = (y_full == 1).sum()
    log(f"  Controls: {n_controls:,}")
    log(f"  Cases: {n_cases:,} ({y_full.mean()*100:.3f}%)")

    full_idx = np.arange(len(df_scenario))

    # ---------------- holdout mode ----------------
    if mode == "holdout":
        log(f"\n=== Creating holdout set ({holdout_size*100:.0f}% of post-filter scenario) ===")

        if temporal_split:
            n_holdout = int(round(holdout_size * len(full_idx)))
            n_holdout = min(max(1, n_holdout), max(1, len(full_idx) - 1))
            holdout_idx_global = full_idx[-n_holdout:]
            dev_idx_global = full_idx[:-n_holdout]
            sch_full = "temporal"
            y_holdout = y_full[holdout_idx_global]
        else:
            strata_full, sch_full = build_working_strata(df_scenario, min_count=2)
            log(f"  Holdout stratification: {sch_full}")
            dev_pos_unsorted, holdout_pos, _, y_holdout = train_test_split(
                full_idx,
                y_full,
                test_size=holdout_size,
                random_state=42,
                stratify=strata_full,
            )

            # Match celiacML_faith.py dev ordering: sorted ascending
            holdout_idx_global = np.array(holdout_pos, dtype=int)
            dev_mask = np.ones(len(full_idx), dtype=bool)
            dev_mask[holdout_idx_global] = False
            dev_idx_global = full_idx[dev_mask]

        df_dev = df_scenario.iloc[dev_idx_global].copy().reset_index(drop=True)
        y_dev = y_full[dev_idx_global]

        log(f"  Full dataset: {len(full_idx):,}")
        log(f"  Holdout set: {len(holdout_idx_global):,} (global idx space)")
        log(f"  Development set: {len(df_dev):,} (dev-local idx space)")
        log(f"  Holdout prevalence: {y_holdout.mean()*100:.3f}%")

        # Save HOLDOUT
        holdout_path = os.path.join(outdir, f"{scenario}_HOLDOUT_idx.csv")
        holdout_meta_path = os.path.join(outdir, f"{scenario}_HOLDOUT_meta.json")

        if (os.path.exists(holdout_path) or os.path.exists(holdout_meta_path)) and (not overwrite):
            raise FileExistsError(
                f"Holdout file(s) already exist in {outdir}. Use --overwrite to replace."
            )

        # Keep holdout indices global so they map directly to the filtered dataset
        # before celiacML_faith.py resets indices for the development cohort.
        pd.DataFrame({"idx": holdout_idx_global}).to_csv(holdout_path, index=False)
        log(f"  ✓ Saved holdout indices: {holdout_path}")

        holdout_meta: Dict[str, Any] = {
            "scenario": scenario,
            "split_type": "holdout",
            "seed": 42,
            "n_holdout": int(len(holdout_idx_global)),
            "n_holdout_pos": int(y_holdout.sum()),
            "prevalence_holdout": float(y_holdout.mean()),
            "split_id_holdout": compute_split_id(holdout_idx_global),
            "stratification_scheme": sch_full,
            "row_filters": rf_stats,
            "index_space": "full",
            "note": "NEVER use this set during development. Final evaluation only.",
        }
        if temporal_split:
            holdout_meta["temporal_split"] = True
            holdout_meta["temporal_col"] = temporal_col
            if len(holdout_idx_global) > 0 and temporal_col in df_scenario.columns:
                holdout_meta["temporal_start_value"] = str(df_scenario.iloc[holdout_idx_global[0]][temporal_col])
                holdout_meta["temporal_end_value"] = str(df_scenario.iloc[holdout_idx_global[-1]][temporal_col])
        if scenario_def["warning"]:
            holdout_meta["reverse_causality_warning"] = scenario_def["warning"]

        with open(holdout_meta_path, "w") as f:
            json.dump(holdout_meta, f, indent=2)
        log(f"  ✓ Saved holdout metadata: {holdout_meta_path}")

        # After removing holdout rows we operate in dev-local space, matching the
        # expectation of celiacML_faith.py when it loads train/test CSV files.
        df_work = df_dev
        y_work = y_dev
        index_space_for_splits = "dev"

    # ---------------- development mode (no holdout) ----------------
    else:
        log("\n=== Development mode (no holdout) ===")
        df_work = df_scenario.copy()
        y_work = y_full
        index_space_for_splits = "full"

    # Base set for splitting (optionally exclude prevalent from VAL/TEST)
    if prevalent_train_only and PREVALENT_LABEL in positives:
        base_mask = df_work[TARGET_COL].isin([CONTROL_LABEL, INCIDENT_LABEL])
        df_base = df_work[base_mask].copy()
        base_idx = df_base.index.to_numpy(dtype=int)
        y_base = (df_base[TARGET_COL] == INCIDENT_LABEL).astype(int).to_numpy()
        log(f"  Prevalent excluded from VAL/TEST. Base split set: {len(df_base):,}")
    else:
        df_base = df_work
        base_idx = np.arange(len(df_work), dtype=int)
        y_base = y_work

    # Build strata on base set
    if temporal_split:
        strata_base, sch_work = None, "temporal"
    else:
        strata_base, sch_work = build_working_strata(df_base, min_count=2)
    log(f"\n=== Stratification for train/val/test: {sch_work} ===")
    log(f"[INFO] Split index space: {index_space_for_splits}")

    log(f"\n=== Generating {n_splits} split(s) (test_size={test_size}, val_size={val_size}) ===")

    for i in range(n_splits):
        seed = seed_start + i
        log(f"\n--- Split {i+1}/{n_splits} (seed={seed}) ---")

        work_idx = np.array(base_idx, dtype=int)
        if temporal_split:
            if len(work_idx) < 2:
                raise ValueError("Temporal split mode requires at least 2 samples after filtering.")
            n_total = len(work_idx)
            n_test = int(round(test_size * n_total))
            n_val = int(round(val_size * n_total)) if val_size > 0 else 0
            n_test = min(max(1, n_test), max(1, n_total - 1))
            n_val = min(max(0, n_val), max(0, n_total - n_test - 1))
            n_train = n_total - n_test - n_val
            if n_train < 1:
                raise ValueError("Temporal split produced empty TRAIN. Reduce --val_size/--test_size.")
            idx_train = work_idx[:n_train]
            idx_val = work_idx[n_train:n_train + n_val] if n_val > 0 else np.array([], dtype=int)
            idx_test = work_idx[n_train + n_val:]
            y_train = y_base[np.isin(work_idx, idx_train)]
            y_val = y_base[np.isin(work_idx, idx_val)] if n_val > 0 else np.array([], dtype=int)
            y_test = y_base[np.isin(work_idx, idx_test)]
        else:
            if val_size and val_size > 0:
                temp_size = float(val_size + test_size)
                idx_train, idx_temp, y_train, y_temp = train_test_split(
                    work_idx,
                    y_base,
                    test_size=temp_size,
                    random_state=seed,
                    stratify=strata_base,
                )
                strata_temp = strata_base.loc[idx_temp]
                rel_test = float(test_size) / temp_size
                idx_val, idx_test, y_val, y_test = train_test_split(
                    idx_temp,
                    y_temp,
                    test_size=rel_test,
                    random_state=seed,
                    stratify=strata_temp,
                )
            else:
                idx_train, idx_test, y_train, y_test = train_test_split(
                    work_idx,
                    y_base,
                    test_size=test_size,
                    random_state=seed,
                    stratify=strata_base,
                )
                idx_val = np.array([], dtype=int)
                y_val = np.array([], dtype=int)

        idx_train = np.sort(idx_train.astype(int))
        idx_val = np.sort(idx_val.astype(int))
        idx_test = np.sort(idx_test.astype(int))

        log(f"  Train: {len(idx_train):,} samples ({int(y_train.sum())} cases, {y_train.mean()*100:.3f}%)")
        if len(idx_val) > 0:
            log(f"  Val:   {len(idx_val):,} samples ({int(y_val.sum())} cases, {y_val.mean()*100:.3f}%)")
        log(f"  Test:  {len(idx_test):,} samples ({int(y_test.sum())} cases, {y_test.mean()*100:.3f}%)")

        rng = np.random.RandomState(seed + 1337)

        # Add prevalent to TRAIN only (optional)
        if prevalent_train_only and PREVALENT_LABEL in positives:
            idx_prev = df_work.index[df_work[TARGET_COL] == PREVALENT_LABEL].to_numpy(dtype=int)
            if prevalent_train_frac >= 1.0:
                idx_prev_keep = idx_prev
            elif prevalent_train_frac <= 0.0 or len(idx_prev) == 0:
                idx_prev_keep = np.array([], dtype=int)
            else:
                n_keep = int(round(prevalent_train_frac * len(idx_prev)))
                n_keep = min(len(idx_prev), max(1, n_keep))
                idx_prev_keep = rng.choice(idx_prev, size=n_keep, replace=False)
            if len(idx_prev_keep) > 0:
                idx_train = np.sort(np.concatenate([idx_train, idx_prev_keep]).astype(int))
                log(f"  Train: added prevalent={len(idx_prev_keep):,} (frac={prevalent_train_frac:.2f})")

        # Downsample controls in TRAIN to case:control ratio
        idx_train = _downsample_controls(
            idx_train,
            df_work,
            case_labels=train_case_labels,
            controls_per_case=train_control_per_case,
            rng=rng,
            label="train",
        )

        y_train = (df_work.loc[idx_train, TARGET_COL].isin(positives)).astype(int).to_numpy()

        if eval_control_per_case is not None and eval_control_per_case > 0:
            if len(idx_val) > 0:
                idx_val = _downsample_controls(
                    idx_val,
                    df_work,
                    case_labels=eval_case_labels,
                    controls_per_case=eval_control_per_case,
                    rng=rng,
                    label="val",
                )
            idx_test = _downsample_controls(
                idx_test,
                df_work,
                case_labels=eval_case_labels,
                controls_per_case=eval_control_per_case,
                rng=rng,
                label="test",
            )

        y_val = (df_work.loc[idx_val, TARGET_COL].isin(eval_case_labels)).astype(int).to_numpy() if len(idx_val) > 0 else np.array([], dtype=int)
        y_test = (df_work.loc[idx_test, TARGET_COL].isin(eval_case_labels)).astype(int).to_numpy()

        log(f"  Final Train: {len(idx_train):,} samples ({int(y_train.sum())} cases, {y_train.mean()*100:.3f}%)")
        if len(idx_val) > 0:
            log(f"  Final Val:   {len(idx_val):,} samples ({int(y_val.sum())} cases, {y_val.mean()*100:.3f}%)")
        log(f"  Final Test:  {len(idx_test):,} samples ({int(y_test.sum())} cases, {y_test.mean()*100:.3f}%)")

        suffix = f"_seed{seed}" if n_splits > 1 else ""
        train_path = os.path.join(outdir, f"{scenario}_train_idx{suffix}.csv")
        val_path = os.path.join(outdir, f"{scenario}_val_idx{suffix}.csv")
        test_path = os.path.join(outdir, f"{scenario}_test_idx{suffix}.csv")

        if (os.path.exists(train_path) or os.path.exists(test_path) or os.path.exists(val_path)) and (not overwrite):
            raise FileExistsError(
                f"Split file(s) already exist. Use --overwrite to replace:\n  {train_path}\n  {test_path}"
            )

        pd.DataFrame({"idx": idx_train}).to_csv(train_path, index=False)
        if len(idx_val) > 0:
            pd.DataFrame({"idx": idx_val}).to_csv(val_path, index=False)
        pd.DataFrame({"idx": idx_test}).to_csv(test_path, index=False)

        log(f"  ✓ Saved: {train_path}")
        if len(idx_val) > 0:
            log(f"  ✓ Saved: {val_path}")
        log(f"  ✓ Saved: {test_path}")

        save_split_metadata(
            outdir=outdir,
            scenario=scenario,
            seed=seed,
            train_idx=idx_train,
            test_idx=idx_test,
            y_train=y_train,
            y_test=y_test,
            val_idx=idx_val,
            y_val=y_val,
            split_type="development",
            strat_scheme=sch_work,
            row_filter_stats=rf_stats,
            index_space=index_space_for_splits,
        )
        if temporal_split:
            meta_path = os.path.join(outdir, f"{scenario}_split_meta_seed{seed}.json")
            with open(meta_path, "r") as f:
                current_meta = json.load(f)
            current_meta["temporal_split"] = True
            current_meta["temporal_col"] = temporal_col
            if len(idx_test) > 0 and temporal_col in df_work.columns:
                current_meta["temporal_test_start_value"] = str(df_work.iloc[idx_test[0]][temporal_col])
                current_meta["temporal_test_end_value"] = str(df_work.iloc[idx_test[-1]][temporal_col])
            if len(idx_train) > 0 and temporal_col in df_work.columns:
                current_meta["temporal_train_end_value"] = str(df_work.iloc[idx_train[-1]][temporal_col])
            with open(meta_path, "w") as f:
                json.dump(current_meta, f, indent=2)

    if mode == "holdout":
        log("\nIMPORTANT: Holdout idx is in FULL index space.")
        log("           Train/Test idx files are in DEV index space.")


# ---------------- main split generation ----------------
def generate_splits_optimized(
    infile: str,
    outdir: str,
    scenarios: Optional[List[str]] = None,
    mode: str = "development",
    n_splits: int = 1,
    test_size: float = 0.30,
    val_size: float = 0.0,
    holdout_size: float = 0.30,
    seed_start: int = 0,
    overwrite: bool = False,
    temporal_split: bool = False,
    temporal_col: str = CED_DATE_COL,
    train_control_per_case: Optional[float] = None,
    eval_control_per_case: Optional[float] = None,
    train_controls_incident_only: bool = False,
    prevalent_train_frac: float = 1.0,
    prevalent_train_only: bool = False,
) -> None:
    """
    Generate train/test splits for one or more scenarios.

    Args:
        infile: Path to input CSV file
        outdir: Output directory for split files
        scenarios: List of scenarios to generate (default: ["IncidentOnly"])
        mode: "development" or "holdout"
        n_splits: Number of repeated splits per scenario
        test_size: Fraction for test set
        holdout_size: Fraction for holdout set (if mode="holdout")
        seed_start: Starting seed for repeated splits
        overwrite: Overwrite existing files
    """
    if scenarios is None:
        scenarios = ["IncidentOnly"]

    if val_size < 0 or test_size <= 0:
        raise ValueError("val_size must be >=0 and test_size must be >0.")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0 for non-empty TRAIN.")

    # Validate scenarios
    for scen in scenarios:
        if scen not in SCENARIO_DEFINITIONS:
            raise ValueError(
                f"Unknown scenario: {scen}. "
                f"Available: {list(SCENARIO_DEFINITIONS.keys())}"
            )

    log(f"Loading data from: {infile}")
    df = pd.read_csv(infile, low_memory=False)
    log(f"  Loaded {len(df):,} samples")

    required = [ID_COL, TARGET_COL] + META_NUM + META_CAT
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Generate splits for each scenario
    for scenario in scenarios:
        _generate_scenario_splits(
            df=df,
            scenario=scenario,
            outdir=outdir,
            mode=mode,
            n_splits=n_splits,
            test_size=test_size,
            val_size=val_size,
            holdout_size=holdout_size,
            seed_start=seed_start,
            overwrite=overwrite,
            temporal_split=temporal_split,
            temporal_col=temporal_col,
            train_control_per_case=train_control_per_case,
            eval_control_per_case=eval_control_per_case,
            train_controls_incident_only=train_controls_incident_only,
            prevalent_train_frac=prevalent_train_frac,
            prevalent_train_only=prevalent_train_only,
        )

    log(f"\n{'='*60}")
    log("=== All split generation complete ===")
    log(f"{'='*60}")
    log(f"Output directory: {outdir}")
    log(f"Scenarios generated: {scenarios}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate optimized train/test splits with holdout validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate IncidentOnly splits (recommended)
  python save_splits.py --infile data.csv --outdir splits/

  # Generate both scenarios with 10 repeated splits
  python save_splits.py --infile data.csv --outdir splits/ \\
      --scenarios IncidentOnly IncidentPlusPrevalent --n_splits 10

  # Generate with holdout set
  python save_splits.py --infile data.csv --outdir splits/ --mode holdout
        """
    )

    parser.add_argument("--infile", required=True, help="Input CSV file")
    parser.add_argument("--outdir", required=True, help="Output directory for splits")

    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(SCENARIO_DEFINITIONS.keys()),
        default=["IncidentOnly"],
        help="Scenarios to generate splits for (default: IncidentOnly only)",
    )
    parser.add_argument(
        "--mode",
        choices=["development", "holdout"],
        default="development",
        help="development: use full scenario. holdout: set aside holdout, then split dev set",
    )
    parser.add_argument("--n_splits", type=int, default=1, help="Number of repeated splits")
    parser.add_argument("--test_size", type=float, default=0.30, help="Fraction of working set for testing")
    parser.add_argument("--val_size", type=float, default=0.0, help="Fraction of working set for validation (0 disables)")
    parser.add_argument("--holdout_size", type=float, default=0.30, help="Fraction of full scenario for holdout")
    parser.add_argument("--seed_start", type=int, default=0, help="Starting seed for repeated splits")

    parser.add_argument("--temporal_split", action="store_true",
                        help=f"Enable chronological splits using --temporal_col (default column: {CED_DATE_COL}).")
    parser.add_argument("--temporal_col", type=str, default=CED_DATE_COL,
                        help="Column used for chronological ordering when --temporal_split is set.")

    parser.add_argument("--train_control_per_case", type=float, default=None,
                        help="Downsample TRAIN controls to N controls per Incident case (e.g., 5 for 1:5).")
    parser.add_argument("--eval_control_per_case", type=float, default=None,
                        help="Downsample VAL/TEST controls to N controls per Incident case.")
    parser.add_argument("--train_controls_incident_only", action="store_true",
                        help="If set, TRAIN control downsampling uses Incident cases only (even if Prevalent are in TRAIN).")
    parser.add_argument("--prevalent_train_frac", type=float, default=1.0,
                        help="Fraction of Prevalent cases to include in TRAIN (only if scenario includes Prevalent).")
    parser.add_argument("--prevalent_train_only", action="store_true",
                        help="If set, exclude Prevalent from VAL/TEST and include only in TRAIN.")

    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing holdout/split files in outdir")

    args = parser.parse_args()

    if not os.path.isfile(args.infile):
        raise FileNotFoundError(f"Input file not found: {args.infile}")

    os.makedirs(args.outdir, exist_ok=True)

    generate_splits_optimized(
        infile=args.infile,
        outdir=args.outdir,
        scenarios=args.scenarios,
        mode=args.mode,
        n_splits=args.n_splits,
        test_size=args.test_size,
        val_size=args.val_size,
        holdout_size=args.holdout_size,
        seed_start=args.seed_start,
        overwrite=bool(args.overwrite),
        temporal_split=bool(args.temporal_split),
        temporal_col=str(args.temporal_col),
        train_control_per_case=args.train_control_per_case,
        eval_control_per_case=args.eval_control_per_case,
        train_controls_incident_only=bool(args.train_controls_incident_only),
        prevalent_train_frac=float(args.prevalent_train_frac),
        prevalent_train_only=bool(args.prevalent_train_only),
    )


if __name__ == "__main__":
    main()
