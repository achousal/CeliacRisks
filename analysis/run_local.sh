#!/usr/bin/env bash
#============================================================
# run_local.sh
#
# Config-driven local pipeline orchestration
# All settings in configs/pipeline_local.yaml
#
# IMPORTANT: This script must be run from the analysis/ directory.
#            Relative paths in configs assume this working directory.
#
# Usage:
#   cd analysis/                      # REQUIRED: Run from analysis/
#   ./run_local.sh                    # Use pipeline_local.yaml
#   PIPELINE_CONFIG=custom.yaml ./run_local.sh  # Custom config
#   DRY_RUN=1 ./run_local.sh          # Override dry_run
#============================================================

set -euo pipefail
IFS=$'\n\t'

log() { echo "[$(date +"%F %T")] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

#==============================================================
# LOAD PIPELINE CONFIG
#==============================================================
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-${BASE_DIR}/configs/pipeline_local.yaml}"

[[ -f "${PIPELINE_CONFIG}" ]] || die "Pipeline config not found: ${PIPELINE_CONFIG}"

# Helper: extract YAML value (simple grep-based parser for flat keys)
get_yaml() {
  local file="$1"
  local key="$2"
  grep "^  ${key}:" "${file}" | sed 's/^[^:]*: *//' | tr -d '"' | tr -d "'" | sed 's/ *#.*//'
}

get_yaml_list() {
  local file="$1"
  local key="$2"
  local line
  line=$(grep "^  ${key}:" "${file}" | sed 's/^[^:]*: *//' | sed 's/ *#.*//')

  # Handle inline array format: [item1, item2]
  if [[ "${line}" =~ ^\[.*\]$ ]]; then
    echo "${line}" | tr -d '[]' | tr -d ' '
  # Handle multiline list format with dashes
  else
    grep -A10 "^  ${key}:" "${file}" | grep "^ *- " | sed 's/^ *- *//' | tr '\n' ',' | sed 's/,$//'
  fi
}

# Get nested YAML value (e.g., "ensemble" "enabled")
get_yaml_nested() {
  local file="$1"
  local section="$2"
  local key="$3"
  awk -v section="${section}:" -v key="  ${key}:" '
    $0 ~ "^"section { in_section=1; next }
    in_section && /^[a-zA-Z]/ { in_section=0 }
    in_section && $0 ~ key { gsub(/^[^:]*: */, ""); gsub(/ *#.*/, ""); gsub(/["'"'"']/, ""); print; exit }
  ' "${file}"
}

# Get nested YAML list (e.g., "ensemble" "base_models")
get_yaml_nested_list() {
  local file="$1"
  local section="$2"
  local key="$3"
  local line
  line=$(get_yaml_nested "${file}" "${section}" "${key}")

  # Handle inline array format: [item1, item2]
  if [[ "${line}" =~ ^\[.*\]$ ]]; then
    echo "${line}" | tr -d '[]' | tr -d ' '
  else
    echo ""
  fi
}

# Read paths
INFILE=$(get_yaml "${PIPELINE_CONFIG}" "infile")
SPLITS_DIR=$(get_yaml "${PIPELINE_CONFIG}" "splits_dir")
RESULTS_DIR=$(get_yaml "${PIPELINE_CONFIG}" "results_dir")
LOGS_DIR=$(get_yaml "${PIPELINE_CONFIG}" "logs_dir")

# Convert relative paths to absolute
INFILE="${BASE_DIR}/${INFILE}"
SPLITS_DIR="${BASE_DIR}/${SPLITS_DIR}"
RESULTS_DIR="${BASE_DIR}/${RESULTS_DIR}"
LOGS_DIR="${BASE_DIR}/${LOGS_DIR}"

# Generate run ID early (needed for logs directory)
RUN_ID=$(date +"%Y%m%d_%H%M%S")

# Create training logs subdirectory
RUN_LOGS_DIR="${LOGS_DIR}/training/run_${RUN_ID}"
mkdir -p "${RUN_LOGS_DIR}"

# Read configs
SPLITS_CONFIG=$(get_yaml "${PIPELINE_CONFIG}" "splits")
TRAINING_CONFIG=$(get_yaml "${PIPELINE_CONFIG}" "training")
SPLITS_CONFIG="${BASE_DIR}/${SPLITS_CONFIG}"
TRAINING_CONFIG="${BASE_DIR}/${TRAINING_CONFIG}"

# Read execution settings
MODELS_STR=$(get_yaml_list "${PIPELINE_CONFIG}" "models")
N_BOOT=$(get_yaml "${PIPELINE_CONFIG}" "n_boot")
OVERWRITE_SPLITS_CFG=$(get_yaml "${PIPELINE_CONFIG}" "overwrite_splits")
DRY_RUN_CFG=$(get_yaml "${PIPELINE_CONFIG}" "dry_run")

# Environment variable overrides
DRY_RUN="${DRY_RUN:-${DRY_RUN_CFG}}"
OVERWRITE_SPLITS="${OVERWRITE_SPLITS:-${OVERWRITE_SPLITS_CFG}}"
RUN_MODELS="${RUN_MODELS:-${MODELS_STR}}"

# Convert true/false/1/0 to normalized 1/0
normalize_bool() {
  local val="${1:-}"
  val="$(printf '%s' "${val}" | tr '[:upper:]' '[:lower:]')"
  case "${val}" in
    1|true|yes|y) echo 1 ;;
    0|false|no|n|'') echo 0 ;;
    *) echo 0 ;;
  esac
}
DRY_RUN="$(normalize_bool "${DRY_RUN}")"
OVERWRITE_SPLITS="$(normalize_bool "${OVERWRITE_SPLITS}")"

#==============================================================
# ENVIRONMENT DETECTION
#==============================================================
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  ENV_TYPE="conda (${CONDA_DEFAULT_ENV})"
elif [[ -f "${BASE_DIR}/venv/bin/activate" ]]; then
  log "Activating venv..."
  source "${BASE_DIR}/venv/bin/activate"
  ENV_TYPE="venv"
else
  die "No Python environment found. Activate conda or run: bash scripts/hpc_setup.sh"
fi

command -v ced &> /dev/null || die "ced command not found. Install: pip install -e ."

#==============================================================
# SETUP
#==============================================================
cd "${BASE_DIR}"
mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}" "${SPLITS_DIR}"

[[ -f "${INFILE}" ]] || die "Input file not found: ${INFILE}"
[[ -f "${SPLITS_CONFIG}" ]] || die "Splits config not found: ${SPLITS_CONFIG}"
[[ -f "${TRAINING_CONFIG}" ]] || die "Training config not found: ${TRAINING_CONFIG}"

# Extract split settings
N_SPLITS=$(grep "^n_splits:" "${SPLITS_CONFIG}" | awk '{print $2}')
SEED_START=$(grep "^seed_start:" "${SPLITS_CONFIG}" | awk '{print $2}')

# Extract ensemble settings from training config
ENSEMBLE_ENABLED=$(get_yaml_nested "${TRAINING_CONFIG}" "ensemble" "enabled")
ENSEMBLE_BASE_MODELS=$(get_yaml_nested_list "${TRAINING_CONFIG}" "ensemble" "base_models")
[[ "${ENSEMBLE_ENABLED}" == "true" ]] && ENSEMBLE_ENABLED=1 || ENSEMBLE_ENABLED=0

log "============================================"
log "CeD-ML Pipeline (Config-Driven)"
log "============================================"
log "Pipeline config: ${PIPELINE_CONFIG}"
log "Environment: ${ENV_TYPE}"
log "Run ID: ${RUN_ID}"
log "Input: ${INFILE}"
log "Splits: ${SPLITS_DIR} (${N_SPLITS} splits, seeds ${SEED_START}-$((SEED_START + N_SPLITS - 1)))"
log "Results: ${RESULTS_DIR}"
log "Logs: ${RUN_LOGS_DIR}"
log "Models: ${RUN_MODELS}"
log "Bootstrap: ${N_BOOT}"
log "Ensemble: ${ENSEMBLE_ENABLED} (base: ${ENSEMBLE_BASE_MODELS:-none})"
log "Dry run: ${DRY_RUN}"
log "============================================"

#==============================================================
# STEP 1: GENERATE SPLITS
#==============================================================
log "Step 1/4: Generate splits"

  SPLITS_EXIST=0
  if ls "${SPLITS_DIR}"/train_idx_*_seed*.csv 1>/dev/null 2>&1 && \
     ls "${SPLITS_DIR}"/val_idx_*_seed*.csv 1>/dev/null 2>&1; then
    SPLITS_EXIST=1
  fi

  if [[ ${SPLITS_EXIST} -eq 1 && ${OVERWRITE_SPLITS} -eq 0 ]]; then
    log "Splits exist. Set overwrite_splits: true in config or OVERWRITE_SPLITS=1"
  else
    if [[ ${DRY_RUN} -eq 1 ]]; then
      log "[DRY RUN] Would run: ced save-splits"
    else
      log "Generating splits..."
      OVERWRITE_FLAG=""
      [[ ${OVERWRITE_SPLITS} -eq 1 ]] && OVERWRITE_FLAG="--overwrite"

      ced save-splits \
        --config "${SPLITS_CONFIG}" \
        --infile "${INFILE}" \
        --outdir "${SPLITS_DIR}" \
        ${OVERWRITE_FLAG}

      log "[OK] Splits generated"
    fi
  fi

#==============================================================
# STEP 2: TRAIN MODELS (SEQUENTIAL, PER-SPLIT)
#==============================================================
log "Step 2/4: Train models (${N_SPLITS} splits)"

  IFS=',' read -r -a MODEL_ARRAY <<< "${RUN_MODELS}"
  COMPLETED_RUNS=()
  SEED_END=$((SEED_START + N_SPLITS - 1))

  for SEED in $(seq ${SEED_START} ${SEED_END}); do
    log "--- Split seed ${SEED} ($(( SEED - SEED_START + 1 ))/${N_SPLITS}) ---"

    for MODEL in "${MODEL_ARRAY[@]}"; do
      MODEL=$(echo "${MODEL}" | xargs)
      [[ -z "${MODEL}" ]] && continue

      JOB_NAME="CeD_${MODEL}_seed${SEED}"

      if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[DRY RUN] Would train: ${MODEL} (seed ${SEED})"
        COMPLETED_RUNS+=("${MODEL}:${SEED}")
      else
        log "Training ${MODEL} (seed ${SEED})..."
        START_TIME=$(date +%s)

        LOG_FILE="${RUN_LOGS_DIR}/${JOB_NAME}.log"

        if ced train \
          --config "${TRAINING_CONFIG}" \
          --model "${MODEL}" \
          --infile "${INFILE}" \
          --split-dir "${SPLITS_DIR}" \
          --outdir "${RESULTS_DIR}/${MODEL}" \
          --split-seed "${SEED}" \
          --override run_id="${RUN_ID}" \
          2>&1 | tee "${LOG_FILE}"; then

          ELAPSED=$(($(date +%s) - START_TIME))
          log "  [OK] ${MODEL} seed ${SEED} (${ELAPSED}s)"
          COMPLETED_RUNS+=("${MODEL}:${SEED}")
        else
          log "  [FAIL] ${MODEL} seed ${SEED} (see ${LOG_FILE})"
        fi
      fi
    done
  done

  log "Completed ${#COMPLETED_RUNS[@]} run(s)"

#==============================================================
# STEP 2.5: TRAIN ENSEMBLE (if enabled)
#==============================================================
  if [[ ${ENSEMBLE_ENABLED} -eq 1 ]]; then
    log "Step 2.5/4: Train ensemble (stacking)"

    # Determine base models: use config or fall back to trained models
    if [[ -n "${ENSEMBLE_BASE_MODELS}" ]]; then
      ENSEMBLE_MODELS="${ENSEMBLE_BASE_MODELS}"
    else
      ENSEMBLE_MODELS="${RUN_MODELS}"
    fi

    for SEED in $(seq ${SEED_START} ${SEED_END}); do
      JOB_NAME="CeD_ENSEMBLE_seed${SEED}"

      if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[DRY RUN] Would train ensemble (seed ${SEED}) with base models: ${ENSEMBLE_MODELS}"
      else
        log "Training ensemble (seed ${SEED})..."
        START_TIME=$(date +%s)

        LOG_FILE="${RUN_LOGS_DIR}/${JOB_NAME}.log"

        # Check if base models have OOF predictions
        BASE_MODELS_READY=1
        IFS=',' read -r -a ENSEMBLE_MODEL_ARRAY <<< "${ENSEMBLE_MODELS}"
        for BASE_MODEL in "${ENSEMBLE_MODEL_ARRAY[@]}"; do
          BASE_MODEL=$(echo "${BASE_MODEL}" | xargs)
          OOF_PATH="${RESULTS_DIR}/${BASE_MODEL}/run_${RUN_ID}/split_seed${SEED}/preds/train_oof/train_oof__${BASE_MODEL}.csv"
          if [[ ! -f "${OOF_PATH}" ]]; then
            log "  [WARN] Missing OOF for ${BASE_MODEL} seed ${SEED}: ${OOF_PATH}"
            BASE_MODELS_READY=0
          fi
        done

        if [[ ${BASE_MODELS_READY} -eq 1 ]]; then
          if ced train-ensemble \
            --config "${TRAINING_CONFIG}" \
            --results-dir "${RESULTS_DIR}" \
            --base-models "${ENSEMBLE_MODELS}" \
            --split-seed "${SEED}" \
            --outdir "${RESULTS_DIR}/ENSEMBLE/run_${RUN_ID}/split_seed${SEED}" \
            2>&1 | tee "${LOG_FILE}"; then

            ELAPSED=$(($(date +%s) - START_TIME))
            log "  [OK] ENSEMBLE seed ${SEED} (${ELAPSED}s)"
            COMPLETED_RUNS+=("ENSEMBLE:${SEED}")
          else
            log "  [FAIL] ENSEMBLE seed ${SEED} (see ${LOG_FILE})"
          fi
        else
          log "  [SKIP] ENSEMBLE seed ${SEED}: base model OOF predictions not ready"
        fi
      fi
    done
  else
    log "Step 2.5/4: Ensemble training disabled (ensemble.enabled: false)"
  fi

#==============================================================
# STEP 3: SAVE RUN METADATA
#==============================================================
log "Step 3/4: Save run metadata"

if [[ ${DRY_RUN} -ne 1 ]]; then
  SEEDS_JSON="["
  for SEED in $(seq ${SEED_START} ${SEED_END:-$((SEED_START + N_SPLITS - 1))}); do
    [[ "${SEEDS_JSON}" != "[" ]] && SEEDS_JSON="${SEEDS_JSON},"
    SEEDS_JSON="${SEEDS_JSON}${SEED}"
  done
  SEEDS_JSON="${SEEDS_JSON}]"

  IFS=',' read -r -a MODEL_ARRAY_META <<< "${RUN_MODELS}"
  for MODEL in "${MODEL_ARRAY_META[@]}"; do
    MODEL=$(echo "${MODEL}" | xargs)
    [[ -z "${MODEL}" ]] && continue

    MODEL_RUN_DIR="${RESULTS_DIR}/${MODEL}/run_${RUN_ID}"
    if [[ -d "${MODEL_RUN_DIR}" ]]; then
      cat > "${MODEL_RUN_DIR}/run_metadata.json" << EOF
{
  "n_splits": ${N_SPLITS},
  "seed_start": ${SEED_START},
  "seeds": ${SEEDS_JSON},
  "model": "${MODEL}",
  "timestamp": "$(date -Iseconds)",
  "environment": "local",
  "pipeline_config": "${PIPELINE_CONFIG}",
  "splits_dir": "${SPLITS_DIR}",
  "training_config": "${TRAINING_CONFIG}",
  "infile": "${INFILE}"
}
EOF
      log "Metadata saved for ${MODEL}"
    else
      log "[SKIP] Model directory not found: ${MODEL_RUN_DIR}"
    fi
  done
fi

#==============================================================
# STEP 4: POSTPROCESSING / AGGREGATION
#==============================================================
log "Step 4/4: Aggregate results (per model)"

if [[ ${DRY_RUN} -eq 1 ]]; then
  log "[DRY RUN] Would run: ced aggregate-splits for each model"
else
  COMPLETED_COUNT=$(find "${RESULTS_DIR}" -path "*/split_seed*/core/test_metrics.csv" 2>/dev/null | wc -l)
  log "Found ${COMPLETED_COUNT} completed split run(s)"

  if [[ ${COMPLETED_COUNT} -gt 0 ]]; then
    # Build list of models to aggregate (include ENSEMBLE if trained)
    MODELS_TO_AGG="${RUN_MODELS}"
    if [[ ${ENSEMBLE_ENABLED} -eq 1 ]]; then
      MODELS_TO_AGG="${MODELS_TO_AGG},ENSEMBLE"
    fi

    IFS=',' read -r -a MODEL_ARRAY_AGG <<< "${MODELS_TO_AGG}"
    for MODEL in "${MODEL_ARRAY_AGG[@]}"; do
      MODEL=$(echo "${MODEL}" | xargs)
      [[ -z "${MODEL}" ]] && continue

      MODEL_DIR="${RESULTS_DIR}/${MODEL}/run_${RUN_ID}"
      if [[ -d "${MODEL_DIR}" ]]; then
        log "Aggregating ${MODEL}..."
        ced aggregate-splits --results-dir "${MODEL_DIR}" --n-boot "${N_BOOT}"
        log "  [OK] ${MODEL} aggregated"
      else
        log "  [SKIP] ${MODEL} run directory not found: ${MODEL_DIR}"
      fi
    done
    log "[OK] Aggregation complete"
  else
    log "No completed runs. Skipping aggregation."
  fi
fi

#==============================================================
# SUMMARY
#==============================================================
log "============================================"
log "Pipeline complete"
log "============================================"
log "Run ID: ${RUN_ID}"
log "Results: ${RESULTS_DIR}"
log "Aggregated: ${RESULTS_DIR}/<model>/run_${RUN_ID}/aggregated/"
log ""
log "Next steps:"
log "  - Edit config: vim ${PIPELINE_CONFIG}"
log "  - Re-aggregate: ced aggregate-splits --results-dir ${RESULTS_DIR}/<model>/run_${RUN_ID}"
log "  - View results: ls ${RESULTS_DIR}/<model>/run_${RUN_ID}/aggregated/"
log "============================================"
