#!/usr/bin/env bash
#============================================================
# run_local.sh
#
# Config-driven local pipeline orchestration
# All settings in configs/pipeline_local.yaml
#
# Usage:
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

# Convert true/false to 1/0
[[ "${DRY_RUN}" == "true" ]] && DRY_RUN=1 || DRY_RUN=0
[[ "${OVERWRITE_SPLITS}" == "true" ]] && OVERWRITE_SPLITS=1 || OVERWRITE_SPLITS=0

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

log "============================================"
log "CeD-ML Pipeline (Config-Driven)"
log "============================================"
log "Pipeline config: ${PIPELINE_CONFIG}"
log "Environment: ${ENV_TYPE}"
log "Input: ${INFILE}"
log "Splits: ${SPLITS_DIR} (${N_SPLITS} splits, seeds ${SEED_START}-$((SEED_START + N_SPLITS - 1)))"
log "Results: ${RESULTS_DIR}"
log "Models: ${RUN_MODELS}"
log "Bootstrap: ${N_BOOT}"
log "Dry run: ${DRY_RUN}"
log "============================================"

#==============================================================
# STEP 1: GENERATE SPLITS
#==============================================================
log "Step 1/3: Generate splits"

SPLITS_EXIST=0
if ls "${SPLITS_DIR}"/*_train_idx_seed0.csv 1>/dev/null 2>&1 && \
   ls "${SPLITS_DIR}"/*_val_idx_seed0.csv 1>/dev/null 2>&1; then
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
log "Step 2/3: Train models (${N_SPLITS} splits)"

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

        LOG_FILE="${LOGS_DIR}/${JOB_NAME}_$(date +%Y%m%d_%H%M%S).log"

        if ced train \
          --config "${TRAINING_CONFIG}" \
          --model "${MODEL}" \
          --infile "${INFILE}" \
          --split-dir "${SPLITS_DIR}" \
          --outdir "${RESULTS_DIR}" \
          --split-seed "${SEED}" \
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

  MODELS_JSON="["
  IFS=',' read -r -a MODEL_ARRAY_META <<< "${RUN_MODELS}"
  for MODEL in "${MODEL_ARRAY_META[@]}"; do
    MODEL=$(echo "${MODEL}" | xargs)
    [[ "${MODELS_JSON}" != "[" ]] && MODELS_JSON="${MODELS_JSON},"
    MODELS_JSON="${MODELS_JSON}\"${MODEL}\""
  done
  MODELS_JSON="${MODELS_JSON}]"

  cat > "${RESULTS_DIR}/run_metadata.json" << EOF
{
  "n_splits": ${N_SPLITS},
  "seed_start": ${SEED_START},
  "seeds": ${SEEDS_JSON},
  "models": ${MODELS_JSON},
  "timestamp": "$(date -Iseconds)",
  "environment": "local",
  "pipeline_config": "${PIPELINE_CONFIG}",
  "splits_dir": "${SPLITS_DIR}",
  "training_config": "${TRAINING_CONFIG}",
  "infile": "${INFILE}"
}
EOF
  log "Metadata saved"
fi

#==============================================================
# STEP 4: POSTPROCESSING / AGGREGATION
#==============================================================
log "Step 4/4: Aggregate results"

if [[ ${DRY_RUN} -eq 1 ]]; then
  log "[DRY RUN] Would run: ced aggregate-splits"
else
  COMPLETED_COUNT=$(find "${RESULTS_DIR}" -path "*/split_seed*/core/test_metrics.csv" 2>/dev/null | wc -l)
  log "Found ${COMPLETED_COUNT} completed split run(s)"

  if [[ ${COMPLETED_COUNT} -gt 0 ]]; then
    log "Aggregating..."
    ced aggregate-splits --results-dir "${RESULTS_DIR}" --n-boot "${N_BOOT}"
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
log "Results: ${RESULTS_DIR}"
log "Aggregated: ${RESULTS_DIR}/aggregated/"
log ""
log "Next steps:"
log "  - Edit config: vim ${PIPELINE_CONFIG}"
log "  - Re-aggregate: ced aggregate-splits --results-dir ${RESULTS_DIR}"
log "  - View results: ls ${RESULTS_DIR}/aggregated/"
log "============================================"
