#!/usr/bin/env bash
#============================================================
# run_local.sh
#
# Local pipeline orchestration using `ced` CLI
# Runs models sequentially (no HPC job submission)
#
# Usage:
#   # Quick smoke test (1 split, 1 model)
#   ./run_local.sh
#
#   # Multiple splits and models
#   N_SPLITS=3 RUN_MODELS="LR_EN,RF" ./run_local.sh
#
#   # Dry run
#   DRY_RUN=1 ./run_local.sh
#
#   # Postprocess only
#   POSTPROCESS_ONLY=1 ./run_local.sh
#============================================================

set -euo pipefail
IFS=$'\n\t'

log() { echo "[$(date +'%F %T')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

#==============================================================
# CONFIGURATION
#==============================================================
# Paths
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFILE="${INFILE:-${BASE_DIR}/../data/Celiac_dataset_proteomics_w_demo.csv}"
SPLITS_DIR="${SPLITS_DIR:-${BASE_DIR}/splits_local}"
RESULTS_DIR="${RESULTS_DIR:-${BASE_DIR}/results_local}"
LOGS_DIR="${LOGS_DIR:-${BASE_DIR}/logs_local}"

# Configs
SPLITS_CONFIG="${SPLITS_CONFIG:-${BASE_DIR}/configs/my_splits_config.yaml}"
TRAINING_CONFIG="${TRAINING_CONFIG:-${BASE_DIR}/configs/my_training_config.yaml}"

# Execution modes
DRY_RUN="${DRY_RUN:-0}"
POSTPROCESS_ONLY="${POSTPROCESS_ONLY:-0}"
OVERWRITE_SPLITS="${OVERWRITE_SPLITS:-0}"

# Local-friendly defaults (override with env vars)
N_SPLITS="${N_SPLITS:-1}"           # 1 split for quick testing
SEED_START="${SEED_START:-0}"
N_BOOT="${N_BOOT:-100}"             # Fewer bootstraps for speed

# Model selection (comma-separated: RF,XGBoost,LinSVM_cal,LR_EN)
# Default: just LR_EN for quick smoke test
RUN_MODELS="${RUN_MODELS:-LR_EN}"

#==============================================================
# ENVIRONMENT DETECTION
#==============================================================
# Detect if we're in conda or need to activate venv
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  log "Detected conda environment: ${CONDA_DEFAULT_ENV}"
  ENV_TYPE="conda"
elif [[ -f "${BASE_DIR}/venv/bin/activate" ]]; then
  log "Activating venv..."
  source "${BASE_DIR}/venv/bin/activate"
  ENV_TYPE="venv"
else
  die "No Python environment found. Either activate conda (conda activate ced_ml) or run: bash scripts/hpc_setup.sh"
fi

# Verify ced command is available
if ! command -v ced &> /dev/null; then
  die "ced command not found. Install package with: pip install -e ."
fi

#==============================================================
# SETUP
#==============================================================
cd "${BASE_DIR}"
mkdir -p "${LOGS_DIR}" "${SPLITS_DIR}" "${RESULTS_DIR}"

[[ -f "${INFILE}" ]] || die "Input file not found: ${INFILE}"
[[ -f "${SPLITS_CONFIG}" ]] || die "Splits config not found: ${SPLITS_CONFIG}"
[[ -f "${TRAINING_CONFIG}" ]] || die "Training config not found: ${TRAINING_CONFIG}"

log "============================================"
log "Celiac Disease ML Pipeline - Local"
log "============================================"
log "Environment: ${ENV_TYPE}"
log "Base dir: ${BASE_DIR}"
log "Input file: ${INFILE}"
log "Splits dir: ${SPLITS_DIR}"
log "Results dir: ${RESULTS_DIR}"
log "Models: ${RUN_MODELS}"
log "N splits: ${N_SPLITS}"
log "N bootstraps: ${N_BOOT}"
log "Dry run: ${DRY_RUN}"
log "============================================"

#==============================================================
# STEP 1: GENERATE SPLITS
#==============================================================
log "Step 1/3: Generate splits"

# Check if splits exist (check for train AND val splits)
SPLITS_EXIST=0
if ls "${SPLITS_DIR}"/*_train_idx_seed0.csv 1>/dev/null 2>&1 && \
   ls "${SPLITS_DIR}"/*_val_idx_seed0.csv 1>/dev/null 2>&1; then
  SPLITS_EXIST=1
fi

if [[ ${SPLITS_EXIST} -eq 1 && ${OVERWRITE_SPLITS} -eq 0 ]]; then
  log "Splits already exist. Set OVERWRITE_SPLITS=1 to regenerate."
else
  if [[ ${DRY_RUN} -eq 1 ]]; then
    log "[DRY RUN] Would run: ced save-splits"
  else
    log "Generating splits..."
    OVERWRITE_FLAG=""
    if [[ ${OVERWRITE_SPLITS} -eq 1 ]]; then
      OVERWRITE_FLAG="--overwrite"
    fi

    ced save-splits \
      --config "${SPLITS_CONFIG}" \
      --infile "${INFILE}" \
      --outdir "${SPLITS_DIR}" \
      --n-splits "${N_SPLITS}" \
      --seed-start "${SEED_START}" \
      ${OVERWRITE_FLAG}

    log "✓ Splits generated"
  fi
fi

#==============================================================
# STEP 2: TRAIN MODELS (SEQUENTIAL)
#==============================================================
if [[ ${POSTPROCESS_ONLY} -eq 1 ]]; then
  log "POSTPROCESS_ONLY=1: Skipping training"
else
  log "Step 2/3: Train models (sequential)"

  # Parse model list
  IFS=',' read -r -a MODEL_ARRAY <<< "${RUN_MODELS}"

  COMPLETED_MODELS=()
  for MODEL in "${MODEL_ARRAY[@]}"; do
    MODEL=$(echo "${MODEL}" | xargs)  # trim whitespace
    [[ -z "${MODEL}" ]] && continue

    JOB_NAME="CeD_${MODEL}"

    if [[ ${DRY_RUN} -eq 1 ]]; then
      log "[DRY RUN] Would train: ${MODEL}"
      COMPLETED_MODELS+=("${MODEL}")
    else
      log "Training ${MODEL}..."
      START_TIME=$(date +%s)

      # Run training and capture output
      LOG_FILE="${LOGS_DIR}/${JOB_NAME}_$(date +%Y%m%d_%H%M%S).log"

      if ced train \
        --config "${TRAINING_CONFIG}" \
        --model "${MODEL}" \
        --infile "${INFILE}" \
        --split-dir "${SPLITS_DIR}" \
        --outdir "${RESULTS_DIR}" \
        2>&1 | tee "${LOG_FILE}"; then

        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        log "  ✓ ${MODEL}: Complete (${ELAPSED}s)"
        COMPLETED_MODELS+=("${MODEL}")
      else
        log "  ✗ ${MODEL}: Failed (see ${LOG_FILE})"
      fi
    fi
  done

  log "Completed ${#COMPLETED_MODELS[@]} model(s): $(IFS=','; echo "${COMPLETED_MODELS[*]}")"
fi

#==============================================================
# STEP 3: POSTPROCESSING
#==============================================================
log "Step 3/3: Postprocessing"

if [[ ${DRY_RUN} -eq 1 ]]; then
  log "[DRY RUN] Would run: ced postprocess"
else
  # Check for completed jobs
  COMPLETED_COUNT=$(find "${RESULTS_DIR}" -name "test_metrics.csv" -path "*/core/*" 2>/dev/null | wc -l)

  log "Found ${COMPLETED_COUNT} completed model run(s)"

  if [[ ${COMPLETED_COUNT} -gt 0 ]]; then
    log "Running postprocessing..."
    ced postprocess --results-dir "${RESULTS_DIR}" --n-boot "${N_BOOT}"
    log "✓ Postprocessing complete"
  else
    log "No completed jobs found. Skipping postprocessing."
  fi
fi

#==============================================================
# SUMMARY
#==============================================================
log "============================================"
log "Pipeline complete"
log "============================================"
log "Results directory:"
log "  ${RESULTS_DIR}"
log ""
log "View results:"
log "  ls -la ${RESULTS_DIR}"
log ""
log "Next steps:"
log "  - Check results: ced postprocess --results-dir ${RESULTS_DIR}"
log "  - Run more models: RUN_MODELS='RF,XGBoost' ./run_local.sh"
log "  - Run more splits: N_SPLITS=10 ./run_local.sh"
log "  - Deploy to HPC: see ./run_production.sh and SETUP_README.md"
log "============================================"
