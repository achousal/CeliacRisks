#!/usr/bin/env bash
#============================================================
# run_production.sh
#
# Production pipeline orchestration using `ced` CLI
#
# Usage:
#   # Standard run (4 models, 10 splits)
#   ./run_production.sh
#
#   # Single split mode
#   N_SPLITS=1 ./run_production.sh
#
#   # Dry run
#   DRY_RUN=1 ./run_production.sh
#
#   # Run subset of models
#   RUN_MODELS="LR_EN,RF" ./run_production.sh
#
#   # Postprocess only (no new jobs)
#   POSTPROCESS_ONLY=1 ./run_production.sh
#============================================================

set -euo pipefail
IFS=$'\n\t'

log() { echo "[$(date +'%F %T')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

#==============================================================
# CONFIGURATION
#==============================================================
# HPC settings
PROJECT="${PROJECT:-acc_Chipuk_Laboratory}"
QUEUE="${QUEUE:-premium}"

# Paths
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFILE="${INFILE:-${BASE_DIR}/../data/Celiac_dataset_proteomics.csv}"
SPLITS_DIR="${SPLITS_DIR:-${BASE_DIR}/splits_production}"
RESULTS_DIR="${RESULTS_DIR:-${BASE_DIR}/results_production}"
LOGS_DIR="${LOGS_DIR:-${BASE_DIR}/logs}"

# Configs
SPLITS_CONFIG="${SPLITS_CONFIG:-${BASE_DIR}/configs/splits_config.yaml}"
TRAINING_CONFIG="${TRAINING_CONFIG:-${BASE_DIR}/configs/training_config.yaml}"

# Execution modes
DRY_RUN="${DRY_RUN:-0}"
POSTPROCESS_ONLY="${POSTPROCESS_ONLY:-0}"
OVERWRITE_SPLITS="${OVERWRITE_SPLITS:-0}"

# Split configuration
N_SPLITS="${N_SPLITS:-10}"
SEED_START="${SEED_START:-0}"

# Model selection (comma-separated: RF,XGBoost,LinSVM_cal,LR_EN)
RUN_MODELS="${RUN_MODELS:-RF,XGBoost,LinSVM_cal,LR_EN}"

#==============================================================
# SETUP
#==============================================================
cd "${BASE_DIR}"
mkdir -p "${LOGS_DIR}" "${SPLITS_DIR}" "${RESULTS_DIR}"

[[ -f "${INFILE}" ]] || die "Input file not found: ${INFILE}"
[[ -f "${SPLITS_CONFIG}" ]] || die "Splits config not found: ${SPLITS_CONFIG}"
[[ -f "${TRAINING_CONFIG}" ]] || die "Training config not found: ${TRAINING_CONFIG}"

# Activate virtual environment
VENV_PATH="${BASE_DIR}/venv/bin/activate"
if [[ -f "${VENV_PATH}" ]]; then
  source "${VENV_PATH}"
  log "Virtual environment activated"
else
  die "Virtual environment not found at ${VENV_PATH}. Run: bash scripts/hpc_setup.sh"
fi

log "============================================"
log "Celiac Disease ML Pipeline - Production"
log "============================================"
log "Base dir: ${BASE_DIR}"
log "Input file: ${INFILE}"
log "Splits dir: ${SPLITS_DIR}"
log "Results dir: ${RESULTS_DIR}"
log "Models: ${RUN_MODELS}"
log "N splits: ${N_SPLITS}"
log "Dry run: ${DRY_RUN}"
log "============================================"

#==============================================================
# STEP 1: GENERATE SPLITS
#==============================================================
log "Step 1/3: Generate splits"

# Check if splits exist (check for any scenario splits)
SPLITS_EXIST=0
if ls "${SPLITS_DIR}"/*_train_idx_seed0.csv 1>/dev/null 2>&1; then
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
# STEP 2: SUBMIT TRAINING JOBS
#==============================================================
if [[ ${POSTPROCESS_ONLY} -eq 1 ]]; then
  log "POSTPROCESS_ONLY=1: Skipping job submission"
else
  log "Step 2/3: Submit training jobs"

  # Parse model list
  IFS=',' read -r -a MODEL_ARRAY <<< "${RUN_MODELS}"

  SUBMITTED_JOBS=()
  for MODEL in "${MODEL_ARRAY[@]}"; do
    MODEL=$(echo "${MODEL}" | xargs)  # trim whitespace
    [[ -z "${MODEL}" ]] && continue

    JOB_NAME="CeD_${MODEL}"

    if [[ ${DRY_RUN} -eq 1 ]]; then
      log "[DRY RUN] Would submit: ${JOB_NAME}"
      SUBMITTED_JOBS+=("DRYRUN_${MODEL}")
    else
      log "Submitting ${MODEL}..."

      # Submit LSF job
      BSUB_OUT=$(bsub \
        -P "${PROJECT}" \
        -q "${QUEUE}" \
        -J "${JOB_NAME}" \
        -n 8 \
        -W 144:00 \
        -R "span[hosts=1] rusage[mem=8000]" \
        -oo "${LOGS_DIR}/${JOB_NAME}.%J.out" \
        -eo "${LOGS_DIR}/${JOB_NAME}.%J.err" \
        -env "MODEL=${MODEL},BASE_DIR=${BASE_DIR},INFILE=${INFILE},SPLITS_DIR=${SPLITS_DIR},RESULTS_DIR=${RESULTS_DIR},TRAINING_CONFIG=${TRAINING_CONFIG}" \
        bash -c "source ${VENV_PATH} && ced train --config ${TRAINING_CONFIG} --model ${MODEL} --infile ${INFILE} --split-dir ${SPLITS_DIR} --outdir ${RESULTS_DIR}" \
        2>&1)

      JOB_ID=$(echo "${BSUB_OUT}" | grep -oE 'Job <[0-9]+>' | head -n1 | tr -cd '0-9')

      if [[ -n "${JOB_ID}" ]]; then
        log "  ✓ ${MODEL}: Job ${JOB_ID}"
        SUBMITTED_JOBS+=("${JOB_ID}")
      else
        log "  ✗ ${MODEL}: Submission failed"
        echo "${BSUB_OUT}"
      fi
    fi
  done

  log "Submitted ${#SUBMITTED_JOBS[@]} job(s): $(IFS=','; echo "${SUBMITTED_JOBS[*]}")"
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
    ced postprocess --results-dir "${RESULTS_DIR}" --n-boot 500
    log "✓ Postprocessing complete"
  else
    log "No completed jobs found. Postprocessing will run when jobs complete."
  fi
fi

#==============================================================
# SUMMARY
#==============================================================
log "============================================"
log "Pipeline submission complete"
log "============================================"
log "Monitor jobs:"
log "  bjobs -w | grep CeD_"
log ""
log "View results:"
log "  ls ${RESULTS_DIR}"
log ""
log "Postprocess after completion:"
log "  ced postprocess --results-dir ${RESULTS_DIR}"
log "============================================"
