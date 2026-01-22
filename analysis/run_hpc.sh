#!/usr/bin/env bash
#============================================================
# run_hpc.sh
#
# Config-driven HPC pipeline orchestration
# All settings in configs/pipeline_hpc.yaml
#
# IMPORTANT: This script must be run from the analysis/ directory.
#            Relative paths in configs assume this working directory.
#
# Usage:
#   cd analysis/                     # REQUIRED: Run from analysis/
#   ./run_hpc.sh                     # Use pipeline_hpc.yaml
#   PIPELINE_CONFIG=custom.yaml ./run_hpc.sh  # Custom config
#   DRY_RUN=1 ./run_hpc.sh           # Override dry_run
#============================================================

set -euo pipefail
IFS=$'\n\t'

log() { echo "[$(date +"%F %T")] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
normalize_bool() {
  local val="${1:-}"
  val="$(printf '%s' "${val}" | tr '[:upper:]' '[:lower:]')"
  case "${val}" in
    1|true|yes|y) echo 1 ;;
    0|false|no|n|'') echo 0 ;;
    *) echo "${val}" ;;
  esac
}

#==============================================================
# LOAD PIPELINE CONFIG
#==============================================================
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-${BASE_DIR}/configs/pipeline_hpc.yaml}"

[[ -f "${PIPELINE_CONFIG}" ]] || die "Pipeline config not found: ${PIPELINE_CONFIG}"

# Helper: extract YAML value
get_yaml() {
  local file="$1"
  local key="$2"
  grep "^  ${key}:" "${file}" | sed 's/^[^:]*: *//' | tr -d '"' | tr -d "'" | sed 's/ *#.*//'
}

get_yaml_list() {
  local file="$1"
  local key="$2"
  grep -A10 "^  ${key}:" "${file}" | grep "^ *- " | sed 's/^ *- *//' | tr '\n' ',' | sed 's/,$//'
}

# Read paths
INFILE=$(get_yaml "${PIPELINE_CONFIG}" "infile")
SPLITS_DIR=$(get_yaml "${PIPELINE_CONFIG}" "splits_dir")
RESULTS_DIR=$(get_yaml "${PIPELINE_CONFIG}" "results_dir")
LOGS_DIR=$(get_yaml "${PIPELINE_CONFIG}" "logs_dir")

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
POSTPROCESS_ONLY_CFG=$(get_yaml "${PIPELINE_CONFIG}" "postprocess_only")

# Read HPC settings
PROJECT=$(get_yaml "${PIPELINE_CONFIG}" "project")
QUEUE=$(get_yaml "${PIPELINE_CONFIG}" "queue")
WALLTIME=$(get_yaml "${PIPELINE_CONFIG}" "walltime")
CORES=$(get_yaml "${PIPELINE_CONFIG}" "cores")
MEM=$(get_yaml "${PIPELINE_CONFIG}" "mem_per_core")

# Environment variable overrides
DRY_RUN="$(normalize_bool "${DRY_RUN:-${DRY_RUN_CFG}}")"
POSTPROCESS_ONLY="$(normalize_bool "${POSTPROCESS_ONLY:-${POSTPROCESS_ONLY_CFG}}")"
OVERWRITE_SPLITS="$(normalize_bool "${OVERWRITE_SPLITS:-${OVERWRITE_SPLITS_CFG}}")"
RUN_MODELS="${RUN_MODELS:-${MODELS_STR}}"
PROJECT="${PROJECT:-YOUR_PROJECT_ALLOCATION}"

#==============================================================
# ENVIRONMENT SETUP
#==============================================================
cd "${BASE_DIR}"
mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}" "${SPLITS_DIR}"

[[ -f "${INFILE}" ]] || die "Input file not found: ${INFILE}"
[[ -f "${SPLITS_CONFIG}" ]] || die "Splits config not found: ${SPLITS_CONFIG}"
[[ -f "${TRAINING_CONFIG}" ]] || die "Training config not found: ${TRAINING_CONFIG}"

[[ "${PROJECT}" == "YOUR_PROJECT_ALLOCATION" ]] && die "HPC project not set. Update configs/pipeline_hpc.yaml"

# Activate venv
VENV_PATH="${BASE_DIR}/venv/bin/activate"
if [[ -f "${VENV_PATH}" ]]; then
  source "${VENV_PATH}"
else
  die "Virtual environment not found at ${VENV_PATH}. Run: bash scripts/hpc_setup.sh"
fi

# Extract split settings
N_SPLITS=$(grep "^n_splits:" "${SPLITS_CONFIG}" | awk '{print $2}')
SEED_START=$(grep "^seed_start:" "${SPLITS_CONFIG}" | awk '{print $2}')

# Generate run ID (timestamp-based)
RUN_ID=$(date +"%Y%m%d_%H%M%S")

log "============================================"
log "CeD-ML Pipeline (HPC, Config-Driven)"
log "============================================"
log "Pipeline config: ${PIPELINE_CONFIG}"
log "Run ID: ${RUN_ID}"
log "Input: ${INFILE}"
log "Splits: ${SPLITS_DIR} (${N_SPLITS} splits, seeds ${SEED_START}-$((SEED_START + N_SPLITS - 1)))"
log "Results: ${RESULTS_DIR}"
log "Models: ${RUN_MODELS}"
log "HPC: ${PROJECT} / ${QUEUE} / ${WALLTIME} / ${CORES}c / ${MEM}MB"
log "Dry run: ${DRY_RUN}"
log "Postprocess only: ${POSTPROCESS_ONLY}"
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
# STEP 2: SUBMIT TRAINING JOBS (PER MODEL x SPLIT)
#==============================================================
if [[ ${POSTPROCESS_ONLY} -eq 1 ]]; then
  log "POSTPROCESS_ONLY=1: Skipping job submission"
else
  log "Step 2/4: Submit training jobs (${N_SPLITS} splits x models)"

  IFS=',' read -r -a MODEL_ARRAY <<< "${RUN_MODELS}"
  SUBMITTED_JOBS=()
  SEED_END=$((SEED_START + N_SPLITS - 1))

  for SEED in $(seq ${SEED_START} ${SEED_END}); do
    for MODEL in "${MODEL_ARRAY[@]}"; do
      MODEL=$(echo "${MODEL}" | xargs)
      [[ -z "${MODEL}" ]] && continue

      JOB_NAME="CeD_${MODEL}_seed${SEED}"

      if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[DRY RUN] Would submit: ${JOB_NAME}"
        SUBMITTED_JOBS+=("DRYRUN_${MODEL}_${SEED}")
      else
        log "Submitting ${MODEL} (seed ${SEED})..."

        LOG_BASE="${LOGS_DIR}/${JOB_NAME}.%J"

        BSUB_OUT=$(bsub \
          -P "${PROJECT}" \
          -q "${QUEUE}" \
          -J "${JOB_NAME}" \
          -n ${CORES} \
          -W "${WALLTIME}" \
          -R "span[hosts=1] rusage[mem=${MEM}]" \
          -oo "${LOG_BASE}.out.live" \
          -eo "${LOG_BASE}.err.live" \
          -env "MODEL=${MODEL},SEED=${SEED},BASE_DIR=${BASE_DIR},INFILE=${INFILE},SPLITS_DIR=${SPLITS_DIR},RESULTS_DIR=${RESULTS_DIR},TRAINING_CONFIG=${TRAINING_CONFIG},LOG_BASE=${LOG_BASE},RUN_ID=${RUN_ID}" \
          bash -c "
            set -euo pipefail
            source \"\${VENV_PATH}\"

            ced train --config \"\${TRAINING_CONFIG}\" --model \"\${MODEL}\" --infile \"\${INFILE}\" --split-dir \"\${SPLITS_DIR}\" --outdir \"\${RESULTS_DIR}/\${MODEL}\" --split-seed \"\${SEED}\" --override run_id=\"\${RUN_ID}\"
            EXIT_CODE=\$?

            if [[ -f \"\${LOG_BASE}.out.live\" ]]; then
              mv \"\${LOG_BASE}.out.live\" \"\${LOG_BASE}.out\"
            fi
            if [[ -f \"\${LOG_BASE}.err.live\" ]]; then
              mv \"\${LOG_BASE}.err.live\" \"\${LOG_BASE}.err\"
            fi

            exit \$EXIT_CODE
          " \
          2>&1)

        JOB_ID=$(echo "${BSUB_OUT}" | grep -oE 'Job <[0-9]+>' | head -n1 | tr -cd '0-9')

        if [[ -n "${JOB_ID}" ]]; then
          log "  [OK] ${MODEL} seed ${SEED}: Job ${JOB_ID}"
          SUBMITTED_JOBS+=("${JOB_ID}")
        else
          log "  [FAIL] ${MODEL} seed ${SEED}: Submission failed"
          echo "${BSUB_OUT}"
        fi
      fi
    done
  done

  log "Submitted ${#SUBMITTED_JOBS[@]} job(s)"
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
  "environment": "hpc",
  "pipeline_config": "${PIPELINE_CONFIG}",
  "splits_dir": "${SPLITS_DIR}",
  "training_config": "${TRAINING_CONFIG}",
  "infile": "${INFILE}",
  "project": "${PROJECT}",
  "queue": "${QUEUE}"
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
    IFS=',' read -r -a MODEL_ARRAY_AGG <<< "${RUN_MODELS}"
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
    log "No completed runs yet. Aggregate after jobs finish (per model):"
    IFS=',' read -r -a MODEL_ARRAY_INFO <<< "${RUN_MODELS}"
    for MODEL in "${MODEL_ARRAY_INFO[@]}"; do
      MODEL=$(echo "${MODEL}" | xargs)
      [[ -z "${MODEL}" ]] && continue
      log "  ced aggregate-splits --results-dir ${RESULTS_DIR}/${MODEL}/run_${RUN_ID}"
    done
  fi
fi

#==============================================================
# SUMMARY
#==============================================================
log "============================================"
log "Pipeline submission complete"
log "============================================"
log "Run ID: ${RUN_ID}"
log "Monitor jobs:"
log "  bjobs -w | grep CeD_"
log "  ls ${LOGS_DIR}/*.live  # Running jobs"
log ""
log "Results: ${RESULTS_DIR}"
log ""
log "Cleanup stale .live logs (if jobs crashed):"
log "  for f in ${LOGS_DIR}/*.live; do mv \"\$f\" \"\${f%.live}\"; done"
log ""
log "Aggregate after all jobs complete:"
log "  ced aggregate-splits --results-dir ${RESULTS_DIR}/<model>/run_${RUN_ID}"
log "============================================"
