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

INFILE="${BASE_DIR}/${INFILE}"
SPLITS_DIR="${BASE_DIR}/${SPLITS_DIR}"
RESULTS_DIR="${BASE_DIR}/${RESULTS_DIR}"
LOGS_DIR="${BASE_DIR}/${LOGS_DIR}"

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

# Read HPC settings
PROJECT=$(get_yaml "${PIPELINE_CONFIG}" "project")
QUEUE=$(get_yaml "${PIPELINE_CONFIG}" "queue")
WALLTIME=$(get_yaml "${PIPELINE_CONFIG}" "walltime")
CORES=$(get_yaml "${PIPELINE_CONFIG}" "cores")
MEM=$(get_yaml "${PIPELINE_CONFIG}" "mem_per_core")

# Environment variable overrides
DRY_RUN="$(normalize_bool "${DRY_RUN:-${DRY_RUN_CFG}}")"
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

# Extract ensemble settings from training config
ENSEMBLE_ENABLED=$(get_yaml_nested "${TRAINING_CONFIG}" "ensemble" "enabled")
ENSEMBLE_BASE_MODELS=$(get_yaml_nested_list "${TRAINING_CONFIG}" "ensemble" "base_models")
[[ "${ENSEMBLE_ENABLED}" == "true" ]] && ENSEMBLE_ENABLED=1 || ENSEMBLE_ENABLED=0

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
log "Logs: ${RUN_LOGS_DIR}"
log "Models: ${RUN_MODELS}"
log "HPC: ${PROJECT} / ${QUEUE} / ${WALLTIME} / ${CORES}c / ${MEM}MB"
log "Ensemble: ${ENSEMBLE_ENABLED} (base: ${ENSEMBLE_BASE_MODELS:-none})"
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
# STEP 2: SUBMIT TRAINING JOBS (PER MODEL x SPLIT)
#==============================================================
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

        LOG_ERR="${RUN_LOGS_DIR}/${JOB_NAME}.%J.err"
        LIVE_LOG="${RUN_LOGS_DIR}/${JOB_NAME}.%J.live.log"

        BSUB_OUT=$(bsub \
          -P "${PROJECT}" \
          -q "${QUEUE}" \
          -J "${JOB_NAME}" \
          -n ${CORES} \
          -W "${WALLTIME}" \
          -R "span[hosts=1] rusage[mem=${MEM}]" \
          -oo /dev/null \
          -eo "${LOG_ERR}" \
          <<EOF
#!/bin/bash
set -euo pipefail

# Force unbuffered output and colors for live logging
export PYTHONUNBUFFERED=1
export FORCE_COLOR=1

source "${VENV_PATH}"

# Stream to both LSF logs and live log with line buffering
stdbuf -oL -eL ced train \
  --config "${TRAINING_CONFIG}" \
  --model "${MODEL}" \
  --infile "${INFILE}" \
  --split-dir "${SPLITS_DIR}" \
  --outdir "${RESULTS_DIR}/${MODEL}" \
  --split-seed "${SEED}" \
  --override run_id="${RUN_ID}" \
  2>&1 | tee -a "${LIVE_LOG}"

exit \${PIPESTATUS[0]}
EOF
        )

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

  log "Submitted ${#SUBMITTED_JOBS[@]} base model job(s)"

#==============================================================
# STEP 2.5: SUBMIT ENSEMBLE JOBS (if enabled, with dependencies)
#==============================================================
  if [[ ${ENSEMBLE_ENABLED} -eq 1 ]]; then
    log "Step 2.5/4: Submit ensemble jobs (with dependencies)"

    # Determine base models: use config or fall back to trained models
    if [[ -n "${ENSEMBLE_BASE_MODELS}" ]]; then
      ENSEMBLE_MODELS="${ENSEMBLE_BASE_MODELS}"
    else
      ENSEMBLE_MODELS="${RUN_MODELS}"
    fi

    ENSEMBLE_JOBS=()

    for SEED in $(seq ${SEED_START} ${SEED_END}); do
      JOB_NAME="CeD_ENSEMBLE_seed${SEED}"

      # Build dependency string: wait for all base model jobs for this seed
      DEP_JOBS=""
      IFS=',' read -r -a ENSEMBLE_MODEL_ARRAY <<< "${ENSEMBLE_MODELS}"
      for BASE_MODEL in "${ENSEMBLE_MODEL_ARRAY[@]}"; do
        BASE_MODEL=$(echo "${BASE_MODEL}" | xargs)
        [[ -z "${BASE_MODEL}" ]] && continue
        DEP_JOB_NAME="CeD_${BASE_MODEL}_seed${SEED}"
        if [[ -n "${DEP_JOBS}" ]]; then
          DEP_JOBS="${DEP_JOBS} && "
        fi
        DEP_JOBS="${DEP_JOBS}done(${DEP_JOB_NAME})"
      done

      if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[DRY RUN] Would submit: ${JOB_NAME} (depends on: ${ENSEMBLE_MODELS})"
        ENSEMBLE_JOBS+=("DRYRUN_ENSEMBLE_${SEED}")
      else
        log "Submitting ${JOB_NAME} (depends on base models)..."

        LOG_ERR="${RUN_LOGS_DIR}/${JOB_NAME}.%J.err"
        LIVE_LOG="${RUN_LOGS_DIR}/${JOB_NAME}.%J.live.log"

        BSUB_OUT=$(bsub \
          -P "${PROJECT}" \
          -q "${QUEUE}" \
          -J "${JOB_NAME}" \
          -n ${CORES} \
          -W "${WALLTIME}" \
          -R "span[hosts=1] rusage[mem=${MEM}]" \
          -w "${DEP_JOBS}" \
          -oo /dev/null \
          -eo "${LOG_ERR}" \
          <<EOF
#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export FORCE_COLOR=1

source "${VENV_PATH}"

stdbuf -oL -eL ced train-ensemble \
  --config "${TRAINING_CONFIG}" \
  --results-dir "${RESULTS_DIR}" \
  --base-models "${ENSEMBLE_MODELS}" \
  --split-seed "${SEED}" \
  --outdir "${RESULTS_DIR}/ENSEMBLE/run_${RUN_ID}/split_seed${SEED}" \
  2>&1 | tee -a "${LIVE_LOG}"

exit \${PIPESTATUS[0]}
EOF
        )

        JOB_ID=$(echo "${BSUB_OUT}" | grep -oE 'Job <[0-9]+>' | head -n1 | tr -cd '0-9')

        if [[ -n "${JOB_ID}" ]]; then
          log "  [OK] ENSEMBLE seed ${SEED}: Job ${JOB_ID} (depends on base models)"
          ENSEMBLE_JOBS+=("${JOB_ID}")
        else
          log "  [FAIL] ENSEMBLE seed ${SEED}: Submission failed"
          echo "${BSUB_OUT}"
        fi
      fi
    done

    log "Submitted ${#ENSEMBLE_JOBS[@]} ensemble job(s)"
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
# STEP 4: POSTPROCESSING INSTRUCTIONS
#==============================================================
log "Step 4/4: Post-processing instructions"

if [[ ${DRY_RUN} -eq 1 ]]; then
  log "[DRY RUN] Would provide post-processing instructions"
else
  log "Jobs submitted. Post-processing will be needed after completion."
  log ""
  log "IMPORTANT: Run post-processing after all jobs complete:"
  log "  bash scripts/post_training_pipeline.sh --run-id ${RUN_ID}"
  log ""
  log "This will:"
  log "  1. Validate base model outputs"
  log "  2. Train ensemble meta-learner (if enabled)"
  log "  3. Aggregate results across splits"
  log "  4. Generate validation reports"
  log ""
  log "Manual alternatives (advanced users):"
  log "  # Train ensemble for each split"
  log "  for seed in \$(seq ${SEED_START} ${SEED_END}); do"
  log "    ced train-ensemble --results-dir ${RESULTS_DIR} --base-models ${RUN_MODELS} --split-seed \$seed"
  log "  done"
  log ""
  log "  # Aggregate per model"
  IFS=',' read -r -a MODEL_ARRAY_INFO <<< "${RUN_MODELS}"
  for MODEL in "${MODEL_ARRAY_INFO[@]}"; do
    MODEL=$(echo "${MODEL}" | xargs)
    [[ -z "${MODEL}" ]] && continue
    log "  ced aggregate-splits --results-dir ${RESULTS_DIR}/${MODEL}/run_${RUN_ID}"
  done
  if [[ ${ENSEMBLE_ENABLED} -eq 1 ]]; then
    log "  ced aggregate-splits --results-dir ${RESULTS_DIR}/ENSEMBLE/run_${RUN_ID}"
  fi
fi

#==============================================================
# SUMMARY
#==============================================================
log "============================================"
log "Pipeline submission complete"
log "============================================"
log "Run ID: ${RUN_ID}"
log ""
log "Monitor jobs:"
log "  bjobs -w | grep CeD_"
if [[ ${ENSEMBLE_ENABLED} -eq 1 ]]; then
  log "  (Ensemble jobs will start after base models complete)"
fi
log ""
log "Live logs (real-time):"
log "  tail -f ${RUN_LOGS_DIR}/*.live.log"
log ""
log "Error logs (post-completion):"
log "  cat ${RUN_LOGS_DIR}/*.err"
log ""
log "After all jobs complete, run post-processing:"
log "  bash scripts/post_training_pipeline.sh --run-id ${RUN_ID}"
log ""
log "Logs: ${RUN_LOGS_DIR}"
log "Results: ${RESULTS_DIR}"
log "============================================"
