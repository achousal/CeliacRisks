#!/usr/bin/env bash
#============================================================
# post_training_pipeline.sh
#
# Comprehensive post-training pipeline with detailed logging:
# 1. Validate base model outputs
# 2. Train ensemble (if enabled)
# 3. Aggregate results across splits
# 4. Generate validation reports
#
# Usage:
#   cd analysis/
#   bash scripts/post_training_pipeline.sh --run-id 20260122_120000
#   bash scripts/post_training_pipeline.sh --run-id 20260122_120000 --train-ensemble
#============================================================

set -euo pipefail
IFS=$'\n\t'

#==============================================================
# ARGUMENT PARSING
#==============================================================
TRAIN_ENSEMBLE=0
RUN_ID=""
RESULTS_DIR=""
CONFIG_FILE=""
BASE_MODELS=""
MIN_SPLITS=1

while [[ $# -gt 0 ]]; do
  case $1 in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --base-models)
      BASE_MODELS="$2"
      shift 2
      ;;
    --train-ensemble)
      TRAIN_ENSEMBLE=1
      shift
      ;;
    --min-splits)
      MIN_SPLITS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

#==============================================================
# SETUP
#==============================================================
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_DIR}"

# Timestamped log file for this post-processing run
# Use separate post directory: logs/post/run_{RUN_ID}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [[ -n "${RUN_ID}" ]]; then
  POST_LOG_DIR="${BASE_DIR}/logs/post/run_${RUN_ID}"
else
  # Fallback for auto run ID
  RUN_ID="${TIMESTAMP}"
  POST_LOG_DIR="${BASE_DIR}/logs/post/run_${RUN_ID}"
fi
mkdir -p "${POST_LOG_DIR}"
LOG_FILE="${POST_LOG_DIR}/post_training.log"

# Logging functions
log() {
  local msg="[$(date +'%Y-%m-%d %H:%M:%S')] $*"
  echo "${msg}" | tee -a "${LOG_FILE}"
}

log_section() {
  local title="$1"
  local line="============================================"
  log ""
  log "${line}"
  log "${title}"
  log "${line}"
}

log_error() {
  local msg="[ERROR] $*"
  echo "${msg}" | tee -a "${LOG_FILE}" >&2
}

log_success() {
  log "[SUCCESS] $*"
}

log_warning() {
  log "[WARNING] $*"
}

# Load defaults from config if not provided
if [[ -z "${CONFIG_FILE}" ]]; then
  CONFIG_FILE="${BASE_DIR}/configs/pipeline_hpc.yaml"
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  log_error "Config file not found: ${CONFIG_FILE}"
  exit 1
fi

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

# Load config
if [[ -z "${RESULTS_DIR}" ]]; then
  RESULTS_DIR=$(get_yaml "${CONFIG_FILE}" "results_dir")
  RESULTS_DIR="${BASE_DIR}/${RESULTS_DIR}"
fi

TRAINING_CONFIG=$(get_yaml "${CONFIG_FILE}" "training")
TRAINING_CONFIG="${BASE_DIR}/${TRAINING_CONFIG}"

# Ensemble training is opt-in via --train-ensemble flag
# No longer read from config file (ensemble.enabled removed)

if [[ -z "${BASE_MODELS}" ]]; then
  BASE_MODELS=$(get_yaml_nested_list "${TRAINING_CONFIG}" "ensemble" "base_models")
  if [[ -z "${BASE_MODELS}" ]]; then
    BASE_MODELS=$(get_yaml_list "${CONFIG_FILE}" "models")
  fi
fi

N_BOOT=$(get_yaml "${CONFIG_FILE}" "n_boot")

# Activate venv (skip if conda env already active)
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
  VENV_PATH="${BASE_DIR}/venv/bin/activate"
  if [[ -f "${VENV_PATH}" ]]; then
    source "${VENV_PATH}"
  else
    log_error "Virtual environment not found: ${VENV_PATH}"
    log_error "Either activate a conda environment (e.g., 'conda activate ced_ml') or create a venv."
    exit 1
  fi
else
  log "Using active conda environment: ${CONDA_DEFAULT_ENV}"
fi

log_section "Post-Training Pipeline"
log "Run ID: ${RUN_ID:-auto}"
log "Results directory: ${RESULTS_DIR}"
log "Config file: ${CONFIG_FILE}"
log "Training config: ${TRAINING_CONFIG}"
log "Base models: ${BASE_MODELS}"
log "Train ensemble: ${TRAIN_ENSEMBLE}"
log "Bootstrap iterations: ${N_BOOT}"
log "Log file: ${LOG_FILE}"

#==============================================================
# STEP 1: VALIDATE BASE MODEL OUTPUTS
#==============================================================
log_section "Step 1: Validate Base Model Outputs"

IFS=',' read -r -a MODEL_ARRAY <<< "${BASE_MODELS}"
SPLITS_DIR="${BASE_DIR}/splits"
SEED_START=$(grep "^seed_start:" "${BASE_DIR}/configs/splits_config.yaml" | awk '{print $2}')
N_SPLITS=$(grep "^n_splits:" "${BASE_DIR}/configs/splits_config.yaml" | awk '{print $2}')
SEED_END=$((SEED_START + N_SPLITS - 1))

VALIDATED_MODELS=()
MISSING_MODELS=()

for MODEL in "${MODEL_ARRAY[@]}"; do
  MODEL=$(echo "${MODEL}" | xargs)
  [[ -z "${MODEL}" ]] && continue

  log "Validating ${MODEL}..."

  # Count completed splits for this model
  COMPLETED_SPLITS=0
  for SEED in $(seq ${SEED_START} ${SEED_END}); do
    # Try both path patterns (for flexibility)
    MODEL_RUN_DIR="${RESULTS_DIR}/${MODEL}/run_${RUN_ID}"
    MODEL_SPLIT_DIR="${MODEL_RUN_DIR}/split_seed${SEED}"

    # Fallback: legacy path (split_{seed} instead of run_{id}/split_seed{seed})
    if [[ ! -d "${MODEL_SPLIT_DIR}" ]]; then
      MODEL_SPLIT_DIR="${RESULTS_DIR}/${MODEL}/split_${SEED}"
    fi
    if [[ ! -d "${MODEL_SPLIT_DIR}" ]]; then
      MODEL_SPLIT_DIR="${RESULTS_DIR}/${MODEL}/split_seed${SEED}"
    fi

    # Check for required output files
    METRICS_FILE="${MODEL_SPLIT_DIR}/core/test_metrics.csv"
    OOF_FILE="${MODEL_SPLIT_DIR}/preds/train_oof/train_oof__${MODEL}.csv"
    TEST_FILE="${MODEL_SPLIT_DIR}/preds/test_preds/test_preds__${MODEL}.csv"
    VAL_FILE="${MODEL_SPLIT_DIR}/preds/val_preds/val_preds__${MODEL}.csv"

    if [[ -f "${METRICS_FILE}" && -f "${OOF_FILE}" && -f "${TEST_FILE}" ]]; then
      COMPLETED_SPLITS=$((COMPLETED_SPLITS + 1))
    else
      log_warning "  Seed ${SEED}: Missing required files in ${MODEL_SPLIT_DIR}"
      [[ ! -f "${METRICS_FILE}" ]] && log_warning "    - Missing: test_metrics.csv"
      [[ ! -f "${OOF_FILE}" ]] && log_warning "    - Missing: train_oof__${MODEL}.csv"
      [[ ! -f "${TEST_FILE}" ]] && log_warning "    - Missing: test_preds__${MODEL}.csv"
    fi
  done

  if [[ ${COMPLETED_SPLITS} -ge ${MIN_SPLITS} ]]; then
    log_success "${MODEL}: ${COMPLETED_SPLITS}/${N_SPLITS} splits completed"
    VALIDATED_MODELS+=("${MODEL}")
  else
    log_error "${MODEL}: Only ${COMPLETED_SPLITS}/${N_SPLITS} splits completed (min: ${MIN_SPLITS})"
    MISSING_MODELS+=("${MODEL}")
  fi
done

if [[ ${#VALIDATED_MODELS[@]} -eq 0 ]]; then
  log_error "No validated base models found. Exiting."
  exit 1
fi

log_success "Validated ${#VALIDATED_MODELS[@]} base model(s): ${VALIDATED_MODELS[*]}"
if [[ ${#MISSING_MODELS[@]} -gt 0 ]]; then
  log_warning "Incomplete models (skipped): ${MISSING_MODELS[*]}"
fi

#==============================================================
# STEP 2: TRAIN ENSEMBLE (if requested)
#==============================================================
if [[ ${TRAIN_ENSEMBLE} -eq 1 ]]; then
  log_section "Step 2: Train Ensemble Meta-Learner"

  ENSEMBLE_MODELS=$(IFS=,; echo "${VALIDATED_MODELS[*]}")
  log "Using base models: ${ENSEMBLE_MODELS}"

  ENSEMBLE_TRAINED=0
  ENSEMBLE_FAILED=0

  for SEED in $(seq ${SEED_START} ${SEED_END}); do
    log "Training ensemble for seed ${SEED}..."

    ENSEMBLE_OUT="${RESULTS_DIR}/ENSEMBLE/run_${RUN_ID}/split_seed${SEED}"

    # Check if already trained
    if [[ -f "${ENSEMBLE_OUT}/core/metrics.json" ]]; then
      log_warning "  Ensemble already exists for seed ${SEED}, skipping"
      ENSEMBLE_TRAINED=$((ENSEMBLE_TRAINED + 1))
      continue
    fi

    # Train ensemble
    set +e
    ced train-ensemble \
      --config "${TRAINING_CONFIG}" \
      --results-dir "${RESULTS_DIR}" \
      --base-models "${ENSEMBLE_MODELS}" \
      --split-seed "${SEED}" \
      --outdir "${ENSEMBLE_OUT}" \
      2>&1 | tee -a "${LOG_FILE}"

    ENSEMBLE_EXIT=$?
    set -e

    if [[ ${ENSEMBLE_EXIT} -eq 0 ]]; then
      log_success "  Ensemble trained for seed ${SEED}"
      ENSEMBLE_TRAINED=$((ENSEMBLE_TRAINED + 1))
    else
      log_error "  Ensemble training failed for seed ${SEED} (exit code: ${ENSEMBLE_EXIT})"
      ENSEMBLE_FAILED=$((ENSEMBLE_FAILED + 1))
    fi
  done

  if [[ ${ENSEMBLE_FAILED} -gt 0 ]]; then
    log_warning "Ensemble training: ${ENSEMBLE_TRAINED} successful, ${ENSEMBLE_FAILED} skipped (missing base models)"
  else
    log_success "Ensemble training: ${ENSEMBLE_TRAINED} successful"
  fi

  if [[ ${ENSEMBLE_TRAINED} -eq 0 ]]; then
    log_warning "No ensemble models trained. Ensure all base models are trained for each split before running ensemble."
  fi
else
  log_section "Step 2: Train Ensemble (Skipped)"
  log "Ensemble training not requested (use --train-ensemble flag to enable)"
fi

#==============================================================
# STEP 3: AGGREGATE RESULTS (per model)
#==============================================================
log_section "Step 3: Aggregate Results Across Splits"

# Build list of models to aggregate (base + ENSEMBLE if trained)
MODELS_TO_AGG=("${VALIDATED_MODELS[@]}")
if [[ ${TRAIN_ENSEMBLE} -eq 1 && ${ENSEMBLE_TRAINED:-0} -gt 0 ]]; then
  MODELS_TO_AGG+=("ENSEMBLE")
fi

AGG_SUCCESS=()
AGG_FAILED=()

for MODEL in "${MODELS_TO_AGG[@]}"; do
  log "Aggregating ${MODEL}..."

  # Determine model directory
  if [[ "${MODEL}" == "ENSEMBLE" ]]; then
    MODEL_DIR="${RESULTS_DIR}/ENSEMBLE/run_${RUN_ID}"
  else
    MODEL_DIR="${RESULTS_DIR}/${MODEL}/run_${RUN_ID}"
    # Fallback: legacy path
    if [[ ! -d "${MODEL_DIR}" ]]; then
      MODEL_DIR="${RESULTS_DIR}/${MODEL}"
    fi
  fi

  if [[ ! -d "${MODEL_DIR}" ]]; then
    log_error "  Model directory not found: ${MODEL_DIR}"
    AGG_FAILED+=("${MODEL}")
    continue
  fi

  # Check if aggregation already done
  AGG_OUT="${MODEL_DIR}/aggregated"
  if [[ -f "${AGG_OUT}/aggregation_metadata.json" ]]; then
    log_warning "  Aggregation already exists for ${MODEL}, skipping"
    AGG_SUCCESS+=("${MODEL}")
    continue
  fi

  # Run aggregation
  set +e
  ced aggregate-splits \
    --results-dir "${MODEL_DIR}" \
    --n-boot "${N_BOOT}" \
    2>&1 | tee -a "${LOG_FILE}"

  AGG_EXIT=$?
  set -e

  if [[ ${AGG_EXIT} -eq 0 ]]; then
    log_success "  Aggregated ${MODEL}"
    AGG_SUCCESS+=("${MODEL}")
  else
    log_error "  Aggregation failed for ${MODEL} (exit code: ${AGG_EXIT})"
    AGG_FAILED+=("${MODEL}")
  fi
done

log_success "Aggregation: ${#AGG_SUCCESS[@]} successful, ${#AGG_FAILED[@]} failed"
if [[ ${#AGG_FAILED[@]} -gt 0 ]]; then
  log_warning "Failed to aggregate: ${AGG_FAILED[*]}"
fi

#==============================================================
# STEP 4: VALIDATION REPORT
#==============================================================
log_section "Step 4: Validation Report"

log "Generating validation summary..."

# Count total aggregated files
TOTAL_FILES=0
for MODEL in "${AGG_SUCCESS[@]}"; do
  if [[ "${MODEL}" == "ENSEMBLE" ]]; then
    MODEL_AGG="${RESULTS_DIR}/ENSEMBLE/run_${RUN_ID}/aggregated"
  else
    MODEL_DIR="${RESULTS_DIR}/${MODEL}/run_${RUN_ID}"
    if [[ ! -d "${MODEL_DIR}" ]]; then
      MODEL_DIR="${RESULTS_DIR}/${MODEL}"
    fi
    MODEL_AGG="${MODEL_DIR}/aggregated"
  fi

  if [[ -d "${MODEL_AGG}" ]]; then
    FILE_COUNT=$(find "${MODEL_AGG}" -type f | wc -l)
    TOTAL_FILES=$((TOTAL_FILES + FILE_COUNT))
    log "  ${MODEL}: ${FILE_COUNT} aggregated files"
  fi
done

log_success "Total aggregated files: ${TOTAL_FILES}"

# Generate summary JSON
SUMMARY_JSON="${POST_LOG_DIR}/pipeline_summary.json"
cat > "${SUMMARY_JSON}" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "run_id": "${RUN_ID:-auto}",
  "results_dir": "${RESULTS_DIR}",
  "base_models": {
    "validated": $(printf '%s\n' "${VALIDATED_MODELS[@]}" | jq -R . | jq -s .),
    "missing": $(printf '%s\n' "${MISSING_MODELS[@]}" | jq -R . | jq -s .)
  },
  "ensemble": {
    "requested": ${TRAIN_ENSEMBLE},
    "trained": ${ENSEMBLE_TRAINED:-0},
    "failed": ${ENSEMBLE_FAILED:-0}
  },
  "aggregation": {
    "successful": $(printf '%s\n' "${AGG_SUCCESS[@]}" | jq -R . | jq -s .),
    "failed": $(printf '%s\n' "${AGG_FAILED[@]}" | jq -R . | jq -s .)
  },
  "files": {
    "total_aggregated": ${TOTAL_FILES},
    "log_file": "${LOG_FILE}"
  }
}
EOF

log_success "Summary JSON: ${SUMMARY_JSON}"

#==============================================================
# FINAL SUMMARY
#==============================================================
log_section "Post-Training Pipeline Complete"
log "Results directory: ${RESULTS_DIR}"
log "Models aggregated: ${AGG_SUCCESS[*]}"
log "Log file: ${LOG_FILE}"
log "Summary: ${SUMMARY_JSON}"

if [[ ${#AGG_FAILED[@]} -gt 0 ]]; then
  log_warning "Some models failed aggregation. Check log for details."
  exit 1
fi

log_success "All steps completed successfully"
exit 0
