#!/usr/bin/env bash
#============================================================
# post_training_pipeline.sh
#
# Comprehensive post-training pipeline with detailed logging:
# 1. Auto-detect models and splits from run-id
# 2. Validate base model outputs
# 3. Train ensemble (if enabled, auto-detects base models)
# 4. Aggregate results across splits
# 5. Generate validation reports
#
# Usage (auto-detection - RECOMMENDED):
#   cd analysis/
#   bash scripts/post_training_pipeline.sh --run-id 20260122_120000
#   bash scripts/post_training_pipeline.sh --run-id 20260122_120000 --train-ensemble
#
# Manual overrides (legacy, disables auto-detection):
#   bash scripts/post_training_pipeline.sh --run-id 20260122_120000 \
#     --results-dir ../results --base-models LR_EN,RF,XGBoost
#
# NOTE: Ensemble training now uses --run-id for full auto-detection.
#       No need to coordinate BASE_MODELS between script and CLI.
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
AUTO_DETECT=1  # Enable auto-detection by default

while [[ $# -gt 0 ]]; do
  case $1 in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      AUTO_DETECT=0  # Disable auto-detection if user provides results-dir
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --base-models)
      BASE_MODELS="$2"
      AUTO_DETECT=0  # Disable auto-detection if user provides base-models
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
      echo "Usage: bash post_training_pipeline.sh --run-id <RUN_ID> [OPTIONS]"
      echo "Required:"
      echo "  --run-id ID              Run identifier (e.g., 20260122_120000)"
      echo "Optional:"
      echo "  --train-ensemble         Train ensemble meta-learner"
      echo "  --min-splits N           Minimum splits required (default: 1)"
      echo "  --results-dir DIR        Override results directory (disables auto-detection)"
      echo "  --base-models M1,M2,...  Override base models (disables auto-detection)"
      echo "  --config FILE            Override config file (default: configs/pipeline_hpc.yaml)"
      exit 1
      ;;
  esac
done

# Validate required argument
if [[ -z "${RUN_ID}" ]]; then
  echo "Error: --run-id is required"
  echo "Usage: bash post_training_pipeline.sh --run-id <RUN_ID> [--train-ensemble] [--min-splits N]"
  exit 1
fi

#==============================================================
# SETUP
#==============================================================
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_DIR}"

# Timestamped log file for this post-processing run
# Logs deposited at root level: ../logs/aggregation/run_{RUN_ID}
ROOT_DIR="$(cd "${BASE_DIR}/.." && pwd)"
POST_LOG_DIR="${ROOT_DIR}/logs/aggregation/run_${RUN_ID:-$(date +"%Y%m%d_%H%M%S")}"
mkdir -p "${POST_LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${POST_LOG_DIR}/post_training_${RUN_ID:-${TIMESTAMP}}.log"

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

#==============================================================
# AUTO-DETECTION (if enabled)
#==============================================================
if [[ ${AUTO_DETECT} -eq 1 ]]; then
  log_section "Auto-Detecting Configuration from Run ID: ${RUN_ID}"

  # Auto-detect results directory
  if [[ -z "${RESULTS_DIR}" ]]; then
    # Try config file first
    RESULTS_DIR_FROM_CONFIG=$(get_yaml "${CONFIG_FILE}" "results_dir")
    RESULTS_DIR="${BASE_DIR}/${RESULTS_DIR_FROM_CONFIG}"

    # Verify it exists
    if [[ ! -d "${RESULTS_DIR}" ]]; then
      # Try default fallback
      RESULTS_DIR="${BASE_DIR}/results"
      if [[ ! -d "${RESULTS_DIR}" ]]; then
        log_error "Results directory not found. Tried:"
        log_error "  - ${BASE_DIR}/${RESULTS_DIR_FROM_CONFIG}"
        log_error "  - ${RESULTS_DIR}"
        exit 1
      fi
    fi
    log "Detected results directory: ${RESULTS_DIR}"
  fi

  # Auto-detect base models by scanning for run_${RUN_ID} directories
  if [[ -z "${BASE_MODELS}" ]]; then
    DETECTED_MODELS=()
    for MODEL_DIR in "${RESULTS_DIR}"/*; do
      [[ ! -d "${MODEL_DIR}" ]] && continue
      MODEL_NAME=$(basename "${MODEL_DIR}")

      # Skip ENSEMBLE (it's generated, not a base model)
      [[ "${MODEL_NAME}" == "ENSEMBLE" ]] && continue

      # Check if this model has run_${RUN_ID} directory
      if [[ -d "${MODEL_DIR}/run_${RUN_ID}" ]]; then
        DETECTED_MODELS+=("${MODEL_NAME}")
      fi
    done

    if [[ ${#DETECTED_MODELS[@]} -eq 0 ]]; then
      log_warning "No models detected with run_${RUN_ID}. Falling back to config."
      # Fallback to config
      TRAINING_CONFIG=$(get_yaml "${CONFIG_FILE}" "training")
      TRAINING_CONFIG="${BASE_DIR}/${TRAINING_CONFIG}"
      BASE_MODELS=$(get_yaml_nested_list "${TRAINING_CONFIG}" "ensemble" "base_models")
      if [[ -z "${BASE_MODELS}" ]]; then
        BASE_MODELS=$(get_yaml_list "${CONFIG_FILE}" "models")
      fi
    else
      BASE_MODELS=$(IFS=,; echo "${DETECTED_MODELS[*]}")
      log "Detected base models: ${BASE_MODELS}"
    fi
  fi
fi

# Load remaining config values
TRAINING_CONFIG=$(get_yaml "${CONFIG_FILE}" "training")
TRAINING_CONFIG="${BASE_DIR}/${TRAINING_CONFIG}"

# If still empty after auto-detection, fall back to config
if [[ -z "${RESULTS_DIR}" ]]; then
  RESULTS_DIR=$(get_yaml "${CONFIG_FILE}" "results_dir")
  RESULTS_DIR="${BASE_DIR}/${RESULTS_DIR}"
fi

if [[ -z "${BASE_MODELS}" ]]; then
  BASE_MODELS=$(get_yaml_nested_list "${TRAINING_CONFIG}" "ensemble" "base_models")
  if [[ -z "${BASE_MODELS}" ]]; then
    BASE_MODELS=$(get_yaml_list "${CONFIG_FILE}" "models")
  fi
fi

N_BOOT=$(get_yaml "${CONFIG_FILE}" "n_boot")

# Activate venv (support both venv and conda)
VENV_PATH="${BASE_DIR}/venv/bin/activate"
if [[ -f "${VENV_PATH}" ]]; then
  source "${VENV_PATH}"
elif [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
  log_error "No Python environment found. Either:"
  log_error "  1. Create and activate a venv: python3 -m venv venv && source venv/bin/activate"
  log_error "  2. Activate a Conda environment: conda activate <env_name>"
  exit 1
else
  log "Using active Conda environment: ${CONDA_DEFAULT_ENV}"
fi

log_section "Post-Training Pipeline Configuration"
log "Run ID: ${RUN_ID}"
log "Auto-detection: $([ ${AUTO_DETECT} -eq 1 ] && echo 'enabled' || echo 'disabled (manual overrides provided)')"
log "Results directory: ${RESULTS_DIR}"
log "Config file: ${CONFIG_FILE}"
log "Training config: ${TRAINING_CONFIG}"
log "Base models: ${BASE_MODELS}"
log "Train ensemble: ${TRAIN_ENSEMBLE}"
log "Bootstrap iterations: ${N_BOOT}"
log "Minimum splits required: ${MIN_SPLITS}"
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

  log "Auto-detecting base models from run_id ${RUN_ID}"

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

    # Train ensemble (auto-detects base models from run-id)
    set +e
    ced train-ensemble \
      --run-id "${RUN_ID}" \
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

  log_success "Ensemble training: ${ENSEMBLE_TRAINED} successful, ${ENSEMBLE_FAILED} failed"

  if [[ ${ENSEMBLE_TRAINED} -eq 0 ]]; then
    log_error "All ensemble training attempts failed. Continuing without ensemble."
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

# Generate summary JSON (safely handle potentially empty arrays)
SUMMARY_JSON="${POST_LOG_DIR}/pipeline_summary_${RUN_ID:-${TIMESTAMP}}.json"

# Convert arrays to JSON safely (filter empty strings)
VALIDATED_JSON=$(printf '%s\n' "${VALIDATED_MODELS[@]:-}" | grep -v '^$' | jq -R . | jq -s .)
MISSING_JSON=$(printf '%s\n' "${MISSING_MODELS[@]:-}" | grep -v '^$' | jq -R . | jq -s .)
AGG_SUCCESS_JSON=$(printf '%s\n' "${AGG_SUCCESS[@]:-}" | grep -v '^$' | jq -R . | jq -s .)
AGG_FAILED_JSON=$(printf '%s\n' "${AGG_FAILED[@]:-}" | grep -v '^$' | jq -R . | jq -s .)

cat > "${SUMMARY_JSON}" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "run_id": "${RUN_ID:-auto}",
  "results_dir": "${RESULTS_DIR}",
  "base_models": {
    "validated": ${VALIDATED_JSON},
    "missing": ${MISSING_JSON}
  },
  "ensemble": {
    "requested": ${TRAIN_ENSEMBLE},
    "trained": ${ENSEMBLE_TRAINED:-0},
    "failed": ${ENSEMBLE_FAILED:-0}
  },
  "aggregation": {
    "successful": ${AGG_SUCCESS_JSON},
    "failed": ${AGG_FAILED_JSON}
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

if [[ ${#AGG_FAILED[@]:-0} -gt 0 ]]; then
  log_warning "Some models failed aggregation. Check log for details."
  exit 1
fi

log_success "All steps completed successfully"
exit 0
