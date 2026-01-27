#!/usr/bin/env bash
#============================================================
# hpc_optimize_panel.sh
#
# Panel optimization via RFE on HPC
# Reuses configs/pipeline_hpc.yaml for paths
#
# Usage:
#   cd analysis/
#   bash scripts/hpc_optimize_panel.sh --run-id 20260127_094454
#   bash scripts/hpc_optimize_panel.sh --run-id 20260127_094454 --model XGBoost --split-seed 5
#============================================================

set -euo pipefail
IFS=$'\n\t'

#==============================================================
# ARGUMENT PARSING
#==============================================================
RUN_ID=""
MODEL=""
SPLIT_SEED=0
START_SIZE=100
MIN_SIZE=5
MIN_AUROC_FRAC=0.90
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --split-seed)
      SPLIT_SEED="$2"
      shift 2
      ;;
    --start-size)
      START_SIZE="$2"
      shift 2
      ;;
    --min-size)
      MIN_SIZE="$2"
      shift 2
      ;;
    --min-auroc-frac)
      MIN_AUROC_FRAC="$2"
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: bash scripts/hpc_optimize_panel.sh --run-id RUN_ID [--model MODEL] [--split-seed N] [--config CONFIG]"
      exit 1
      ;;
  esac
done

if [[ -z "${RUN_ID}" ]]; then
  echo "ERROR: --run-id is required"
  exit 1
fi

#==============================================================
# SETUP
#==============================================================
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_DIR}"

# Load config
if [[ -z "${CONFIG_FILE}" ]]; then
  CONFIG_FILE="${BASE_DIR}/configs/pipeline_hpc.yaml"
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "ERROR: Config file not found: ${CONFIG_FILE}"
  exit 1
fi

# Helper: extract YAML value
get_yaml() {
  local file="$1"
  local key="$2"
  grep "^  ${key}:" "${file}" | sed 's/^[^:]*: *//' | tr -d '"' | tr -d "'" | sed 's/ *#.*//'
}

# Load paths from config
INFILE=$(get_yaml "${CONFIG_FILE}" "infile")
INFILE="${BASE_DIR}/${INFILE}"
SPLITS_DIR=$(get_yaml "${CONFIG_FILE}" "splits_dir")
SPLITS_DIR="${BASE_DIR}/${SPLITS_DIR}"
RESULTS_DIR=$(get_yaml "${CONFIG_FILE}" "results_dir")
RESULTS_DIR="${BASE_DIR}/${RESULTS_DIR}"

# Auto-detect model if not specified
if [[ -z "${MODEL}" ]]; then
  MODELS_AVAIL=()
  for model_dir in "${RESULTS_DIR}"/*/; do
    model_name=$(basename "${model_dir}")
    if [[ "${model_name}" == "investigations" ]] || [[ "${model_name}" == "ENSEMBLE" ]]; then
      continue
    fi
    if [[ -d "${model_dir}/run_${RUN_ID}" ]]; then
      MODELS_AVAIL+=("${model_name}")
    fi
  done

  if [[ ${#MODELS_AVAIL[@]} -eq 0 ]]; then
    echo "ERROR: No models found for run ${RUN_ID}"
    exit 1
  fi

  MODEL="${MODELS_AVAIL[0]}"
  echo "Auto-detected model: ${MODEL}"
fi

MODEL_PATH="${RESULTS_DIR}/${MODEL}/run_${RUN_ID}/split_seed${SPLIT_SEED}/core/${MODEL}__final_model.joblib"

# Verify model exists
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ERROR: Model not found: ${MODEL_PATH}"
  echo "Available models for run ${RUN_ID}:"
  find "${RESULTS_DIR}" -type d -name "run_${RUN_ID}" 2>/dev/null || echo "  None found"
  exit 1
fi

# Activate venv (support both venv and conda)
VENV_PATH="${BASE_DIR}/venv/bin/activate"
if [[ -f "${VENV_PATH}" ]]; then
  source "${VENV_PATH}"
elif [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
  echo "ERROR: No Python environment found. Either:"
  echo "  1. Create and activate a venv: python3 -m venv venv && source venv/bin/activate"
  echo "  2. Activate a Conda environment: conda activate <env_name>"
  exit 1
else
  echo "Using active Conda environment: ${CONDA_DEFAULT_ENV}"
fi

echo "============================================"
echo "Panel Optimization"
echo "============================================"
echo "Run ID: ${RUN_ID}"
echo "Model: ${MODEL}"
echo "Split seed: ${SPLIT_SEED}"
echo "Model path: ${MODEL_PATH}"
echo "Input file: ${INFILE}"
echo "Split dir: ${SPLITS_DIR}"
echo "Start size: ${START_SIZE}"
echo "Min size: ${MIN_SIZE}"
echo "Min AUROC fraction: ${MIN_AUROC_FRAC}"
echo "============================================"

# Run panel optimization
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting panel optimization..."

ced optimize-panel \
  --model-path "${MODEL_PATH}" \
  --infile "${INFILE}" \
  --split-dir "${SPLITS_DIR}" \
  --split-seed "${SPLIT_SEED}" \
  --start-size "${START_SIZE}" \
  --min-size "${MIN_SIZE}" \
  --min-auroc-frac "${MIN_AUROC_FRAC}" \
  --verbose 2

EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
  OUTDIR="${RESULTS_DIR}/${MODEL}/run_${RUN_ID}/split_seed${SPLIT_SEED}/optimize_panel"
  echo ""
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] Panel optimization completed"
  echo "Results saved to: ${OUTDIR}"
  echo ""

  if [[ -d "${OUTDIR}" ]]; then
    echo "Generated files:"
    ls -lh "${OUTDIR}/"
  fi
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [FAILED] Panel optimization failed (exit code: ${EXIT_CODE})"
  exit ${EXIT_CODE}
fi
