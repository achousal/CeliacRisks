#!/usr/bin/env bash
#============================================================
# post_training_pipeline.sh
#
# Comprehensive post-training pipeline with detailed logging:
# 1. Validate base model outputs
# 2. Train ensemble (if 2+ models available)
# 3. Aggregate results across splits
# 4. Generate validation reports
#
# Usage:
#   cd analysis/
#   bash scripts/post_training_pipeline.sh --run-id 20260122_120000
#   bash scripts/post_training_pipeline.sh --run-id 20260122_120000 --skip-ensemble
#============================================================

set -euo pipefail
IFS=$'\n\t'

#==============================================================
# ARGUMENT PARSING
#==============================================================
TRAIN_ENSEMBLE=1
SKIP_ENSEMBLE=0
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
    --skip-ensemble)
      SKIP_ENSEMBLE=1
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
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/..") && pwd)"
PROJECT_ROOT="$(cd "${BASE_DIR}/.." && pwd)"
cd "${BASE_DIR}"

# Timestamped log file for this post-processing run
# Use separate post directory: logs/post/run_{RUN_ID} (at project root, not inside analysis/)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [[ -n "${RUN_ID}" ]]; then
  POST_LOG_DIR="${PROJECT_ROOT}/logs/post/run_${RUN_ID}"
else
  # Fallback for auto run ID
  RUN_ID="${TIMESTAMP}"
  POST_LOG_DIR="${PROJECT_ROOT}/logs/post/run_${RUN_ID}"
fi

log_success "All steps completed successfully"
exit 0
