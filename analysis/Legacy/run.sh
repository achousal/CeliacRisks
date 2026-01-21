#!/usr/bin/env bash
#============================================================
# run.sh
#
# OPTIMIZED PIPELINE ORCHESTRATION
#
# Pipeline overview:
#   - Uses save_splits.py for holdout-aware, repeated development splits.
#   - Submits CeD_optimized.lsf for the four production models (RF, XGBoost, LinSVM_cal, LR_EN).
#   - Runs IncidentPlusPrevalent with Prevalent in TRAIN only (VAL/TEST incident-only).
#   - Supports single-split, holdout, and repeated-split modes via environment knobs.
#
# Usage:
#   # Standard run (single split, no holdout)
#   ./run.sh
#
#   # Production run (holdout + single development split)
#   MODE=holdout ./run.sh
#
#   # Full robust run (holdout + 10)
#   MODE=holdout N_SPLITS=10 ./run.sh
#
#   # Dry run
#   DRY_RUN=1 ./run.sh
#
#   # Run a subset of models (comma- or space-separated)
#   RUN_MODELS=RF,LR ./run.sh
#
#   # Run postprocessing NOW on completed jobs (visualize progress)
#   POSTPROCESS_NOW=1 ./run.sh
#   # Or skip new job submission entirely:
#   POSTPROCESS_ONLY=1 ./run.sh
#
#   # Skip multi-model postprocessing submission (useful for single-model runs)
#   SKIP_MULTIPLE_POSTPROCESS=1 ./run.sh
#
#   # Aggregate splits for a single model (requires all splits complete)
#   COMPARE_MODEL_SPLITS=1 ./run.sh
#============================================================

set -euo pipefail
IFS=$'\n\t'

log(){ echo "[$(date +'%F %T')] $*"; }
die(){ echo "ERROR: $*" >&2; exit 1; }
need_file(){ [[ -f "$1" ]] || die "Missing file: $1"; }
need_dir(){ [[ -d "$1" ]] || die "Missing directory: $1"; }

# Check if a model run is already complete (has test_metrics.csv)
# Usage: is_model_complete MODEL_NAME SEED RESULTS_BASE
# Returns 0 if complete, 1 if not
is_model_complete() {
  local MODEL_NAME="$1"
  local SEED="$2"
  local RESULTS_BASE="$3"
  local MODEL_FULL=""

  case "${MODEL_NAME}" in
    RF)  MODEL_FULL="RF" ;;
    XGB) MODEL_FULL="XGBoost" ;;
    SVM) MODEL_FULL="LinSVM_cal" ;;
    LR)  MODEL_FULL="LR_EN" ;;
    *)   MODEL_FULL="${MODEL_NAME}" ;;
  esac

  # Build glob pattern to find any completed run (any FOLDS×REPEATS)
  local SEED_SUFFIX=""
  [[ -n "${SEED}" ]] && SEED_SUFFIX="__seed${SEED}"

  local PATTERN="${RESULTS_BASE}/IncidentPlusPrevalent__${MODEL_FULL}__*__val${VAL_SIZE}__test${TEST_SIZE}__hybrid${SEED_SUFFIX}/core/test_metrics.csv"

  # Check if any matching file exists
  if compgen -G "${PATTERN}" > /dev/null 2>&1; then
    return 0  # Complete
  else
    return 1  # Not complete
  fi
}

model_fully_complete() {
  local MODEL_NAME="$1"
  local RESULTS_BASE="$2"

  if [[ "${N_SPLITS}" -gt 1 ]]; then
    for ((i=0; i<N_SPLITS; i++)); do
      local CURRENT_SEED=$((SEED_START + i))
      if ! is_model_complete "${MODEL_NAME}" "${CURRENT_SEED}" "${RESULTS_BASE}"; then
        return 1
      fi
    done
  else
    if ! is_model_complete "${MODEL_NAME}" "" "${RESULTS_BASE}"; then
      return 1
    fi
  fi
  return 0
}

normalize_model_name() {
  local raw="$1"
  local upper="${raw^^}"
  case "${upper}" in
    RF) echo "RF" ;;
    XGB|XGBOOST) echo "XGB" ;;
    SVM|LINSVM|LINSVM_CAL) echo "SVM" ;;
    LR|LR_EN|LREN) echo "LR" ;;
    *) return 1 ;;
  esac
}

model_idx_from_name() {
  case "$1" in
    RF) echo "1" ;;
    XGB) echo "2" ;;
    SVM) echo "3" ;;
    LR) echo "4" ;;
    *) return 1 ;;
  esac
}

model_log_name() {
  case "$1" in
    RF) echo "RF" ;;
    XGB) echo "XGBoost" ;;
    SVM) echo "LinSVM" ;;
    LR) echo "LR_EN" ;;
    *) echo "$1" ;;
  esac
}

model_full_name() {
  case "$1" in
    RF) echo "RF" ;;
    XGB) echo "XGBoost" ;;
    SVM) echo "LinSVM_cal" ;;
    LR) echo "LR_EN" ;;
    *) echo "$1" ;;
  esac
}
#==============================================================#
#
#
#==============================================================#
#
#
#==============================================================#
#
# ---------------- user config ----------------
PROJECT="acc_Chipuk_Laboratory"
QUEUE="premium"

BASE_DIR="/sc/arion/projects/Chipuk_Laboratory/chousa01/CeliacRiskML/sklearn"
INFILE="${BASE_DIR}/../Celiac_dataset_proteomics.csv"

SPLITS_DIR="${BASE_DIR}/splits_0-9"
RESULTS_DIR="${BASE_DIR}/results_LR_1.18.26_a"
LOGS_DIR="${BASE_DIR}/logs_LR_1.18.26_a"

PY_SAVE_SPLITS="${BASE_DIR}/save_splits.py"
PY_POSTPROC="${BASE_DIR}/postprocess_compare.py"
PY_SCRIPT="${BASE_DIR}/celiacML_faith.py"
PY_DASHBOARD="${BASE_DIR}/generate_dashboard.py"
R_VIZ="${BASE_DIR}/compare_models_faith.R"
ARRAY_LSF="${BASE_DIR}/CeD_optimized.lsf"

#==============================================================
# MODEL-SPECIFIC RESOURCE CONFIGURATION
#==============================================================
#   Options: RF, XGBoost (or XGB), LinSVM_cal (or SVM), LR_EN (or LR)

RUN_MODELS="${RUN_MODELS:-LR}"

# ============================================================
# RF: Random Forest
# ============================================================
RF_CPUS="${RF_CPUS:-8}"
RF_MEM="${RF_MEM:-8000}"
RF_WALLTIME="${RF_WALLTIME:-144:00}"
RF_EXTRA_RESOURCES="${RF_EXTRA_RESOURCES:-select[mem>192000]}"  # Prefer high-mem nodes

# ============================================================
# XGBoost: Gradient Boosting
# ============================================================
XGB_CPUS="${XGB_CPUS:-8}"
XGB_MEM="${XGB_MEM:-4000}"
XGB_WALLTIME="${XGB_WALLTIME:-144:00}" 

# ============================================================
# LinSVM_cal: Linear SVM with Sigmoid Calibration
# ============================================================
SVM_CPUS="${SVM_CPUS:-8}"
SVM_MEM="${SVM_MEM:-2000}"
SVM_WALLTIME="${SVM_WALLTIME:-144:00}"

# ============================================================
# LR_EN: Logistic Regression with ElasticNet Regularization
# ============================================================
LR_CPUS="${LR_CPUS:-8}"
LR_MEM="${LR_MEM:-2000}"
LR_WALLTIME="${LR_WALLTIME:-144:00}"

# ============================================================
# SPLIT & SCENARIO CONFIGURATION
# ============================================================
MODE="${MODE:-development}" #   "development","holdout"
SEED_START="${SEED_START:-0}"
N_SPLITS="${N_SPLITS:-10}"
TEST_SIZE="${TEST_SIZE:-0.25}"
VAL_SIZE="${VAL_SIZE:-0.25}"
HOLDOUT_SIZE="${HOLDOUT_SIZE:-0.30}"
TRAIN_INCIDENT_ONLY="${TRAIN_INCIDENT_ONLY:-0}"
EVAL_CONTROL_PER_CASE="${EVAL_CONTROL_PER_CASE:-5}"

# ============================================================
# POSTPROCESSING ANALYSIS PARAMETERS
POSTPROCESS_NOW="${POSTPROCESS_NOW:-0}"     # Run postprocessing immediately on completed jobs
POSTPROCESS_ONLY="${POSTPROCESS_ONLY:-0}"   # Skip job submission, only run postprocessing
SKIP_MULTIPLE_POSTPROCESS="${SKIP_MULTIPLE_POSTPROCESS:-1}"  # Skip multi-model postprocessing
COMPARE_MODEL_SPLITS="${COMPARE_MODEL_SPLITS:-1}"            # Run single-model split aggregation
INDIVIDUALPROCESS_NOW="${INDIVIDUALPROCESS_NOW:-0}"          # Regenerate individual model plots

# DCA parameters (Bootstrap iterations for CIs defined in CeD_optimized.lsf)
DCA_THRESHOLD_MIN="0.0005"                          # Min threshold for DCA curve
DCA_THRESHOLD_MAX="1.0"                             # Max threshold for DCA curve - 1.0 to find zero crossing
DCA_THRESHOLD_STEP="0.001"                          # Step size for DCA sweep
DCA_REPORT_POINTS="${DCA_REPORT_POINTS:-0.005,0.01,0.02,0.05}"  # Key thresholds for DCA summary reports
DCA_MAX_PT="1.0"                                   # Max threshold for DCA visualization in compare_models_faith.R (20%)
DCA_STEP="0.005"                                    # Step size for DCA visualization
CALIB_BINS="${CALIB_BINS:-10}"                      # Number of bins for calibration plots

# Visualization parameters (used by compare_models_faith.R)
TOP_FEATURES=25                                      # Number of top features to display
SPEC_TARGETS="${SPEC_TARGETS:-0.95,0.99,0.995}"     # Specificity targets for performance summaries

# Splits control
OVERWRITE_SPLITS="${OVERWRITE_SPLITS:-0}"

#################################


RUN_MODELS_ARR=()
OLD_IFS="$IFS"
IFS=', '
read -r -a RUN_MODELS_RAW_ARR <<< "${RUN_MODELS}"
IFS="$OLD_IFS"
for model in "${RUN_MODELS_RAW_ARR[@]}"; do
  [[ -z "${model}" ]] && continue
  model_norm="$(normalize_model_name "${model}")" || die "Unknown model in RUN_MODELS: ${model}"
  if [[ " ${RUN_MODELS_ARR[*]} " != *" ${model_norm} "* ]]; then
    RUN_MODELS_ARR+=("${model_norm}")
  fi
done
MODEL_COUNT="${#RUN_MODELS_ARR[@]}"
[[ "${MODEL_COUNT}" -gt 0 ]] || die "RUN_MODELS did not resolve to any models."
RUN_MODELS_LABEL="$(IFS=','; echo "${RUN_MODELS_ARR[*]}")"
MODEL_GREP_PARTS=()
for model in "${RUN_MODELS_ARR[@]}"; do
  MODEL_GREP_PARTS+=("CeD_${model}")
done
MODEL_GREP="$(IFS='|'; echo "${MODEL_GREP_PARTS[*]}")"

# --------------------------------------------

# Validate required files and directories
need_dir "${BASE_DIR}"
need_file "${INFILE}"
need_file "${PY_SAVE_SPLITS}"
need_file "${PY_POSTPROC}"
need_file "${R_VIZ}"
need_file "${ARRAY_LSF}"

mkdir -p "${LOGS_DIR}" "${SPLITS_DIR}" "${RESULTS_DIR}"

# conda/module bootstrap
ACTIVATE_SNIPPET='
if command -v module >/dev/null 2>&1; then
  module load anaconda3 >/dev/null 2>&1 || module load anaconda >/dev/null 2>&1 || true
fi
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate csb || conda activate base
fi
'

if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
  log "[DRY_RUN=1] Will not execute, only print planned actions."
fi

#==============================================================
# POSTPROCESS NOW FUNCTION
#==============================================================
# Runs all postprocessing scripts immediately on completed jobs
# This is useful for visualizing progress while jobs are still running
run_postprocess_now() {
  local RESULTS_BASE="$1"

  log "============================================"
  log "POSTPROCESS NOW - Visualizing completed jobs"
  log "============================================"

  # Count completed model runs
  local COMPLETED_COUNT=0
  local TOTAL_EXPECTED=0
  local COMPLETED_MODELS=()

  if [[ "${N_SPLITS}" -gt 1 ]]; then
    TOTAL_EXPECTED=$((N_SPLITS * MODEL_COUNT))  # selected models × N_SPLITS seeds
    for ((i=0; i<N_SPLITS; i++)); do
      CURRENT_SEED=$((SEED_START + i))
      for MODEL in "${RUN_MODELS_ARR[@]}"; do
        if is_model_complete "${MODEL}" "${CURRENT_SEED}" "${RESULTS_BASE}"; then
          ((COMPLETED_COUNT++)) || true
          COMPLETED_MODELS+=("${MODEL}_seed${CURRENT_SEED}")
        fi
      done
    done
  else
    TOTAL_EXPECTED="${MODEL_COUNT}"
    for MODEL in "${RUN_MODELS_ARR[@]}"; do
      if is_model_complete "${MODEL}" "" "${RESULTS_BASE}"; then
        ((COMPLETED_COUNT++)) || true
        COMPLETED_MODELS+=("${MODEL}")
      fi
    done
  fi

  log "Progress: ${COMPLETED_COUNT}/${TOTAL_EXPECTED} model runs complete"

  if [[ ${COMPLETED_COUNT} -eq 0 ]]; then
    log "No completed jobs found. Nothing to postprocess."
    return 1
  fi

  log "Completed models: ${COMPLETED_MODELS[*]}"
  log ""

  # ALWAYS run postprocess_compare.py (creates COMBINED/ folder with cross-model metrics)
  log "Running postprocess_compare.py..."
  local POSTPROC_CMD="cd ${BASE_DIR} && python -u ${PY_POSTPROC} \
    --results_dir ${RESULTS_BASE} \
    --dca_threshold_min ${DCA_THRESHOLD_MIN} \
    --dca_threshold_max ${DCA_THRESHOLD_MAX} \
    --dca_threshold_step ${DCA_THRESHOLD_STEP} \
    --dca_report_points \"${DCA_REPORT_POINTS}\" \
    --dca_max_pt ${DCA_MAX_PT} \
    --dca_step ${DCA_STEP} \
    --calib_bins ${CALIB_BINS}"

  if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
    echo "${POSTPROC_CMD}"
  else
    eval "${ACTIVATE_SNIPPET}"
    eval "${POSTPROC_CMD}" || log "WARNING: postprocess_compare.py failed (may need more completed jobs)"
  fi

  # CONDITIONALLY run compare_models_faith.R based on SKIP_MULTIPLE_POSTPROCESS
  if [[ "${SKIP_MULTIPLE_POSTPROCESS:-0}" -eq 1 ]]; then
    log ""
    log "SKIP_MULTIPLE_POSTPROCESS=1: Skipping compare_models_faith.R"
  else
    # Run compare_models_faith.R
    log ""
    log "Running compare_models_faith.R..."
    # Load R module to ensure tidyverse packages (readr, dplyr, etc.) are available
    local R_MODULE_SNIPPET='
if command -v module >/dev/null 2>&1; then
  module load R/4.3.0 2>/dev/null || module load R/4.2.0 2>/dev/null || module load R 2>/dev/null || true
fi
'
    local VIZ_CMD="cd ${BASE_DIR} && ${R_MODULE_SNIPPET} Rscript ${R_VIZ} \
      --results_root ${RESULTS_BASE} \
      --outdir ${RESULTS_BASE}/compare_figs \
      --top_features ${TOP_FEATURES} \
      --dca_max_pt ${DCA_MAX_PT} \
      --dca_step ${DCA_STEP} \
      --calib_bins ${CALIB_BINS} \
      --spec_targets ${SPEC_TARGETS}"

    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
      echo "${VIZ_CMD}"
    else
      eval "${VIZ_CMD}" || log "WARNING: compare_models_faith.R failed (may need COMBINED/ from postprocess or missing R packages)"
    fi
  fi

  if [[ "${COMPARE_MODEL_SPLITS:-0}" -eq 1 ]]; then
    log ""
    log "Running single-model split summaries..."
    for MODEL in "${RUN_MODELS_ARR[@]}"; do
      if model_fully_complete "${MODEL}" "${RESULTS_BASE}"; then
        MODEL_FULL="$(model_full_name "${MODEL}")"
        local SINGLE_CMD="cd ${BASE_DIR} && python -u ${PY_POSTPROC} \
          --mode single \
          --results_dir ${RESULTS_BASE} \
          --model ${MODEL_FULL} \
          --expected_splits ${N_SPLITS} \
          --single_outdir ${RESULTS_BASE}/${MODEL_FULL}_COMBINED"
        if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
          echo "${SINGLE_CMD}"
        else
          eval "${ACTIVATE_SNIPPET}"
          eval "${SINGLE_CMD}" || log "WARNING: single-model postprocess failed for ${MODEL_FULL}"
        fi
      else
        log "Skipping single-model summary for ${MODEL} (not all splits complete)"
      fi
    done
  fi

  # Run generate_dashboard.py
  log ""
  log "Running generate_dashboard.py..."
  local DASHBOARD_CMD="cd ${BASE_DIR} && python -u ${PY_DASHBOARD} \
    --results_dir ${RESULTS_BASE} \
    --outfile ${RESULTS_BASE}/dashboard_progress.html"

  if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
    echo "${DASHBOARD_CMD}"
  else
    if [[ -f "${PY_DASHBOARD}" ]]; then
      eval "${DASHBOARD_CMD}" || log "WARNING: generate_dashboard.py failed"
    else
      log "SKIPPED: generate_dashboard.py not found at ${PY_DASHBOARD}"
    fi
  fi

  log ""
  log "============================================"
  log "POSTPROCESS COMPLETE"
  log "============================================"
  log "Progress: ${COMPLETED_COUNT}/${TOTAL_EXPECTED} jobs ($(awk "BEGIN {printf \"%.0f\", ${COMPLETED_COUNT}/${TOTAL_EXPECTED}*100}")%)"
  log ""
  log "Output locations:"
  if [[ "${SKIP_MULTIPLE_POSTPROCESS:-0}" -ne 1 ]]; then
    log "  COMBINED metrics: ${RESULTS_BASE}/COMBINED/"
    log "  R visualizations: ${RESULTS_BASE}/compare_figs/"
  fi
  if [[ "${COMPARE_MODEL_SPLITS:-0}" -eq 1 ]]; then
    log "  Single-model outputs: ${RESULTS_BASE}/*_COMBINED/"
  fi
  log "  HTML dashboard:   ${RESULTS_BASE}/dashboard_progress.html"
  log "============================================"

  return 0
}

#==============================================================
# INDIVIDUALPROCESS NOW FUNCTION
#==============================================================
# Regenerate plots for individual completed model runs
run_individualprocess_now() {
  local RESULTS_BASE="$1"

  log "============================================"
  log "INDIVIDUALPROCESS NOW - Regenerating plots for completed runs"
  log "============================================"

  # Find all completed model runs
  local COMPLETED_RUNS=()
  local REGENERATED_COUNT=0

  # Pattern: IncidentPlusPrevalent__*__*x*__val*__test*/core/test_metrics.csv
  while IFS= read -r -d '' metrics_file; do
    RUN_DIR="$(dirname "$(dirname "$metrics_file")")"
    COMPLETED_RUNS+=("${RUN_DIR}")
  done < <(find "${RESULTS_BASE}" -type f -name "test_metrics.csv" -path "*/core/*" -print0)

  log "Found ${#COMPLETED_RUNS[@]} completed run(s)"

  if [[ ${#COMPLETED_RUNS[@]} -eq 0 ]]; then
    log "No completed runs found. Nothing to regenerate."
    return 1
  fi

  # Regenerate plots for each run
  for RUN_DIR in "${COMPLETED_RUNS[@]}"; do
    RUN_NAME="$(basename "${RUN_DIR}")"
    log "Regenerating plots for: ${RUN_NAME}"

    REGEN_CMD="cd ${BASE_DIR} && python -u ${PY_SCRIPT} \
      --regenerate_plots \
      --run_dir ${RUN_DIR} \
      --force_overwrite"

    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
      echo "${REGEN_CMD}"
    else
      eval "${ACTIVATE_SNIPPET}"
      if eval "${REGEN_CMD}"; then
        ((REGENERATED_COUNT++)) || true
      else
        log "WARNING: Plot regeneration failed for ${RUN_NAME}"
      fi
    fi
  done

  log ""
  log "============================================"
  log "INDIVIDUALPROCESS COMPLETE"
  log "============================================"
  log "Regenerated plots for: ${REGENERATED_COUNT}/${#COMPLETED_RUNS[@]} runs"
  log "============================================"

  return 0
}

# ---------- helpers: split presence ----------
# OPTIMIZED: Check IncidentPlusPrevalent splits (VAL/TEST included)
split_paths=()
if [[ "${N_SPLITS}" -eq 1 ]]; then
  split_paths+=(
    "${SPLITS_DIR}/IncidentPlusPrevalent_train_idx.csv"
    "${SPLITS_DIR}/IncidentPlusPrevalent_val_idx.csv"
    "${SPLITS_DIR}/IncidentPlusPrevalent_test_idx.csv"
  )
else
  for ((i=0; i<N_SPLITS; i++)); do
    seed=$((SEED_START + i))
    split_paths+=(
      "${SPLITS_DIR}/IncidentPlusPrevalent_train_idx_seed${seed}.csv"
      "${SPLITS_DIR}/IncidentPlusPrevalent_val_idx_seed${seed}.csv"
      "${SPLITS_DIR}/IncidentPlusPrevalent_test_idx_seed${seed}.csv"
    )
  done
fi

if [[ "${MODE}" == "holdout" ]]; then
  split_paths+=("${SPLITS_DIR}/IncidentPlusPrevalent_HOLDOUT_idx.csv")
fi

have_all_splits() {
  local f
  for f in "${split_paths[@]}"; do
    [[ -s "$f" ]] || return 1
  done
  return 0
}

# ---------- 1) generate splits ----------
if have_all_splits && [[ "${OVERWRITE_SPLITS}" -eq 0 ]]; then
  log "Step 1/4: Reusing existing splits in ${SPLITS_DIR} (set OVERWRITE_SPLITS=1 to regenerate)"
else
  if have_all_splits && [[ "${OVERWRITE_SPLITS}" -eq 1 ]]; then
    log "Step 1/4: OVERWRITE_SPLITS=1 -> removing existing split files before regeneration"
    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
      printf 'rm -f %q\n' "${split_paths[@]}"
    else
      rm -f "${split_paths[@]}"
    fi
  else
    log "Step 1/4: Split files missing -> generating new splits into ${SPLITS_DIR}"
  fi

  SPLIT_CMD="cd ${BASE_DIR} && ${ACTIVATE_SNIPPET} python -u ${PY_SAVE_SPLITS} \
    --infile ${INFILE} \
    --outdir ${SPLITS_DIR} \
    --mode ${MODE} \
    --scenarios IncidentPlusPrevalent \
    --n_splits ${N_SPLITS} \
    --val_size ${VAL_SIZE} \
    --test_size ${TEST_SIZE} \
    --holdout_size ${HOLDOUT_SIZE} \
    --seed_start ${SEED_START} \
    --prevalent_train_only \
    --prevalent_train_frac 0.5 \
    --train_control_per_case 5"

  if [[ "${TRAIN_INCIDENT_ONLY}" -eq 1 ]]; then
    SPLIT_CMD="${SPLIT_CMD} --train_controls_incident_only"
  fi
  if [[ -n "${EVAL_CONTROL_PER_CASE}" ]]; then
    SPLIT_CMD="${SPLIT_CMD} --eval_control_per_case ${EVAL_CONTROL_PER_CASE}"
  fi

  if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
    echo "${SPLIT_CMD}"
  else
    bash -lc "${SPLIT_CMD}"
  fi
fi

# Verify split files exist now
for f in "${split_paths[@]}"; do
  need_file "$f"
done
log "Split files verified."

if [[ "${MODE}" == "holdout" ]]; then
  log "WARNING: Holdout set created. Do NOT use for model development!"
fi

# ---------- POSTPROCESS_ONLY mode ----------
if [[ "${POSTPROCESS_ONLY:-0}" -eq 1 ]]; then
  if [[ "${SKIP_MULTIPLE_POSTPROCESS:-0}" -eq 1 && "${COMPARE_MODEL_SPLITS:-0}" -ne 1 ]]; then
    log "SKIP_MULTIPLE_POSTPROCESS=1 overrides POSTPROCESS_ONLY; skipping postprocessing and exiting."
    exit 0
  fi
  log "POSTPROCESS_ONLY=1: Skipping job submission, running postprocessing on completed jobs"
  run_postprocess_now "${RESULTS_DIR}"
  exit $?
fi

# ---------- 2) submit model jobs ----------
log "Step 2/4: Submit model jobs"

# Export environment variables for LSF script
export ROOT="${BASE_DIR}"
export INFILE="${INFILE}"
export SPLITS_DIR="${SPLITS_DIR}"
export RESULTS_ROOT="${RESULTS_DIR}"
export RESULTS_DIR="${RESULTS_DIR}"
export LOGS_DIR="${LOGS_DIR}"
export PY_SCRIPT="${PY_SCRIPT}"
export TEST_SIZE="${TEST_SIZE}"
export VAL_SIZE="${VAL_SIZE}"

log "Exported env vars: ROOT=${ROOT}, SPLITS_DIR=${SPLITS_DIR}, RESULTS_ROOT=${RESULTS_ROOT}"

# Helper: parse job id robustly from bsub output (handles warnings, etc.)
parse_jobid() {
  # Extract first occurrence of "Job <12345>"
  grep -oE 'Job <[0-9]+>' | head -n1 | tr -cd '0-9'
}

#==============================================================
# MODEL-SPECIFIC JOB SUBMISSION
#==============================================================
# Models: RF=1, XGBoost=2, LinSVM_cal=3, LR_EN=4
# Each model gets tailored resources for optimal performance

submit_model_job() {
  local MODEL_NAME="$1"
  local MODEL_IDX="$2"
  local SEED="$3"
  local CPUS="$4"
  local MEM="$5"
  local WALLTIME="$6"
  local QUEUE_TO_USE="$7"
  local EXTRA_RESOURCES="${8:-}"

  local JOB_NAME="CeD_${MODEL_NAME}"
  [[ -n "${SEED}" ]] && JOB_NAME="${JOB_NAME}_seed${SEED}"

  # Build resource list (one -R per requirement for LSF compatibility)
  local RESOURCE_OPTS=("span[hosts=1]" "rusage[mem=${MEM}]")
  if [[ -n "${EXTRA_RESOURCES}" ]]; then
    # Split by whitespace to support multiple tokens (e.g., "select[mem>192000] affinity[core]")
    read -r -a EXTRA_ARRAY <<< "${EXTRA_RESOURCES}"
    for res in "${EXTRA_ARRAY[@]}"; do
      [[ -n "${res}" ]] && RESOURCE_OPTS+=("${res}")
    done
  fi

  log "  Submitting ${MODEL_NAME} (idx=${MODEL_IDX}): queue=${QUEUE_TO_USE}, cpus=${CPUS}, mem=${MEM}MB, walltime=${WALLTIME}" >&2

  if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
    local DRY_CMD="bsub -P ${PROJECT} -q ${QUEUE_TO_USE} -J ${JOB_NAME} -n ${CPUS} -W ${WALLTIME}"
    for res in "${RESOURCE_OPTS[@]}"; do
      DRY_CMD+=" -R \"${res}\""
    done
    DRY_CMD+=" -cwd ${BASE_DIR} < ${ARRAY_LSF}"
    echo "${DRY_CMD}"
    echo "DRYRUN_${MODEL_NAME}_${SEED:-0}"
    return
  fi

  # Build bsub command
  # Use -J "jobname[1-1]" to create a 1-job array with LSB_JOBINDEX=1
  # This is mapped based on MODEL_IDX in the LSF script
  local BSUB_CMD=(
    bsub -P "${PROJECT}" -q "${QUEUE_TO_USE}"
    -J "${JOB_NAME}[${MODEL_IDX}-${MODEL_IDX}]"
    -n "${CPUS}"
    -W "${WALLTIME}"
    -cwd "${BASE_DIR}"
    -oo "${LOGS_DIR}/${JOB_NAME}.%J.%I.out"
    -eo "${LOGS_DIR}/${JOB_NAME}.%J.%I.err"
  )
  for res in "${RESOURCE_OPTS[@]}"; do
    BSUB_CMD+=(-R "${res}")
  done

  SUB_OUT="$(
    "${BSUB_CMD[@]}" < "${ARRAY_LSF}" 2>&1
  )"
  echo "${SUB_OUT}" >&2

  if echo "${SUB_OUT}" | grep -qiE "Request aborted|not submitted|esub"; then
    die "Job submission rejected for ${MODEL_NAME}. See bsub output above."
  fi

  local JOB_ID
  JOB_ID="$(echo "${SUB_OUT}" | parse_jobid)"
  [[ -n "${JOB_ID}" ]] || die "Could not parse JOB_ID for ${MODEL_NAME}."

  echo "${JOB_ID}"
}

submit_selected_model() {
  local MODEL="$1"
  local SEED="$2"
  local PREFIX="$3"
  local IDX
  local LOG_NAME
  local CPUS
  local MEM
  local WALLTIME
  local EXTRA=""

  IDX="$(model_idx_from_name "${MODEL}")" || die "Unknown model code: ${MODEL}"
  LOG_NAME="$(model_log_name "${MODEL}")"

  case "${MODEL}" in
    RF)
      CPUS="${RF_CPUS}"
      MEM="${RF_MEM}"
      WALLTIME="${RF_WALLTIME}"
      EXTRA="${RF_EXTRA_RESOURCES}"
      ;;
    XGB)
      CPUS="${XGB_CPUS}"
      MEM="${XGB_MEM}"
      WALLTIME="${XGB_WALLTIME}"
      ;;
    SVM)
      CPUS="${SVM_CPUS}"
      MEM="${SVM_MEM}"
      WALLTIME="${SVM_WALLTIME}"
      ;;
    LR)
      CPUS="${LR_CPUS}"
      MEM="${LR_MEM}"
      WALLTIME="${LR_WALLTIME}"
      ;;
  esac

  if is_model_complete "${MODEL}" "${SEED}" "${RESULTS_DIR}"; then
    log "${PREFIX}${LOG_NAME}: SKIPPED (already complete)"
    ((SKIPPED_MODELS++)) || true
    return
  fi

  if [[ "${MODEL}" == "XGB" ]]; then
    export XGB_TREE_METHOD="hist"
  fi

  JOB_ID=$(submit_model_job "${MODEL}" "${IDX}" "${SEED}" "${CPUS}" "${MEM}" "${WALLTIME}" "${QUEUE}" "${EXTRA}")
  [[ -n "${JOB_ID}" ]] || die "${LOG_NAME} submission failed${SEED:+ for seed ${SEED}}"
  ALL_JOB_IDS+=("${JOB_ID}")
  if [[ "${MODEL}" == "RF" && -z "${SEED}" ]]; then
    log "${PREFIX}${LOG_NAME}: ${JOB_ID} (high-memory: ${MEM}MB/CPU)"
  else
    log "${PREFIX}${LOG_NAME}: ${JOB_ID}"
  fi
}

ALL_JOB_IDS=()
SKIPPED_MODELS=0

# Submit jobs for each seed (or single submission if N_SPLITS=1)
if [[ "${N_SPLITS}" -gt 1 ]]; then
  log "Repeated splits mode: checking ${N_SPLITS} seeds × ${MODEL_COUNT} model(s) (${RUN_MODELS_LABEL})"
  export SPLIT_SEED_START="${SEED_START}"
  export SPLIT_SEED_END=$((SEED_START + N_SPLITS - 1))

  for ((i=0; i<N_SPLITS; i++)); do
    CURRENT_SEED=$((SEED_START + i))
    export CURRENT_SPLIT_SEED="${CURRENT_SEED}"

    log "Seed ${CURRENT_SEED} ($((i+1))/${N_SPLITS}):"
    for MODEL in "${RUN_MODELS_ARR[@]}"; do
      submit_selected_model "${MODEL}" "${CURRENT_SEED}" "    "
    done
  done

else
  # Single split mode
  log "Single split mode: checking ${MODEL_COUNT} model job(s) (${RUN_MODELS_LABEL})"
  CURRENT_SEED="${SEED_START}"
  export CURRENT_SPLIT_SEED="${CURRENT_SEED}"

  # For single split, use empty seed in completion check (matches no __seedN suffix)
  SEED_FOR_CHECK=""
  for MODEL in "${RUN_MODELS_ARR[@]}"; do
    submit_selected_model "${MODEL}" "${SEED_FOR_CHECK}" "  "
  done
fi

log "Skipped ${SKIPPED_MODELS} already-complete model(s)"

# Build dependency expression for all submitted jobs
DEP=""
for jid in "${ALL_JOB_IDS[@]}"; do
  [[ -z "$DEP" ]] && DEP="done(${jid})" || DEP="${DEP} && done(${jid})"
done
ARRAY_DEPENDENCY_EXPR="${DEP}"
ARRAY_DEPENDENCY_HUMAN="$(IFS=','; echo "${ALL_JOB_IDS[*]}")"

log "Model jobs submitted: ${#ALL_JOB_IDS[@]} (skipped: ${SKIPPED_MODELS})"
if [[ -z "${ARRAY_DEPENDENCY_EXPR}" ]]; then
  log "No pending model jobs - all complete. Running all postprocessing locally..."
  run_postprocess_now "${RESULTS_DIR}"
  exit 0
fi

log "Dependency expr: ${ARRAY_DEPENDENCY_EXPR}"

if [[ "${SKIP_MULTIPLE_POSTPROCESS:-0}" -eq 1 ]]; then
  log "SKIP_MULTIPLE_POSTPROCESS=1: Skipping postprocess_compare.py and compare_models_faith.R submission"
  POST_JOB_ID="SKIPPED"
  VIZ_JOB_ID="SKIPPED"
else
  # ---------- 3) submit postprocess (dependent or immediate) ----------
  log "Step 3/4: Submit postprocess_compare.py"

  POSTPROC_CMD="cd ${BASE_DIR} && ${ACTIVATE_SNIPPET} python -u ${PY_POSTPROC} \
    --results_dir ${RESULTS_DIR} \
    --dca_threshold_min ${DCA_THRESHOLD_MIN} \
    --dca_threshold_max ${DCA_THRESHOLD_MAX} \
    --dca_threshold_step ${DCA_THRESHOLD_STEP} \
    --dca_report_points \"${DCA_REPORT_POINTS}\" \
    --dca_max_pt ${DCA_MAX_PT} \
    --dca_step ${DCA_STEP} \
    --calib_bins ${CALIB_BINS}"

  if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
    if [[ -n "${ARRAY_DEPENDENCY_EXPR}" ]]; then
      echo "bsub -P ${PROJECT} -w \"${ARRAY_DEPENDENCY_EXPR}\" -J postproc_opt -q ${QUEUE} -n 4 -W 2:00 -R \"span[hosts=1] rusage[mem=8000]\" \"${POSTPROC_CMD}\""
    else
      echo "bsub -P ${PROJECT} -J postproc_opt -q ${QUEUE} -n 4 -W 2:00 -R \"span[hosts=1] rusage[mem=8000]\" \"${POSTPROC_CMD}\""
    fi
    POST_JOB_ID="DRYRUN_POST"
  else
    # Build bsub command - with or without dependency
    BSUB_POST_CMD=(
      bsub -P "${PROJECT}"
      -J "postproc_opt" -q "${QUEUE}" -n 4 -W 2:00
      -R "span[hosts=1] rusage[mem=8000]"
      -cwd "${BASE_DIR}"
      -oo "${LOGS_DIR}/postproc_opt.%J.out"
      -eo "${LOGS_DIR}/postproc_opt.%J.err"
    )
    # Only add dependency if there are pending jobs
    if [[ -n "${ARRAY_DEPENDENCY_EXPR}" ]]; then
      BSUB_POST_CMD+=(-w "${ARRAY_DEPENDENCY_EXPR}")
    fi

    POST_OUT="$("${BSUB_POST_CMD[@]}" "${POSTPROC_CMD}" 2>&1)"
    echo "${POST_OUT}"
    POST_JOB_ID="$(echo "${POST_OUT}" | parse_jobid)"
    [[ -n "${POST_JOB_ID}" ]] || die "Could not parse POST_JOB_ID from bsub output."
  fi

  log "Postprocess job id: ${POST_JOB_ID}"

  # ---------- 4) submit viz (dependent) ----------
  log "Step 4/5: Submit compare_models_faith.R"

  VIZ_CMD="cd ${BASE_DIR} && ${ACTIVATE_SNIPPET} Rscript ${R_VIZ} \
    --results_root ${RESULTS_DIR} \
    --outdir ${RESULTS_DIR}/compare_figs \
    --top_features ${TOP_FEATURES} \
    --dca_max_pt ${DCA_MAX_PT} \
    --dca_step ${DCA_STEP} \
    --calib_bins ${CALIB_BINS} \
    --spec_targets ${SPEC_TARGETS}"

  if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
    if [[ -n "${ARRAY_DEPENDENCY_EXPR}" ]]; then
      echo "bsub -P ${PROJECT} -w \"done(${POST_JOB_ID})\" -J viz_opt -q ${QUEUE} -n 2 -W 1:00 -R \"span[hosts=1] rusage[mem=4000]\" \"${VIZ_CMD}\""
    else
      echo "bsub -P ${PROJECT} -J viz_opt -q ${QUEUE} -n 2 -W 1:00 -R \"span[hosts=1] rusage[mem=4000]\" \"${VIZ_CMD}\""
    fi
    VIZ_JOB_ID="DRYRUN_VIZ"
  else
    # Build bsub command - viz ALWAYS depends on postproc completing
    BSUB_VIZ_CMD=(
      bsub -P "${PROJECT}"
      -J "viz_opt" -q "${QUEUE}" -n 2 -W 1:00
      -R "span[hosts=1] rusage[mem=4000]"
      -cwd "${BASE_DIR}"
      -oo "${LOGS_DIR}/viz_opt.%J.out"
      -eo "${LOGS_DIR}/viz_opt.%J.err"
      -w "done(${POST_JOB_ID})"
    )

    VIZ_OUT="$("${BSUB_VIZ_CMD[@]}" "${VIZ_CMD}" 2>&1)"
    echo "${VIZ_OUT}"
    VIZ_JOB_ID="$(echo "${VIZ_OUT}" | parse_jobid || true)"
  fi
fi

# ---------- 3b) submit single-model split aggregation ----------
SINGLE_POST_JOB_IDS=()
if [[ "${COMPARE_MODEL_SPLITS:-0}" -eq 1 ]]; then
  log "Step 3b/4: Submit single-model split aggregation"
  for MODEL in "${RUN_MODELS_ARR[@]}"; do
    MODEL_FULL="$(model_full_name "${MODEL}")"
    SINGLE_CMD="cd ${BASE_DIR} && ${ACTIVATE_SNIPPET} python -u ${PY_POSTPROC} \
      --mode single \
      --results_dir ${RESULTS_DIR} \
      --model ${MODEL_FULL} \
      --expected_splits ${N_SPLITS} \
      --single_outdir ${RESULTS_DIR}/${MODEL_FULL}_COMBINED"

    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
      if [[ -n "${ARRAY_DEPENDENCY_EXPR}" ]]; then
        echo "bsub -P ${PROJECT} -w \"${ARRAY_DEPENDENCY_EXPR}\" -J postproc_single_${MODEL} -q ${QUEUE} -n 2 -W 2:00 -R \"span[hosts=1] rusage[mem=8000]\" \"${SINGLE_CMD}\""
      else
        echo "bsub -P ${PROJECT} -J postproc_single_${MODEL} -q ${QUEUE} -n 2 -W 2:00 -R \"span[hosts=1] rusage[mem=8000]\" \"${SINGLE_CMD}\""
      fi
      SINGLE_POST_JOB_IDS+=("DRYRUN_SINGLE_${MODEL}")
      continue
    fi

    BSUB_SINGLE_CMD=(
      bsub -P "${PROJECT}"
      -J "postproc_single_${MODEL}"
      -q "${QUEUE}" -n 2 -W 2:00
      -R "span[hosts=1] rusage[mem=8000]"
      -cwd "${BASE_DIR}"
      -oo "${LOGS_DIR}/postproc_single_${MODEL}.%J.out"
      -eo "${LOGS_DIR}/postproc_single_${MODEL}.%J.err"
    )
    if [[ -n "${ARRAY_DEPENDENCY_EXPR}" ]]; then
      BSUB_SINGLE_CMD+=(-w "${ARRAY_DEPENDENCY_EXPR}")
    fi

    SINGLE_OUT="$("${BSUB_SINGLE_CMD[@]}" "${SINGLE_CMD}" 2>&1)"
    echo "${SINGLE_OUT}"
    SINGLE_JOB_ID="$(echo "${SINGLE_OUT}" | parse_jobid || true)"
    SINGLE_POST_JOB_IDS+=("${SINGLE_JOB_ID:-UNKNOWN}")
  done
fi

# ---------- 5) submit dashboard (dependent on all postprocessing) ----------
ALL_POST_JOB_IDS=()
[[ "${POST_JOB_ID:-}" != "SKIPPED" && -n "${POST_JOB_ID:-}" ]] && ALL_POST_JOB_IDS+=("${POST_JOB_ID}")
[[ "${VIZ_JOB_ID:-}" != "SKIPPED" && -n "${VIZ_JOB_ID:-}" ]] && ALL_POST_JOB_IDS+=("${VIZ_JOB_ID}")
for jid in "${SINGLE_POST_JOB_IDS[@]}"; do
  [[ "${jid}" != "UNKNOWN" ]] && ALL_POST_JOB_IDS+=("${jid}")
done

if [[ ${#ALL_POST_JOB_IDS[@]} -gt 0 || -n "${ARRAY_DEPENDENCY_EXPR}" ]]; then
  log "Step 5/5: Submit generate_dashboard.py"

  DASHBOARD_CMD="cd ${BASE_DIR} && ${ACTIVATE_SNIPPET} python -u ${PY_DASHBOARD} \
    --results_dir ${RESULTS_DIR} \
    --outfile ${RESULTS_DIR}/dashboard_progress.html"

  # Build dependency expression for dashboard
  DASH_DEP=""
  if [[ ${#ALL_POST_JOB_IDS[@]} -gt 0 ]]; then
    for jid in "${ALL_POST_JOB_IDS[@]}"; do
      [[ -z "$DASH_DEP" ]] && DASH_DEP="done(${jid})" || DASH_DEP="${DASH_DEP} && done(${jid})"
    done
  else
    # Fallback to model jobs if no postprocessing was submitted
    DASH_DEP="${ARRAY_DEPENDENCY_EXPR}"
  fi

  if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
    if [[ -n "${DASH_DEP}" ]]; then
      echo "bsub -P ${PROJECT} -w \"${DASH_DEP}\" -J dashboard_opt -q ${QUEUE} -n 2 -W 1:00 -R \"span[hosts=1] rusage[mem=4000]\" \"${DASHBOARD_CMD}\""
    else
      echo "bsub -P ${PROJECT} -J dashboard_opt -q ${QUEUE} -n 2 -W 1:00 -R \"span[hosts=1] rusage[mem=4000]\" \"${DASHBOARD_CMD}\""
    fi
    DASH_JOB_ID="DRYRUN_DASH"
  else
    BSUB_DASH_CMD=(
      bsub -P "${PROJECT}"
      -J "dashboard_opt" -q "${QUEUE}" -n 2 -W 1:00
      -R "span[hosts=1] rusage[mem=4000]"
      -cwd "${BASE_DIR}"
      -oo "${LOGS_DIR}/dashboard_opt.%J.out"
      -eo "${LOGS_DIR}/dashboard_opt.%J.err"
    )
    [[ -n "${DASH_DEP}" ]] && BSUB_DASH_CMD+=(-w "${DASH_DEP}")

    DASH_OUT="$("${BSUB_DASH_CMD[@]}" "${DASHBOARD_CMD}" 2>&1)"
    echo "${DASH_OUT}"
    DASH_JOB_ID="$(echo "${DASH_OUT}" | parse_jobid || true)"
  fi
fi

log "============================================"
log "OPTIMIZED PIPELINE SUBMITTED"
log "============================================"
log "Configuration:"
log "  Mode: ${MODE} (holdout=${MODE})"
log "  Splits: ${N_SPLITS} (repeated splits mode)"
log "  Test size: ${TEST_SIZE}"
log "  Scoring: neg_brier_score (calibration-focused)"
log "  Models: ${RUN_MODELS_LABEL}"
log ""
log "Model-specific resources:"
log "  RF:      ${RF_CPUS} CPUs, ${RF_MEM}MB/CPU, ${RF_WALLTIME} (high-memory for permutation importance)"
log "  XGBoost: ${XGB_CPUS} CPUs, ${XGB_MEM}MB/CPU, ${XGB_WALLTIME}, GPU=${XGB_USE_GPU:-1} (tree_method=${XGB_TREE_METHOD:-auto})"
log "  LinSVM:  ${SVM_CPUS} CPUs, ${SVM_MEM}MB/CPU, ${SVM_WALLTIME}"
log "  LR_EN:   ${LR_CPUS} CPUs, ${LR_MEM}MB/CPU, ${LR_WALLTIME}"
log ""
log "Submitted jobs: ${#ALL_JOB_IDS[@]} total"
log "  model jobs: ${ARRAY_DEPENDENCY_HUMAN}"
log "  postproc:   ${POST_JOB_ID}"
log "  viz:        ${VIZ_JOB_ID:-see bsub output}"
log "  dashboard:  ${DASH_JOB_ID:-see bsub output}"
if [[ "${COMPARE_MODEL_SPLITS:-0}" -eq 1 ]]; then
  log "  single postproc: $(IFS=','; echo "${SINGLE_POST_JOB_IDS[*]:-SKIPPED}")"
fi
log ""
log "Monitor:"
log "  bjobs -w | egrep \"${MODEL_GREP}|postproc_opt|viz_opt|dashboard_opt|postproc_single\""
log ""
log "Results will be in: ${RESULTS_DIR}"
log "============================================"

# ---------- POSTPROCESS_NOW mode ----------
# Run postprocessing immediately on completed jobs (in addition to queued jobs)
if [[ "${POSTPROCESS_NOW:-0}" -eq 1 ]]; then
  if [[ "${SKIP_MULTIPLE_POSTPROCESS:-0}" -eq 1 && "${COMPARE_MODEL_SPLITS:-0}" -ne 1 ]]; then
    log "POSTPROCESS_NOW=1: SKIP_MULTIPLE_POSTPROCESS=1 and COMPARE_MODEL_SPLITS=0 -> skipping"
  else
    log ""
    log "POSTPROCESS_NOW=1: Also running postprocessing immediately on completed jobs"
    run_postprocess_now "${RESULTS_DIR}" || true
  fi
fi

# ---------- INDIVIDUALPROCESS_NOW mode ----------
# Regenerate plots for individual completed model runs
if [[ "${INDIVIDUALPROCESS_NOW:-0}" -eq 1 ]]; then
  log ""
  log "INDIVIDUALPROCESS_NOW=1: Regenerating plots for all completed runs"
  run_individualprocess_now "${RESULTS_DIR}" || true
fi
