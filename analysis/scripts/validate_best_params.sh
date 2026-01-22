#!/usr/bin/env bash
#============================================================
# validate_best_params.sh
#
# Diagnostic script to validate best_params_per_split.csv
# files across all split seeds and models
#
# Usage:
#   bash scripts/validate_best_params.sh [RESULTS_DIR]
#============================================================

set -euo pipefail

RESULTS_DIR="${1:-../results}"

echo "============================================"
echo "Validating best_params_per_split.csv files"
echo "============================================"
echo "Results dir: ${RESULTS_DIR}"
echo ""

# Find all best_params files
PARAM_FILES=$(find "${RESULTS_DIR}" -name "best_params_per_split.csv" 2>/dev/null | sort)

if [[ -z "${PARAM_FILES}" ]]; then
  echo "ERROR: No best_params_per_split.csv files found in ${RESULTS_DIR}"
  exit 1
fi

echo "Found $(echo "${PARAM_FILES}" | wc -l) best_params file(s)"
echo ""

# Check each file
TOTAL_FILES=0
TOTAL_ROWS=0
ISSUES=0

while IFS= read -r FILE; do
  TOTAL_FILES=$((TOTAL_FILES + 1))

  # Extract split_seed and check path structure
  if [[ "${FILE}" =~ split_seed([0-9]+) ]]; then
    SEED="${BASH_REMATCH[1]}"
  else
    echo "[WARN] ${FILE}: Cannot extract split_seed from path"
    SEED="unknown"
  fi

  # Count rows (excluding header)
  ROW_COUNT=$(tail -n +2 "${FILE}" | wc -l | tr -d ' ')
  TOTAL_ROWS=$((TOTAL_ROWS + ROW_COUNT))

  # Check for duplicate model names (should all be same model)
  MODEL_COUNT=$(tail -n +2 "${FILE}" | cut -d',' -f1 | sort -u | wc -l | tr -d ' ')

  # Get unique models
  MODELS=$(tail -n +2 "${FILE}" | cut -d',' -f1 | sort -u | tr '\n' ',' | sed 's/,$//')

  # Check for expected columns
  HEADER=$(head -n1 "${FILE}")
  HAS_REPEAT=$(echo "${HEADER}" | grep -c "repeat" || true)
  HAS_OUTER_SPLIT=$(echo "${HEADER}" | grep -c "outer_split" || true)
  HAS_BEST_PARAMS=$(echo "${HEADER}" | grep -c "best_params" || true)

  # Status
  STATUS="OK"
  if [[ ${ROW_COUNT} -eq 0 ]]; then
    STATUS="EMPTY"
    ISSUES=$((ISSUES + 1))
  elif [[ ${ROW_COUNT} -eq 1 ]]; then
    STATUS="ONLY_1_ROW"
    ISSUES=$((ISSUES + 1))
  elif [[ ${MODEL_COUNT} -ne 1 ]]; then
    STATUS="MULTI_MODEL"
    ISSUES=$((ISSUES + 1))
  fi

  echo "File: ${FILE}"
  echo "  Split seed: ${SEED}"
  echo "  Rows (excl header): ${ROW_COUNT}"
  echo "  Unique models: ${MODEL_COUNT} (${MODELS})"
  echo "  Status: ${STATUS}"

  if [[ ${HAS_REPEAT} -eq 0 || ${HAS_OUTER_SPLIT} -eq 0 || ${HAS_BEST_PARAMS} -eq 0 ]]; then
    echo "  [WARN] Missing expected columns (repeat, outer_split, best_params)"
    ISSUES=$((ISSUES + 1))
  fi

  # Show first few rows for debugging
  if [[ ${ROW_COUNT} -gt 0 && ${ROW_COUNT} -le 3 ]]; then
    echo "  First ${ROW_COUNT} row(s):"
    tail -n +2 "${FILE}" | head -n 3 | sed 's/^/    /'
  fi

  echo ""
done <<< "${PARAM_FILES}"

echo "============================================"
echo "Summary"
echo "============================================"
echo "Total files: ${TOTAL_FILES}"
echo "Total rows (all files): ${TOTAL_ROWS}"
echo "Issues found: ${ISSUES}"
echo ""

if [[ ${ISSUES} -gt 0 ]]; then
  echo "VALIDATION FAILED: ${ISSUES} issue(s) detected"
  echo ""
  echo "Common issues:"
  echo "  - EMPTY: No data rows (file exists but empty)"
  echo "  - ONLY_1_ROW: Only 1 outer fold instead of expected N (e.g., 15 for 5 folds x 3 repeats)"
  echo "  - MULTI_MODEL: Multiple models in same file (should be 1 model per file)"
  echo ""
  echo "Expected structure:"
  echo "  results/"
  echo "    ├── split_seed0/cv/best_params_per_split.csv  (15 rows: 1 per outer fold)"
  echo "    ├── split_seed1/cv/best_params_per_split.csv  (15 rows)"
  echo "    └── ..."
  exit 1
else
  echo "VALIDATION PASSED: All files OK"
  exit 0
fi
