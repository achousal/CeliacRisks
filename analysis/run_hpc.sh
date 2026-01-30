#!/usr/bin/env bash
#============================================================
# DEPRECATED: run_hpc.sh has been replaced by:
#
#   ced run-pipeline --hpc
#
# Examples:
#   ced run-pipeline --hpc                          # Submit with defaults
#   ced run-pipeline --hpc --dry-run                # Preview without submitting
#   ced run-pipeline --hpc --models LR_EN,RF,XGBoost --split-seeds 0,1,2
#   ced run-pipeline --hpc --hpc-config configs/pipeline_custom.yaml
#
# The --hpc flag reads configs/pipeline_hpc.yaml, submits per-seed
# training jobs, and a post-processing job with LSF dependency chains.
#============================================================

echo "DEPRECATED: run_hpc.sh has been replaced by 'ced run-pipeline --hpc'"
echo ""
echo "Usage:"
echo "  ced run-pipeline --hpc                    # Submit with defaults from pipeline_hpc.yaml"
echo "  ced run-pipeline --hpc --dry-run           # Preview without submitting"
echo "  ced run-pipeline --hpc --models LR_EN,RF   # Custom models"
echo ""
echo "See: ced run-pipeline --help"
exit 1
