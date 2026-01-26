# Investigations Folder

This folder contains investigation scripts and documentation for exploring specific aspects of the CeliacRisks ML pipeline.

## Contents

### Investigation Scripts
- **`run_investigation_simple.py`**: Simple investigation runner for basic analyses
- **`run_investigation_oof.py`**: Out-of-fold investigation runner for calibration analyses
- **`run_investigation.sh`**: Shell script to orchestrate investigation runs
- **`investigate_case_type_calibration.py`**: Analyze calibration differences between prevalent and incident cases
- **`investigate_case_type_scores.py`**: Compare risk score distributions between case types
- **`investigate_feature_bias.py`**: Explore potential feature-level biases between case types

### Configuration
- **`splits_config_investigation.yaml`**: Split configuration for investigation runs

### Documentation
- **`INVESTIGATION_SETUP.md`**: Setup instructions for running investigations
- **`INVESTIGATION_SUMMARY.md`**: Summary of investigation findings
- **`IMPLEMENTATION_SUMMARY.md`**: Technical implementation details
- **`prevalent_vs_incident_scores.md`**: Detailed analysis of prevalent vs incident case scores

## Purpose

These investigations explore the interaction between prevalent and incident CeD cases in the training pipeline, specifically:
- Calibration differences between case types
- Risk score distribution patterns
- Feature-level signals that might distinguish case types
- Impact on model generalization

## Usage

See [INVESTIGATION_SETUP.md](INVESTIGATION_SETUP.md) for detailed instructions on running investigations.

Quick start:
```bash
cd analysis/docs/investigations/
bash run_investigation.sh
```

## Note

These are exploratory analyses separate from the main production pipeline. Results inform but do not directly modify the core pipeline configuration.
