# CeD-ML Project Overview

## Purpose
ML pipeline to predict **incident Celiac Disease risk** from proteomics biomarkers measured before clinical diagnosis. Generates calibrated risk scores for screening decisions.

## Dataset
- 43,960 samples (43,662 controls, 148 incident CeD, 150 prevalent CeD)
- 2,920 proteins (`*_resid` columns)
- Demographics: age, BMI, sex, ethnicity (17% missing as category)
- Data file: `../data/Celiac_dataset_proteomics.csv` (~2.5 GB)

## Key Design Decisions
1. **IncidentPlusPrevalent scenario**: Prevalent in TRAIN only (50% sampling)
2. **Three-way split**: 50% TRAIN / 25% VAL / 25% TEST
3. **Brier score optimization**: Calibration over ranking
4. **Control downsampling**: 1:5 case:control ratio
5. **Missing as category**: Ethnicity missingness preserved

## Package Architecture
```
analysis/src/ced_ml/
  cli/        # Entry points: save-splits, train, postprocess, eval-holdout, config
  config/     # Pydantic schemas, YAML loading, validation
  data/       # I/O, splits, persistence, filters, schema
  features/   # screening, kbest, stability, corr_prune, panels
  models/     # registry, hyperparams, training, calibration, prevalence
  metrics/    # discrimination, thresholds, dca, bootstrap
  evaluation/ # predict, reports, holdout
  plotting/   # roc_pr, calibration, risk_dist, dca, learning_curve
  utils/      # logging, paths, random, serialization
```

## Models
- RF (Random Forest)
- XGBoost
- LinSVM_cal (Linear SVM with calibration)
- LR_EN (Logistic Regression with ElasticNet)

All use nested CV (5x10x5), RandomizedSearchCV (200 iter), Brier optimization.

## Statistics
- 15,109 lines of code
- 753 tests, 82% coverage
- Python 3.8+
