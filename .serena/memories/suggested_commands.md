# Suggested Commands

## Installation
```bash
cd analysis
pip install -e .          # Editable install
pip install -e ".[dev]"   # With dev dependencies
```

## CLI Commands
```bash
ced --help                          # Show all commands
ced save-splits --config configs/splits_config.yaml --infile <data.csv>
ced train --config configs/training_config.yaml --model LR_EN
ced postprocess --results-dir results/ --n-boot 500
ced eval-holdout --config configs/holdout_config.yaml --model-artifact <model.joblib>
ced config validate config.yaml --strict
ced config migrate --command train --args "--folds 10" -o config.yaml
ced config diff config1.yaml config2.yaml
```

## Testing
```bash
pytest tests/ -v                              # All tests
pytest tests/ -v --cov=src/ced_ml             # With coverage
pytest tests/test_models_*.py -v              # Specific module
pytest tests/ -v -m "not slow"                # Skip slow tests
pytest tests/test_e2e_pipeline.py -v          # End-to-end
```

## Formatting and Linting
```bash
black analysis/src analysis/tests             # Format code
ruff check analysis/src analysis/tests        # Lint
ruff check analysis/src --fix                 # Auto-fix lint issues
mypy analysis/src/ced_ml/                     # Type checking
```

## Pre-commit
```bash
pre-commit install                            # Install hooks
pre-commit run --all-files                    # Run all hooks
```

## Git (Darwin/macOS)
```bash
git status
git diff
git add -p                                    # Stage interactively
git commit -m "type(scope): message"
git log --oneline -10
```

## System (Darwin)
```bash
ls -la
find . -name "*.py" -type f
grep -r "pattern" analysis/src/
wc -l analysis/src/ced_ml/**/*.py
