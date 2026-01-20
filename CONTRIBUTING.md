# Contributing to CeliacRiskML

Thank you for your interest in contributing to CeliacRiskML. This document provides guidelines for contributing to the project.

## Code of Conduct

This project follows standard scientific collaboration principles:
- Prioritize correctness and scientific validity
- Ensure reproducibility
- Maintain clear documentation
- Respect intellectual property and data privacy

## Non-Negotiables

Before contributing, ensure you understand these mandatory requirements:

1. **No secrets in code**: Never commit API keys, passwords, credentials, or sensitive data
2. **No emojis**: Code, comments, strings, and documentation must not contain emojis
3. **Test before commit**: All tests must pass before submitting PR
4. **No debug artifacts**: Remove console.log (JS), print() (Python), browser() (R) before commit
5. **Reproducibility**: All analyses must be reproducible with fixed seeds

## Development Setup

### Local Environment

```bash
# Clone repository
git clone git@github.com:achousal/CeliacRiskML.git
cd CeliacRiskML/analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

### Pre-commit Hooks

Install pre-commit hooks to catch issues early:

```bash
pip install pre-commit
pre-commit install
```

## Branching Strategy

- `main`: Stable, production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `fix/*`: Bug fixes
- `refactor/*`: Code improvements without behavior changes

### Workflow

1. Create branch from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. Make changes and commit:
   ```bash
   git add .
   git commit -m "feat(module): brief description"
   ```

3. Push and create PR:
   ```bash
   git push origin feature/your-feature-name
   ```

## Commit Message Convention

Follow conventional commits format:

```
type(scope): summary

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring (no behavior change)
- `test`: Adding or updating tests
- `docs`: Documentation changes
- `chore`: Maintenance tasks
- `perf`: Performance improvements

**Examples:**
```
feat(models): add support for XGBoost GPU training
fix(metrics): correct Brier score calculation for imbalanced data
refactor(features): extract correlation pruning into separate module
test(data): add tests for split generation edge cases
docs(readme): update installation instructions for HPC
```

## Testing Requirements

### Minimum Requirements
- All new features must include tests
- Bug fixes must include regression tests
- Test coverage must remain >= 80% for core modules
- All tests must pass before PR approval

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_models_training.py -v

# With coverage
pytest tests/ -v --cov=src/ced_ml --cov-report=term

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Writing Tests

```python
# tests/test_example.py
import pytest
from ced_ml.models import train_model

def test_train_model_basic():
    """Test basic model training workflow."""
    # Arrange
    X, y = create_toy_data()

    # Act
    model = train_model(X, y, model_type='LR_EN')

    # Assert
    assert model is not None
    assert hasattr(model, 'predict_proba')
```

## Code Style

### Python

- Follow PEP 8
- Use Black for formatting (line length: 88)
- Use Ruff for linting
- Type hints for public functions
- F-strings for formatting
- Pathlib for file paths
- Logging instead of print()

**Example:**
```python
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_data(file_path: Path) -> pd.DataFrame:
    """Load dataset from CSV file.

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame with loaded data

    Raises:
        FileNotFoundError: If file does not exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)
```

### R

- Use explicit namespacing (dplyr::mutate)
- Avoid setwd() in scripts
- Use message()/warning()/stop() for output
- Call set.seed() when using RNG
- Use styler for formatting

## Scientific Validity

### When Changes Affect Results

If your change modifies model outputs, metrics, or data processing:

1. **Document the change**: Explain scientific rationale
2. **Validate results**: Compare before/after metrics
3. **Update tests**: Add regression tests
4. **Update documentation**: Note impact in CLAUDE.md

### Reproducibility Checklist

- [ ] Random seeds are fixed and documented
- [ ] Configuration files updated
- [ ] Results validated against baseline
- [ ] Run metadata recorded (parameters, versions)
- [ ] No manual steps in pipeline

## Pull Request Process

### Before Submitting

1. **All tests pass locally**
   ```bash
   pytest tests/ -v
   ```

2. **Code is formatted**
   ```bash
   black src/ tests/
   ruff check src/ tests/
   ```

3. **No secrets committed**
   ```bash
   git diff | grep -i "password\|secret\|key\|token"  # Should return nothing
   ```

4. **CLAUDE.md updated** (if user-facing changes)

### PR Template

The PR template will guide you through required information:
- Description of changes
- Type of change
- Scientific impact
- Testing completed
- Reproducibility checks

### Review Process

1. Automated checks must pass (tests, lint, security)
2. Code review by maintainer
3. Scientific validation (if results changed)
4. Approval and merge

## Documentation

### When to Update Docs

- New features: Add to README and CLAUDE.md
- API changes: Update docstrings and examples
- Configuration changes: Update config documentation
- Workflow changes: Update CONTRIBUTING.md

### Documentation Style

- Clear, concise language
- Code examples for usage
- Links to relevant papers/references
- No emojis

## HPC Considerations

When contributing HPC-related code:

- No interactive prompts
- Deterministic behavior
- Portable across schedulers (LSF, Slurm, PBS)
- Clear resource requirements documented
- Resumable jobs when feasible

## Getting Help

- Open an issue for bugs or questions
- Use GitHub Discussions for general questions
- Tag @achousal for urgent issues

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).
