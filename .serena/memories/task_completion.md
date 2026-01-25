# Task Completion Checklist

## Before Marking Task Complete

### 1. Tests Pass
```bash
cd analysis
pytest tests/ -v
```
All 753+ tests must pass.

### 2. Code Formatted
```bash
black analysis/src analysis/tests
ruff check analysis/src analysis/tests --fix
```

### 3. Type Checks (if types modified)
```bash
mypy analysis/src/ced_ml/
```

### 4. No Secrets
```bash
git diff | grep -iE "(password|secret|key|token|credential)"
# Should return empty
```

### 5. No Debug Artifacts
- No `print()` in production code (use logger)
- No `console.log` (if JS exists)
- No `browser()` or `debug()` in R

### 6. Documentation Updated (if user-facing changes)
- Update `README.md` for workflow/API/CLI changes
- Update docstrings for function signature changes
- Update `analysis/SETUP_README.md` for setup/configuration changes

### 7. Pre-commit Passes
```bash
pre-commit run --all-files
```

## Coverage Target
- Maintain >= 82% coverage
- New features require tests
- Bug fixes require regression tests

## Scientific Changes
If change affects results/metrics:
- Document rationale
- Compare before/after
- Add regression test
- Note in README.md and relevant ADR
