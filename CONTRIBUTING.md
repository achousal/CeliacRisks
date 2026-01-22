# Contributing to CeliacRiskML

## Non-Negotiables

1. No secrets in code (API keys, passwords, credentials)
2. No emojis in code, comments, or docs
3. Tests pass before commit
4. No debug artifacts (console.log, print(), browser())
5. Reproducibility: fixed seeds, documented parameters

## Quick Start

```bash
# Setup
cd analysis/
pip install -e ".[dev]"

# Test
pytest tests/ -v

# Format/lint
black src/ tests/
ruff check src/ tests/
```

## Workflow

1. Branch from `main`:
   ```bash
   git checkout -b feature/your-feature
   ```

2. Commit with conventional format:
   ```
   type(scope): summary
   ```
   Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`

3. Push and create PR

## Testing Requirements

- New features require tests
- Bug fixes require regression tests
- Target: >=80% coverage for core modules
- All tests must pass

```bash
pytest tests/ -v                     # All tests
pytest tests/ --cov=src/ced_ml       # With coverage
pytest tests/ -m "not slow"          # Skip slow tests
```

## Code Style

**Python**:
- Black (line length 88), Ruff linting
- Type hints for public functions
- F-strings, pathlib, logging (not print)

**R**:
- Explicit namespacing (dplyr::mutate)
- No setwd(), use message()/warning()/stop()
- Call set.seed() when using RNG

## Scientific Validity

If changes affect results:
1. Document scientific rationale
2. Validate before/after metrics
3. Add regression tests
4. Update [CLAUDE.md](.claude/CLAUDE.md)

## PR Checklist

- [ ] Tests pass locally
- [ ] Code formatted (black, ruff)
- [ ] No secrets committed
- [ ] [CLAUDE.md](.claude/CLAUDE.md) updated (if user-facing changes)

## HPC Code

- No interactive prompts
- Deterministic behavior
- Portable across schedulers
- Clear resource requirements

## Help

- Open an issue for bugs/questions
- Tag @achousal for urgent issues
