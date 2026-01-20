# Code Style and Conventions

## Non-Negotiables
- No emojis anywhere (code, comments, docs)
- No secrets/credentials in code
- Test before commit
- No debug artifacts (print() in production, console.log, browser())

## Python Style
- **Formatter**: Black (line-length: 100)
- **Linter**: Ruff (E, W, F, I, B, C4, UP rules)
- **Type checker**: mypy (check_untyped_defs=true)
- **Python version**: 3.8+

### Conventions
- f-strings for formatting
- pathlib.Path for file paths
- logging module (not print)
- Type hints for public functions
- Docstrings: Google style

### Example
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
    """
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)
```

## R Style
- Explicit namespacing: `dplyr::mutate`
- No `setwd()` in scripts
- Use `message()/warning()/stop()` for output
- Call `set.seed()` for RNG
- Formatter: styler (tidyverse_style)
- Linter: lintr

## Commit Messages
```
type(scope): summary

Types: feat, fix, refactor, test, docs, chore, perf
```

## Pre-commit Hooks (enforced)
- black, ruff, mypy
- detect-secrets
- no print() in src/ced_ml (except cli, logging)
- no emojis
- no .env files
