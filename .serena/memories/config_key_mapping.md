# Configuration Key Mapping

## Issue Identified
Two different config keys exist in the codebase:
- **`training_config.yaml`** (main/default): `feature_selection_strategy: hybrid_stability`
- **`training_config_production.yaml`** (production): `feature_select: hybrid`

## Resolution: Legacy Support
Both keys are **fully supported** thanks to a validator in `FeatureConfig` (schema.py):

```python
@model_validator(mode="after")
def validate_strategy(self) -> "FeatureConfig":
    """Validate feature selection strategy configuration."""
    # Legacy support: map old feature_select to new feature_selection_strategy
    if self.feature_select is not None:
        if self.feature_select == "hybrid":
            self.feature_selection_strategy = "hybrid_stability"
            logger.warning("feature_select='hybrid' is deprecated. Use feature_selection_strategy='hybrid_stability'")
```

## Mapping
| Old Key (deprecated) | New Key (current) | Mapped Value |
|---|---|---|
| `feature_select: hybrid` | `feature_selection_strategy` | `hybrid_stability` |
| `feature_select: none` | `feature_selection_strategy` | `none` |

## Recommendation
Update `training_config_production.yaml` to use the canonical key:
```yaml
features:
  feature_selection_strategy: hybrid_stability  # Changed from feature_select: hybrid
```

This ensures consistency with the main config and eliminates deprecation warnings.
