# Documentation Index

This folder contains production documentation for the CeliacRiskML pipeline. Historical development notes are archived in `./archive/`.

## Essential References (Start here)

- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Migrate from legacy scripts to the `ced` CLI package
- **[HPC_MIGRATION_GUIDE.md](HPC_MIGRATION_GUIDE.md)** - Deploy on LSF/HPC systems
- **[KNOBS_CHEATSHEET.txt](KNOBS_CHEATSHEET.txt)** - Quick reference for configuration parameters
- **[PARAMETERS_REFERENCE.txt](PARAMETERS_REFERENCE.txt)** - Detailed parameter documentation

## Configuration Examples

- **[examples/training_config.yaml](examples/training_config.yaml)** - Typical training configuration
- **[examples/splits_config.yaml](examples/splits_config.yaml)** - Splits generation configuration
- **[examples/phase_c_demo.md](examples/phase_c_demo.md)** - Example workflow

## Reference Data

- **[table1_cohort_description.csv](table1_cohort_description.csv)** - Dataset characteristics and cohort summary

## Archive

For historical context on development phases, feature implementation, and audits, see **[./archive/](./archive/)**. Files here document the refactoring process from legacy to package architecture and are not needed for routine use.
