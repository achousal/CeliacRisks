"""LSF job submission utilities for HPC pipeline orchestration.

Provides functions to build and submit LSF (bsub) job scripts for running
the CeD-ML pipeline on HPC clusters with job dependency chains.
"""

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def detect_environment(base_dir: Path) -> dict[str, str]:
    """Detect active Python environment (venv or conda).

    Args:
        base_dir: Base directory to search for venv.

    Returns:
        Dict with keys: type, activation (bash command to activate).

    Raises:
        RuntimeError: If no Python environment is detected.
    """
    venv_activate = base_dir / "venv" / "bin" / "activate"
    if venv_activate.exists():
        return {
            "type": "venv",
            "activation": f'source "{venv_activate}"',
        }

    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env:
        return {
            "type": "conda",
            "activation": f"conda activate {conda_env}",
        }

    raise RuntimeError(
        "No Python environment detected. "
        "Expected venv at analysis/venv/ or active conda environment. "
        "Run: bash scripts/hpc_setup.sh"
    )


def load_hpc_config(config_path: Path) -> dict:
    """Load and validate HPC pipeline config.

    Args:
        config_path: Path to pipeline_hpc.yaml.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If required HPC fields are missing.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"HPC config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    hpc = config.get("hpc", {})
    required_fields = ["project", "queue", "walltime", "cores", "mem_per_core"]
    missing = [f for f in required_fields if f not in hpc]
    if missing:
        raise ValueError(f"Missing required HPC settings in {config_path}: {', '.join(missing)}")

    if hpc["project"] in ("YOUR_PROJECT_ALLOCATION", "YOUR_ALLOCATION"):
        raise ValueError(f"HPC project not configured. Update 'hpc.project' in {config_path}")

    return config


def build_job_script(
    *,
    job_name: str,
    command: str,
    project: str,
    queue: str,
    cores: int,
    mem_per_core: int,
    walltime: str,
    env_activation: str,
    log_dir: Path,
    dependency: str | None = None,
) -> str:
    """Build an LSF job script.

    Args:
        job_name: LSF job name (-J).
        command: Shell command(s) to execute.
        project: HPC project allocation (-P).
        queue: Queue name (-q).
        cores: Number of cores (-n).
        mem_per_core: Memory per core in MB.
        walltime: Wall time limit as "HH:MM" (-W).
        env_activation: Bash command to activate Python environment.
        log_dir: Directory for log files.
        dependency: Optional LSF dependency expression for -w flag.

    Returns:
        Complete bash script string for bsub submission.
    """
    log_err = log_dir / f"{job_name}.%J.err"
    live_log = log_dir / f"{job_name}.%J.live.log"

    dep_line = ""
    if dependency:
        dep_line = f'#BSUB -w "{dependency}"'

    script = f"""#!/bin/bash
#BSUB -P {project}
#BSUB -q {queue}
#BSUB -J {job_name}
#BSUB -n {cores}
#BSUB -W {walltime}
#BSUB -R "span[hosts=1] rusage[mem={mem_per_core}]"
#BSUB -oo /dev/null
#BSUB -eo {log_err}
{dep_line}

set -euo pipefail

export PYTHONUNBUFFERED=1
export FORCE_COLOR=1

{env_activation}

stdbuf -oL -eL {command} 2>&1 | tee -a "{live_log}"

exit ${{PIPESTATUS[0]}}
"""
    return script


def submit_job(script: str, dry_run: bool = False) -> str | None:
    """Submit an LSF job via bsub.

    Args:
        script: Job script content to submit via stdin.
        dry_run: If True, log the script but do not submit.

    Returns:
        Job ID string if submitted, None if dry_run or failure.

    Raises:
        RuntimeError: If bsub command is not available.
    """
    if dry_run:
        logger.info("[DRY RUN] Would submit job script:")
        for line in script.strip().split("\n"):
            if line.startswith("#BSUB") or line.startswith("stdbuf"):
                logger.info(f"  {line}")
        return None

    if not shutil.which("bsub"):
        raise RuntimeError("bsub command not found. LSF scheduler is required for --hpc mode.")

    result = subprocess.run(
        ["bsub"],
        input=script,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        logger.error(f"bsub failed: {result.stderr}")
        return None

    match = re.search(r"Job <(\d+)>", result.stdout)
    if match:
        return match.group(1)

    logger.error(f"Could not parse job ID from bsub output: {result.stdout}")
    return None


def _build_training_command(
    *,
    config_file: Path,
    infile: Path,
    split_dir: Path,
    outdir: Path,
    models: list[str],
    split_seed: int,
    run_id: str,
    enable_ensemble: bool,
    enable_consensus: bool,
    enable_optimize_panel: bool,
) -> str:
    """Build ced run-pipeline command for a single split seed (training only).

    Each per-seed job trains all models for that seed but skips aggregation
    and post-processing (handled by the dependency job).
    """
    models_str = ",".join(models)
    parts = [
        "ced run-pipeline",
        f'--config "{config_file}"',
        f'--infile "{infile}"',
        f'--split-dir "{split_dir}"',
        f'--outdir "{outdir}"',
        f"--models {models_str}",
        f"--split-seeds {split_seed}",
        f"--run-id {run_id}",
        # Disable post-processing in per-seed jobs -- the post job handles it
        "--no-ensemble",
        "--no-consensus",
        "--no-optimize-panel",
        "-v",
    ]
    return " \\\n  ".join(parts)


def _build_postprocessing_command(
    *,
    config_file: Path,
    run_id: str,
    outdir: Path,
    infile: Path,
    split_dir: Path,
    models: list[str],
    split_seeds: list[int],
    enable_ensemble: bool,
    enable_consensus: bool,
    enable_optimize_panel: bool,
) -> str:
    """Build post-processing commands (aggregation, ensemble, panel optimization).

    Returns a multi-line bash script fragment that runs each step sequentially.
    """
    lines = [
        "set -euo pipefail",
        "",
        f'echo "Post-processing for run {run_id}"',
        "",
    ]

    # Aggregate base models
    for model in models:
        lines.append(f'echo "Aggregating {model}..."')
        lines.append(f"ced aggregate-splits --run-id {run_id} --model {model}")
        lines.append("")

    # Train ensemble per seed
    if enable_ensemble:
        for seed in split_seeds:
            lines.append(f'echo "Training ensemble seed {seed}..."')
            lines.append(f"ced train-ensemble --run-id {run_id} --split-seed {seed}")
        lines.append("")
        lines.append('echo "Aggregating ENSEMBLE..."')
        lines.append(f"ced aggregate-splits --run-id {run_id} --model ENSEMBLE")
        lines.append("")

    # Optimize panel
    if enable_optimize_panel:
        lines.append('echo "Optimizing panels..."')
        lines.append(f"ced optimize-panel --run-id {run_id}")
        lines.append("")

    # Consensus panel
    if enable_consensus:
        lines.append('echo "Generating consensus panel..."')
        lines.append(f"ced consensus-panel --run-id {run_id}")
        lines.append("")

    lines.append(f'echo "Post-processing complete for run {run_id}"')

    return "\n".join(lines)


def submit_hpc_pipeline(
    *,
    config_file: Path,
    infile: Path,
    split_dir: Path,
    outdir: Path,
    models: list[str],
    split_seeds: list[int],
    run_id: str,
    enable_ensemble: bool,
    enable_consensus: bool,
    enable_optimize_panel: bool,
    hpc_config: dict,
    logs_dir: Path,
    dry_run: bool,
    pipeline_logger: logging.Logger,
) -> dict:
    """Submit complete HPC pipeline with dependency chains.

    Submits:
    1. N training jobs (one per split seed, all models)
    2. 1 post-processing job (depends on all training jobs)

    Args:
        config_file: Path to training config YAML.
        infile: Path to input data file.
        split_dir: Path to split indices directory.
        outdir: Path to results output directory.
        models: List of model names.
        split_seeds: List of split seeds.
        run_id: Shared run identifier.
        enable_ensemble: Enable ensemble training in post-processing.
        enable_consensus: Enable consensus panel in post-processing.
        enable_optimize_panel: Enable panel optimization in post-processing.
        hpc_config: Parsed pipeline_hpc.yaml config dict.
        logs_dir: Directory for job logs.
        dry_run: Preview without submitting.
        pipeline_logger: Logger instance.

    Returns:
        Dict with run_id, training_jobs, postprocessing_job, logs_dir.
    """
    hpc = hpc_config["hpc"]
    base_dir = Path.cwd()

    # Detect environment
    env_info = detect_environment(base_dir)
    pipeline_logger.info(f"Python environment: {env_info['type']}")

    # Create log directory
    run_logs_dir = logs_dir / "training" / f"run_{run_id}"
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    # Common bsub parameters
    bsub_params = {
        "project": hpc["project"],
        "queue": hpc["queue"],
        "cores": hpc["cores"],
        "mem_per_core": hpc["mem_per_core"],
        "walltime": hpc["walltime"],
        "env_activation": env_info["activation"],
        "log_dir": run_logs_dir,
    }

    # Submit training jobs (one per seed)
    pipeline_logger.info(f"Submitting {len(split_seeds)} training jobs...")
    training_job_ids = []

    for seed in split_seeds:
        job_name = f"CeD_{run_id}_seed{seed}"

        command = _build_training_command(
            config_file=config_file.resolve(),
            infile=infile.resolve(),
            split_dir=split_dir.resolve(),
            outdir=outdir.resolve(),
            models=models,
            split_seed=seed,
            run_id=run_id,
            enable_ensemble=enable_ensemble,
            enable_consensus=enable_consensus,
            enable_optimize_panel=enable_optimize_panel,
        )

        script = build_job_script(
            job_name=job_name,
            command=command,
            **bsub_params,
        )

        job_id = submit_job(script, dry_run=dry_run)
        if job_id:
            pipeline_logger.info(f"  Seed {seed}: Job {job_id}")
            training_job_ids.append(job_id)
        elif dry_run:
            pipeline_logger.info(f"  [DRY RUN] Seed {seed}: {job_name}")
            training_job_ids.append(f"DRYRUN_{seed}")
        else:
            pipeline_logger.error(f"  Seed {seed}: Submission failed")

    # Submit post-processing job with dependency
    post_job_name = f"CeD_{run_id}_post"
    dependency_expr = f"done(CeD_{run_id}_seed*)"

    post_command = _build_postprocessing_command(
        config_file=config_file.resolve(),
        run_id=run_id,
        outdir=outdir.resolve(),
        infile=infile.resolve(),
        split_dir=split_dir.resolve(),
        models=models,
        split_seeds=split_seeds,
        enable_ensemble=enable_ensemble,
        enable_consensus=enable_consensus,
        enable_optimize_panel=enable_optimize_panel,
    )

    post_script = build_job_script(
        job_name=post_job_name,
        command=post_command,
        dependency=dependency_expr,
        **bsub_params,
    )

    post_job_id = submit_job(post_script, dry_run=dry_run)
    if post_job_id:
        pipeline_logger.info(f"  Post-processing: Job {post_job_id} (depends on training)")
    elif dry_run:
        pipeline_logger.info(f"  [DRY RUN] Post-processing: {post_job_name}")

    return {
        "run_id": run_id,
        "training_jobs": training_job_ids,
        "postprocessing_job": post_job_id or f"DRYRUN_{post_job_name}",
        "logs_dir": run_logs_dir,
    }
