"""
CAA Experiment Manager

Reads a declarative experiment YAML config and submits SLURM array jobs
for each model, auto-detecting layer ranges from vector files.

Usage:
    # Dry run (print commands without submitting)
    python scripts/run_experiment.py configs/experiment/caa_tulu8b.yaml --dry-run

    # Submit all models
    python scripts/run_experiment.py configs/experiment/caa_tulu8b.yaml

    # Submit only a specific model
    python scripts/run_experiment.py configs/experiment/caa_tulu8b.yaml --model allenai_llama_3_1_tulu_3_1_8B

Environment:
    Requires env.sh to be sourced first (sets PROJECT_ROOT, PYTHON_BIN, etc.)
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import yaml
from pydantic import BaseModel, field_validator


# config schema

class SlurmConfig(BaseModel):
    partition: str = "gpu"
    qos: str = "gpu"
    gres: str = "gpu:1"
    cpus_per_task: int = 8
    mem: str = "82G"
    time: str = "04:00:00"


class CAAConfig(BaseModel):
    coeff: float = 1.0
    layers: Optional[str] = None

class ExperimentConfig(BaseModel):
    name: str
    models: Dict[str, str]
    datasets: str = "glob(*,exclude=dummy)"
    judge: str = "contains_abstention_keyword"
    single_job: bool = True
    mode: str = "local"
    judge_label: str = "Keywords_judge"
    tag: str = ""
    caa: CAAConfig = CAAConfig()
    slurm: SlurmConfig = SlurmConfig()

    @field_validator("models", mode="before")
    @classmethod
    def validate_models(cls, v):
        if not isinstance(v, dict):
            raise ValueError(
                "models must be a mapping of model_id: vector_directory_name, e.g.\n"
                "  allenai_llama_3_1_tulu_3_1_8B: Llama3_1_Tulu_3_1_8B"
            )
        return v


# env helpers

def get_env_var(name: str, dry_run: bool = False) -> str:
    """Get a required environment variable (from env.sh).
    In dry-run mode, returns a placeholder instead of crashing."""
    value = os.environ.get(name)
    if not value:
        if dry_run:
            placeholder = f"<{name}>"
            print(f"  Warning: ${name} not set, using placeholder '{placeholder}'")
            return placeholder
        print(f"Error: ${name} is not set. Source env.sh first:", file=sys.stderr)
        print(f"  source env.sh", file=sys.stderr)
        sys.exit(1)
    return value


# layer auto-detection

def detect_layers(vector_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    """
    Scan vector_dir for vec_layer_N.pt files and return (min_layer, max_layer).
    In dry-run mode, returns a placeholder range if the directory doesn't exist.
    """
    if not vector_dir.is_dir():
        if dry_run:
            print(f"  Warning: Vector dir not found ({vector_dir}), using placeholder range 0-31")
            return 0, 31
        print(f"Error: Vector directory not found: {vector_dir}", file=sys.stderr)
        sys.exit(1)

    pattern = re.compile(r"vec_layer_(\d+)\.pt$")
    layers = []
    for f in vector_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            layers.append(int(m.group(1)))

    if not layers:
        if dry_run:
            print(f"  Warning: No vec_layer_*.pt files in {vector_dir}, using placeholder range 0-31")
            return 0, 31
        print(f"Error: No vec_layer_*.pt files in {vector_dir}", file=sys.stderr)
        sys.exit(1)

    layers.sort()
    return layers[0], layers[-1]


def parse_layer_range(layer_spec: str) -> tuple[int, int]:
    """Parse explicit layer spec like '15-31'."""
    parts = layer_spec.split("-")
    if len(parts) != 2:
        print(f"Error: Invalid layer range '{layer_spec}'. Expected 'MIN-MAX'.", file=sys.stderr)
        sys.exit(1)
    return int(parts[0]), int(parts[1])


# SLURM submission

def build_sbatch_command(
    config: ExperimentConfig,
    model_id: str,
    vector_dir_name: str,
    min_layer: int,
    max_layer: int,
    project_root: str,
    python_bin: str,
    user_email: str,
) -> list[str]:
    """Build the sbatch command for a single model."""

    vector_dir = f"{project_root}/data/vectors/{vector_dir_name}"

    # common_dir base path
    coeff_str = str(config.caa.coeff).replace(".", "_")
    tag_suffix = f"_{config.tag}" if config.tag else ""
    common_dir_name = f"{vector_dir_name}_{config.judge_label}_CAA_coeff_{coeff_str}{tag_suffix}"
    common_dir_base = f"{project_root}/data/{common_dir_name}"

    job_name = f"caa_{model_id}"
    job_template = "scripts/job_template.sh"

    # env vars to export to the job
    export_vars = [
        "ALL",
        f"EXP_MODEL_ID={model_id}",
        f"EXP_VECTOR_DIR={vector_dir}",
        f"EXP_COMMON_DIR_BASE={common_dir_base}",
        f"EXP_DATASETS={config.datasets}",
        f"EXP_JUDGE={config.judge}",
        f"EXP_SINGLE_JOB={config.single_job}",
        f"EXP_MODE={config.mode}",
        f"EXP_COEFF={config.caa.coeff}",
        f"EXP_PYTHON_BIN={python_bin}",
        f"EXP_USER_EMAIL={user_email}",
    ]

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--array={min_layer}-{max_layer}",
        f"--partition={config.slurm.partition}",
        f"--qos={config.slurm.qos}",
        f"--gres={config.slurm.gres}",
        f"--cpus-per-task={config.slurm.cpus_per_task}",
        f"--mem={config.slurm.mem}",
        f"--time={config.slurm.time}",
        f"--export={','.join(export_vars)}",
        job_template,
    ]

    return cmd


# main

def load_config(path: str) -> ExperimentConfig:
    """Load and validate an experiment YAML config."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    try:
        return ExperimentConfig(**raw)
    except Exception as e:
        print(f"Error loading config {path}:\n{e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="CAA Experiment Orchestrator — submit SLURM jobs from YAML configs"
    )
    parser.add_argument(
        "config",
        help="Path to experiment YAML config (e.g. configs/experiment/caa_tulu8b.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run only this model (must be a key in the experiment's models map)",
    )
    args = parser.parse_args()

    # load env vars
    project_root = get_env_var("PROJECT_ROOT", dry_run=args.dry_run)
    python_bin = get_env_var("PYTHON_BIN", dry_run=args.dry_run)
    user_email = os.environ.get("USER_EMAIL", "")

    # load env config
    config = load_config(args.config)
    print(f"Experiment: {config.name}")
    print(f"Project root: {project_root}")
    print()

    # filter models if --model is specified
    models = config.models
    if args.model:
        if args.model not in models:
            available = ", ".join(models.keys())
            print(
                f"Error: Model '{args.model}' not in config. Available: {available}",
                file=sys.stderr,
            )
            sys.exit(1)
        models = {args.model: models[args.model]}

    for model_id, vector_dir_name in models.items():
        vector_dir = Path(project_root) / "data" / "vectors" / vector_dir_name

        # resolve layer range
        if config.caa.layers:
            min_layer, max_layer = parse_layer_range(config.caa.layers)
            print(f"Model: {model_id}")
            print(f"  Layers: {min_layer}-{max_layer} (from config)")
        else:
            print(f"Model: {model_id}")
            print(f"  Scanning {vector_dir} for vectors...")
            min_layer, max_layer = detect_layers(vector_dir, dry_run=args.dry_run)
            print(f"  Layers: {min_layer}-{max_layer} (auto-detected)")

        # sbatch
        cmd = build_sbatch_command(
            config=config,
            model_id=model_id,
            vector_dir_name=vector_dir_name,
            min_layer=min_layer,
            max_layer=max_layer,
            project_root=project_root,
            python_bin=python_bin,
            user_email=user_email,
        )

        if args.dry_run:
            print(f"\n  [DRY RUN] Would execute:")
            print(f"    {' \\\n      '.join(cmd)}")
            print()
        else:
            print(f"  Submitting...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✓ {result.stdout.strip()}")
            else:
                print(f"  ✗ sbatch failed (exit {result.returncode}):", file=sys.stderr)
                print(f"    {result.stderr.strip()}", file=sys.stderr)
                sys.exit(1)
            print()

    if args.dry_run:
        print("Dry run complete. No jobs were submitted.")
    else:
        print("All jobs submitted.")


if __name__ == "__main__":
    main()
