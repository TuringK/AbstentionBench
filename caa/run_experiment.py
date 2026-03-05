"""
CAA Experiment Manager

Reads a declarative experiment YAML config and submits SLURM array jobs
for each model, auto-detecting layer ranges from vector files.

Usage:
    # Dry run (print commands without submitting)
    python caa/run_experiment.py configs/experiment/caa_all.yaml --dry-run

    # Submit all models
    python caa/run_experiment.py configs/experiment/caa_all.yaml

    # Submit only a specific model
    python caa/run_experiment.py configs/experiment/caa_all.yaml --model allenai_llama_3_1_tulu_3_1_8B

Environment:
    Requires env.sh to be sourced first (sets PROJECT_ROOT, PYTHON_BIN, etc.)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, field_validator

from caa.utils import (
    SlurmConfig,
    add_common_cli_args,
    detect_layers_from_vectors,
    filter_models,
    get_env_var,
    load_yaml_config,
    parse_layer_range,
    submit_sbatch,
)


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


def main():
    parser = argparse.ArgumentParser(
        description="CAA Experiment Orchestrator — submit SLURM jobs from YAML configs"
    )
    parser.add_argument(
        "config",
        help="Path to experiment YAML config (e.g. configs/experiment/caa_all.yaml)",
    )
    add_common_cli_args(parser)
    args = parser.parse_args()

    # load env vars
    project_root = get_env_var("PROJECT_ROOT", dry_run=args.dry_run)
    python_bin = get_env_var("PYTHON_BIN", dry_run=args.dry_run)
    user_email = os.environ.get("USER_EMAIL", "")

    # load config
    config = load_yaml_config(args.config, ExperimentConfig)
    print(f"Experiment: {config.name}")
    print(f"Project root: {project_root}")
    print()

    # filter models
    models = filter_models(config.models, args.model)

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
            min_layer, max_layer = detect_layers_from_vectors(vector_dir, dry_run=args.dry_run)
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

        submit_sbatch(cmd, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run complete. No jobs were submitted.")
    else:
        print("All jobs submitted.")


if __name__ == "__main__":
    main()
