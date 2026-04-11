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
    parse_vector_indices,
    submit_sbatch,
)


class CAAConfig(BaseModel):
    coeffs: list[float] = [1.0]
    layers: Optional[str] = None


class ExperimentConfig(BaseModel):
    name: str
    models: Dict[str, str]
    vector_dir: str = "data/vectors"
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
    layer_str: str,
    project_root: str,
    python_bin: str,
    user_email: str,
    current_coeff: float,
) -> tuple[list[str], dict[str, str]]:
    """Build the sbatch command for a single model."""

    vector_dir = f"{project_root}/{config.vector_dir}/{vector_dir_name}"

    # common_dir base path
    coeff_str = str(current_coeff).replace(".", "_")
    tag_suffix = f"_{config.tag}" if config.tag else ""
    common_dir_name = (
        f"{vector_dir_name}_{config.judge_label}_CAA_coeff_{coeff_str}{tag_suffix}"
    )
    common_dir_base = f"{project_root}/data/{common_dir_name}"

    job_name = f"caa_{model_id}"
    job_template = "scripts/job_template.sh"

    # env vars to export to the job
    env_vars = {
        "EXP_MODEL_ID": model_id,
        "EXP_VECTOR_DIR": str(vector_dir),
        "EXP_COMMON_DIR_BASE": str(common_dir_base),
        "EXP_DATASETS": str(config.datasets),
        "EXP_JUDGE": str(config.judge),
        "EXP_SINGLE_JOB": str(config.single_job),
        "EXP_MODE": str(config.mode),
        "EXP_COEFF": str(current_coeff),
        "EXP_PYTHON_BIN": str(python_bin),
        "EXP_USER_EMAIL": str(user_email),
    }

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--array={layer_str}",
        f"--partition={config.slurm.partition}",
        f"--qos={config.slurm.qos}",
        f"--gres={config.slurm.gres}",
        f"--cpus-per-task={config.slurm.cpus_per_task}",
        f"--mem={config.slurm.mem}",
        f"--time={config.slurm.time}",
        "--export=ALL",
        job_template,
    ]

    return cmd, env_vars


def main():
    parser = argparse.ArgumentParser(
        description="CAA Experiment Orchestrator — submit SLURM jobs from YAML configs"
    )
    parser.add_argument(
        "config",
        help="Path to experiment YAML config (e.g. configs/experiment/caa_all.yaml)",
    )
    parser.add_argument(
        "--force-4d",
        action="store_true",
        help="Bypass safety net to allow alpha sweeps across all auto-detected layers",
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

    coeffs_to_sweep = config.caa.coeffs

    for model_id, vector_dir_name in models.items():
        vector_dir = Path(project_root) / config.vector_dir / vector_dir_name

        is_layer_explicit = False
        # resolve layer range or list
        if args.layers:
            layers_list = parse_vector_indices(args.layers)
            layer_str = ",".join(map(str, layers_list))
            print(f"Model: {model_id}")
            print(f"  Layers: {layer_str} (from CLI subset)")
            is_layer_explicit = True
        elif config.caa.layers:
            layers_list = parse_vector_indices([config.caa.layers])
            layer_str = ",".join(map(str, layers_list))
            print(f"Model: {model_id}")
            print(f"  Layers: {layer_str} (from config)")
            is_layer_explicit = True
        else:
            print(f"Model: {model_id}")
            print(f"  Scanning {vector_dir} for vectors...")
            min_layer, max_layer = detect_layers_from_vectors(
                vector_dir, dry_run=args.dry_run
            )
            layer_str = f"{min_layer}-{max_layer}"
            print(f"  Layers: {layer_str} (auto-detected)")
            is_layer_explicit = False

        if len(coeffs_to_sweep) > 1 and not is_layer_explicit and not args.force_4d:
            print(
                f"Error: Attempting to sweep {len(coeffs_to_sweep)} coeffs across all auto-detected layers for {model_id}.",
                file=sys.stderr,
            )
            print(
                "This creates a 4D sweep which may generate too many jobs.",
                file=sys.stderr,
            )
            print(
                "Safety net: Please either specify layers explicitly (in config or via --layers), or use the --force-4d flag.",
                file=sys.stderr,
            )
            sys.exit(1)

        for current_coeff in coeffs_to_sweep:
            print(f"  Coeff: {current_coeff}")
            # sbatch
            cmd, env_vars = build_sbatch_command(
                config=config,
                model_id=model_id,
                vector_dir_name=vector_dir_name,
                layer_str=layer_str,
                project_root=project_root,
                python_bin=python_bin,
                user_email=user_email,
                current_coeff=current_coeff,
            )

            env = os.environ.copy()
            env.update(env_vars)
            submit_sbatch(cmd, dry_run=args.dry_run, env=env)

    if args.dry_run:
        print("Dry run complete. No jobs were submitted.")
    else:
        print("All jobs submitted.")


if __name__ == "__main__":
    main()
