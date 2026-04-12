"""
Angular steering benchmark orchestrator.

Submits one SLURM job per (model, degree) using ``scripts/job_template_angular.sh``,
mirroring ``caa/run_experiment.py`` but without a layer dimension.

Prerequisites:
  - Run ``angular/run_extraction.py`` so each model has
    ``{output_base}/{model_id}/{output_filename}`` and, with
    ``save_notebook_config: true``, the companion ``*_steering_config.npy``.

Environment:
  - ``PROJECT_ROOT``, ``PYTHON_BIN`` from ``env.sh`` (same as CAA).

Example:
    python angular/run_experiment.py configs/experiment/angular/benchmark/angular_deg_sweep.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, field_validator

from caa.utils import SlurmConfig, add_common_cli_args, filter_models, get_env_var, load_yaml_config, submit_sbatch


def steering_npy_from_pt(output_filename: str) -> str:
    """Match ``extract_angular.py``: ``.pt`` -> ``_steering_config.npy``."""
    if not output_filename.endswith(".pt"):
        return output_filename + "_steering_config.npy"
    return output_filename.replace(".pt", "_steering_config.npy")


class AngularExperimentOptions(BaseModel):
    """Must match extraction defaults for locating the steering npy next to the .pt."""

    output_filename: str = "angular_direction.pt"


class ExperimentConfig(BaseModel):
    name: str
    models: Dict[str, str]
    vector_dir: str = "data/vectors_angular"
    datasets: str = "glob(*,exclude=dummy)"
    judge: str = "contains_abstention_keyword"
    single_job: bool = True
    mode: str = "local"
    judge_label: str = "Keywords_judge"
    tag: str = ""
    angular: AngularExperimentOptions = AngularExperimentOptions()
    degrees: list[float] = [30.0]
    adaptive_mode: int = 1
    prompt_only: bool = False
    slurm: SlurmConfig = SlurmConfig()

    @field_validator("models", mode="before")
    @classmethod
    def validate_models(cls, v):
        if not isinstance(v, dict):
            raise ValueError("models must be a mapping of model_id: vector_subdirectory_name")
        return v


def build_sbatch_command(
    config: ExperimentConfig,
    model_id: str,
    vector_subdir: str,
    steering_config_path: str,
    degree: float,
    project_root: str,
    python_bin: str,
    user_email: str,
) -> tuple[list[str], dict[str, str]]:
    coeff_str = str(degree).replace(".", "_")
    tag_suffix = f"_{config.tag}" if config.tag else ""
    common_dir_name = (
        f"{vector_subdir}_{config.judge_label}_Angular_deg_{coeff_str}{tag_suffix}"
    )
    common_dir_base = f"{project_root}/data/{common_dir_name}"

    job_name = f"angular_{model_id}_{coeff_str}"

    env_vars = {
        "ANG_EXP_MODEL_ID": model_id,
        "ANG_EXP_STEERING_CONFIG": str(steering_config_path),
        "ANG_EXP_COMMON_DIR_BASE": str(common_dir_base),
        "ANG_EXP_DATASETS": str(config.datasets),
        "ANG_EXP_JUDGE": str(config.judge),
        "ANG_EXP_SINGLE_JOB": str(config.single_job),
        "ANG_EXP_MODE": str(config.mode),
        "ANG_EXP_DEGREE": str(degree),
        "ANG_EXP_ADAPTIVE_MODE": str(config.adaptive_mode),
        "ANG_EXP_PROMPT_ONLY": "1" if config.prompt_only else "0",
        "ANG_EXP_PYTHON_BIN": str(python_bin),
        "ANG_EXP_USER_EMAIL": str(user_email),
    }

    email_args = ""
    if user_email and config.mode != "local":
        email_args = (
            "+hydra.launcher.additional_parameters.mail-type=ALL "
            f"+hydra.launcher.additional_parameters.mail-user={user_email}"
        )
    env_vars["ANG_EXP_EMAIL_ARGS"] = email_args

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={config.slurm.partition}",
        f"--qos={config.slurm.qos}",
        f"--gres={config.slurm.gres}",
        f"--cpus-per-task={config.slurm.cpus_per_task}",
        f"--mem={config.slurm.mem}",
        f"--time={config.slurm.time}",
        "--export=ALL",
        "scripts/job_template_angular.sh",
    ]

    return cmd, env_vars


def resolve_steering_config(
    project_root: Path,
    vector_dir: str,
    vector_subdir: str,
    output_filename: str,
    dry_run: bool,
) -> Optional[Path]:
    npy_name = steering_npy_from_pt(output_filename)
    path = project_root / vector_dir / vector_subdir / npy_name
    if not path.is_file():
        if dry_run:
            print(f"  Warning: steering config not found (dry-run): {path}")
            return path
        print(f"Error: Angular steering config not found: {path}", file=sys.stderr)
        print("Run angular/run_extraction.py first (with save_notebook_config).", file=sys.stderr)
        sys.exit(1)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Submit Angular steering benchmark jobs from a YAML config"
    )
    parser.add_argument(
        "config",
        help="Path to experiment YAML (e.g. configs/experiment/angular/benchmark/angular_deg_sweep.yaml)",
    )
    add_common_cli_args(parser, include_layers=False)
    parser.add_argument(
        "--degrees",
        type=float,
        nargs="*",
        default=None,
        metavar="DEG",
        help="Override config degrees (e.g. --degrees 15 30 45). If omitted, uses YAML list.",
    )
    args = parser.parse_args()

    project_root = Path(get_env_var("PROJECT_ROOT", dry_run=args.dry_run))
    python_bin = get_env_var("PYTHON_BIN", dry_run=args.dry_run)
    user_email = os.environ.get("USER_EMAIL", "")

    config = load_yaml_config(args.config, ExperimentConfig)
    print(f"Experiment: {config.name}")
    print(f"Project root: {project_root}")
    print()

    models = filter_models(config.models, args.model)

    degrees = args.degrees if args.degrees is not None else config.degrees
    if not degrees:
        parser.error("degrees list is empty")

    for model_id, vector_subdir in models.items():
        steering_path = resolve_steering_config(
            project_root,
            config.vector_dir,
            vector_subdir,
            config.angular.output_filename,
            args.dry_run,
        )
        if steering_path is None:
            continue

        for degree in degrees:
            print(f"Model: {model_id}  degree={degree}")
            cmd, env_vars = build_sbatch_command(
                config=config,
                model_id=model_id,
                vector_subdir=vector_subdir,
                steering_config_path=str(steering_path),
                degree=degree,
                project_root=str(project_root),
                python_bin=python_bin,
                user_email=user_email,
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
