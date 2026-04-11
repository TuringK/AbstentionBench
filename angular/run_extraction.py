"""
Angular Steering Extraction Orchestrator

Reads a declarative YAML config and submits SLURM jobs (or runs locally)
to extract one Angular steering artifact per model.
"""

import argparse
import subprocess
import sys
from typing import Dict

from pydantic import BaseModel, field_validator

from caa.utils import (
    SlurmConfig,
    add_common_cli_args,
    filter_models,
    get_env_var,
    load_yaml_config,
    submit_sbatch,
)


class AngularExtractionOptions(BaseModel):
    use_system_prompt: bool = True
    max_samples: int = 512
    batch_size: int = 4
    norm_floor: float = 0.0
    exclude_tasks: str = ""
    seed: int = 42
    dedupe: bool = True
    stratified: bool = False
    suffix_pool: str = "last"
    save_notebook_config: bool = True
    log_level: str = "INFO"
    output_filename: str = "angular_steering.pt"


class AngularExtractionConfig(BaseModel):
    name: str
    models: Dict[str, str]  # model_id -> HF model name
    data_path: str = "data/abstention_training_dataset.json"
    output_base: str = "data/angular_vectors"
    extraction: AngularExtractionOptions = AngularExtractionOptions()
    slurm: SlurmConfig = SlurmConfig(time="02:00:00")

    @field_validator("models", mode="before")
    @classmethod
    def validate_models(cls, v):
        if not isinstance(v, dict):
            raise ValueError(
                "models must be a mapping of model_id: hf_model_name, e.g.\n"
                "  Qwen2_5_0_5B_Instruct: Qwen/Qwen2.5-0.5B-Instruct"
            )
        return v


def build_sbatch_command(
    config: AngularExtractionConfig,
    model_id: str,
    hf_model_name: str,
    project_root: str,
    python_bin: str,
) -> list[str]:
    data_path = f"{project_root}/{config.data_path}"
    output_path = (
        f"{project_root}/{config.output_base}/{model_id}/"
        f"{config.extraction.output_filename}"
    )

    export_vars = [
        "ALL",
        f"ANG_MODEL_NAME={hf_model_name}",
        f"ANG_DATA_PATH={data_path}",
        f"ANG_OUTPUT_PATH={output_path}",
        f"ANG_PYTHON_BIN={python_bin}",
        f"ANG_USE_SYSTEM_PROMPT={'1' if config.extraction.use_system_prompt else '0'}",
        f"ANG_MAX_SAMPLES={config.extraction.max_samples}",
        f"ANG_BATCH_SIZE={config.extraction.batch_size}",
        f"ANG_NORM_FLOOR={config.extraction.norm_floor}",
        f"ANG_EXCLUDE_TASKS={config.extraction.exclude_tasks}",
        f"ANG_SEED={config.extraction.seed}",
        f"ANG_DEDUPE={'1' if config.extraction.dedupe else '0'}",
        f"ANG_STRATIFIED={'1' if config.extraction.stratified else '0'}",
        f"ANG_SUFFIX_POOL={config.extraction.suffix_pool}",
        f"ANG_SAVE_NOTEBOOK_CONFIG={'1' if config.extraction.save_notebook_config else '0'}",
        f"ANG_LOG_LEVEL={config.extraction.log_level}",
    ]

    return [
        "sbatch",
        f"--job-name=extract_angular_{model_id}",
        f"--partition={config.slurm.partition}",
        f"--qos={config.slurm.qos}",
        f"--gres={config.slurm.gres}",
        f"--cpus-per-task={config.slurm.cpus_per_task}",
        f"--mem={config.slurm.mem}",
        f"--time={config.slurm.time}",
        f"--export={','.join(export_vars)}",
        "scripts/extraction_angular_template.sh",
    ]


def run_local(
    config: AngularExtractionConfig,
    hf_model_name: str,
    model_id: str,
    project_root: str,
    python_bin: str,
    dry_run: bool = False,
) -> None:
    data_path = f"{project_root}/{config.data_path}"
    output_path = (
        f"{project_root}/{config.output_base}/{model_id}/"
        f"{config.extraction.output_filename}"
    )

    cmd = [
        python_bin, "angular/extract_angular.py",
        "--model_name", hf_model_name,
        "--data_path", data_path,
        "--output_path", output_path,
        "--max_samples", str(config.extraction.max_samples),
        "--batch_size", str(config.extraction.batch_size),
        "--norm_floor", str(config.extraction.norm_floor),
        "--seed", str(config.extraction.seed),
        "--suffix_pool", config.extraction.suffix_pool,
        "--log_level", config.extraction.log_level,
    ]

    if config.extraction.use_system_prompt:
        cmd.append("--use_system_prompt")
    if config.extraction.exclude_tasks:
        cmd.extend(["--exclude_tasks", config.extraction.exclude_tasks])
    if not config.extraction.dedupe:
        cmd.append("--no_dedupe")
    if config.extraction.stratified:
        cmd.append("--stratified")
    if config.extraction.save_notebook_config:
        cmd.append("--save_notebook_config")

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return

    print(f"  Output: {output_path}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            f"Angular extraction failed for {model_id} (exit {result.returncode})",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Angular extraction orchestrator - submit SLURM jobs or run locally"
    )
    parser.add_argument(
        "config",
        help="Path to angular extraction YAML config",
    )
    add_common_cli_args(parser, include_layers=False)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run extraction locally instead of submitting SLURM jobs",
    )
    args = parser.parse_args()

    project_root = get_env_var("PROJECT_ROOT", dry_run=args.dry_run)
    python_bin = get_env_var("PYTHON_BIN", dry_run=args.dry_run)
    config = load_yaml_config(args.config, AngularExtractionConfig)

    mode_label = "local" if args.local else "sbatch"
    print(f"Angular extraction job: {config.name} ({mode_label})")
    print(f"Project root: {project_root}")
    print(f"Output base: {config.output_base}")
    print()

    models = filter_models(config.models, args.model)
    for model_id, hf_model_name in models.items():
        print(f"Model: {model_id} ({hf_model_name})")
        if args.local:
            run_local(
                config=config,
                hf_model_name=hf_model_name,
                model_id=model_id,
                project_root=project_root,
                python_bin=python_bin,
                dry_run=args.dry_run,
            )
        else:
            cmd = build_sbatch_command(
                config=config,
                model_id=model_id,
                hf_model_name=hf_model_name,
                project_root=project_root,
                python_bin=python_bin,
            )
            submit_sbatch(cmd, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run complete. No jobs were submitted/executed.")
    elif not args.local:
        print("All jobs submitted.")


if __name__ == "__main__":
    main()

