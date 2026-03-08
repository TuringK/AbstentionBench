"""
CAA Vector Extraction Orchestrator

Reads a declarative YAML config and submits SLURM array jobs (or runs
locally) to extract steering vectors for each model across a range of layers.

Usage:
    # Dry run (print commands without submitting)
    python caa/run_extraction.py configs/experiment/extract_all.yaml --dry-run

    # Submit all models via SLURM
    python caa/run_extraction.py configs/experiment/extract_all.yaml

    # Run locally / in an interactive session (no sbatch)
    python caa/run_extraction.py configs/experiment/extract_all.yaml --local

    # Local, single model
    python caa/run_extraction.py configs/experiment/extract_all.yaml --local --model Qwen2_5_0_5B_Instruct

Environment:
    Requires env.sh to be sourced first (sets PROJECT_ROOT, PYTHON_BIN, etc.)
"""

import argparse
import subprocess
import sys
from typing import Dict, Optional

from pydantic import BaseModel, field_validator

from caa.utils import (
    SlurmConfig,
    add_common_cli_args,
    detect_layers_from_model,
    filter_models,
    get_env_var,
    load_yaml_config,
    parse_layer_range,
    submit_sbatch,
)


class ExtractionOptions(BaseModel):
    use_system_prompt: bool = True
    weighted: bool = False
    exclude_scenarios: str = ""
    layers: Optional[str] = None


class ExtractionConfig(BaseModel):
    name: str
    models: Dict[str, str]   # model_id -> HF model name
    data_path: str = "data/sample_pairs_with_scenario.csv"
    output_base: str = "data/vectors"
    extraction: ExtractionOptions = ExtractionOptions()
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


def build_extraction_sbatch(
    config: ExtractionConfig,
    model_id: str,
    hf_model_name: str,
    min_layer: int,
    max_layer: int,
    project_root: str,
    python_bin: str,
) -> list[str]:
    """Build the sbatch command for extracting vectors for a single model."""

    data_path = f"{project_root}/{config.data_path}"
    output_dir = f"{project_root}/{config.output_base}/{model_id}"

    job_name = f"extract_{model_id}"
    job_template = "scripts/extraction_template.sh"

    # env vars to export to the job
    export_vars = [
        "ALL",
        f"EXT_MODEL_NAME={hf_model_name}",
        f"EXT_MODEL_ID={model_id}",
        f"EXT_DATA_PATH={data_path}",
        f"EXT_OUTPUT_DIR={output_dir}",
        f"EXT_PYTHON_BIN={python_bin}",
        f"EXT_USE_SYSTEM_PROMPT={'1' if config.extraction.use_system_prompt else '0'}",
        f"EXT_WEIGHTED={'1' if config.extraction.weighted else '0'}",
        f"EXT_EXCLUDE_SCENARIOS={config.extraction.exclude_scenarios}",
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


def run_local(
    config: ExtractionConfig,
    model_id: str,
    hf_model_name: str,
    min_layer: int,
    max_layer: int,
    project_root: str,
    python_bin: str,
    dry_run: bool = False,
) -> None:
    """Run extraction locally, looping through layers sequentially."""

    data_path = f"{project_root}/{config.data_path}"
    output_dir = f"{project_root}/{config.output_base}/{model_id}"

    for layer_idx in range(min_layer, max_layer + 1):
        output_file = f"{output_dir}/vec_layer_{layer_idx}.pt"

        cmd = [
            python_bin, "caa/extract_caa_vectors.py",
            "--model_name", hf_model_name,
            "--data_path", data_path,
            "--output_path", output_file,
            "--layer_idx", str(layer_idx),
        ]

        if config.extraction.use_system_prompt:
            cmd.append("--use_system_prompt")
        if config.extraction.weighted:
            cmd.append("--weighted")
        if config.extraction.exclude_scenarios:
            cmd.extend(["--exclude_scenarios", config.extraction.exclude_scenarios])

        if dry_run:
            print(f"  [DRY RUN] Layer {layer_idx}: {' '.join(cmd)}")
            continue

        print(f"\nLayer {layer_idx}/{max_layer}")
        print(f"Output: {output_file}\n")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Extraction failed for layer {layer_idx} (exit {result.returncode})",
                  file=sys.stderr)
            sys.exit(1)

    if not dry_run:
        print(f"\nAll layers ({min_layer}-{max_layer}) extracted for {model_id}.")


def main():
    parser = argparse.ArgumentParser(
        description="CAA Vector Extraction Orchestrator - submit SLURM jobs or run locally"
    )
    parser.add_argument(
        "config",
        help="Path to extraction YAML config (e.g. configs/experiment/extract_all.yaml)",
    )
    add_common_cli_args(parser)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run extraction locally (sequential) instead of submitting SLURM jobs",
    )
    args = parser.parse_args()

    # load env vars
    project_root = get_env_var("PROJECT_ROOT", dry_run=args.dry_run)
    python_bin = get_env_var("PYTHON_BIN", dry_run=args.dry_run)

    # load config
    config = load_yaml_config(args.config, ExtractionConfig)
    mode_label = "local" if args.local else "sbatch"
    print(f"Extraction job: {config.name} ({mode_label})")
    print(f"Project root: {project_root}")
    print(f"Output base: {config.output_base}")
    if config.extraction.weighted:
        excl = config.extraction.exclude_scenarios or "(none)"
        print(f"Aggregation: scenario-weighted (excluding: {excl})")
    else:
        print("Aggregation: naive (global mean)")
    print()

    # filter models
    models = filter_models(config.models, args.model)

    for model_id, hf_model_name in models.items():
        # resolve layer range
        if config.extraction.layers:
            min_layer, max_layer = parse_layer_range(config.extraction.layers)
            print(f"Model: {model_id} ({hf_model_name})")
            print(f"  Layers: {min_layer}-{max_layer} (from config)")
        else:
            print(f"Model: {model_id} ({hf_model_name})")
            print("  Auto-detecting layer range from model config...")
            min_layer, max_layer = detect_layers_from_model(
                hf_model_name, dry_run=args.dry_run
            )
            print(f"  Layers: {min_layer}-{max_layer} (auto-detected)")

        if args.local:
            run_local(
                config=config,
                model_id=model_id,
                hf_model_name=hf_model_name,
                min_layer=min_layer,
                max_layer=max_layer,
                project_root=project_root,
                python_bin=python_bin,
                dry_run=args.dry_run,
            )
        else:
            cmd = build_extraction_sbatch(
                config=config,
                model_id=model_id,
                hf_model_name=hf_model_name,
                min_layer=min_layer,
                max_layer=max_layer,
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
