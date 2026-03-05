"""
Shared utilities for CAA orchestration scripts.

Provides common SLURM helpers, config parsing, environment variable handling,
and layer detection logic used by both run_experiment.py and run_extraction.py.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class SlurmConfig(BaseModel):
    partition: str = "gpu"
    qos: str = "gpu"
    gres: str = "gpu:1"
    cpus_per_task: int = 8
    mem: str = "82G"
    time: str = "04:00:00"

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


def detect_layers_from_vectors(vector_dir: Path, dry_run: bool = False) -> tuple[int, int]:
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


def detect_layers_from_model(model_name: str, dry_run: bool = False) -> tuple[int, int]:
    """
    Auto-detect layer range from a HF model config.
    Returns (num_layers // 2 - 1, num_layers - 1) following the paper's
    convention of sweeping the second half of the network.
    In dry-run mode, attempts the lookup but falls back to a placeholder.
    """
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        num_layers = config.num_hidden_layers
        min_layer = num_layers // 2 - 1
        max_layer = num_layers - 1
        return min_layer, max_layer
    except Exception as e:
        if dry_run:
            print(f"  Warning: Could not load config for {model_name} ({e}), "
                  f"using placeholder range 11-23")
            return 11, 23
        print(f"Error: Could not detect layers for {model_name}: {e}", file=sys.stderr)
        sys.exit(1)


def parse_layer_range(layer_spec: str) -> tuple[int, int]:
    """Parse explicit layer spec like '15-31'."""
    parts = layer_spec.split("-")
    if len(parts) != 2:
        print(f"Error: Invalid layer range '{layer_spec}'. Expected 'MIN-MAX'.", file=sys.stderr)
        sys.exit(1)
    return int(parts[0]), int(parts[1])


def submit_sbatch(cmd: list[str], dry_run: bool = False) -> None:
    """Submit an sbatch command, or print it in dry-run mode."""
    if dry_run:
        print(f"\n  [DRY RUN] Would execute:")
        print("    " + " \\\n      ".join(cmd))
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


def load_yaml_config(path: str, schema_cls: type[BaseModel]) -> BaseModel:
    """Load and validate a YAML config against a pydantic schema."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    try:
        return schema_cls(**raw)
    except Exception as e:
        print(f"Error loading config {path}:\n{e}", file=sys.stderr)
        sys.exit(1)


def add_common_cli_args(parser) -> None:
    """Add --dry-run and --model arguments common to all orchestrators."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run only this model (must be a key in the config's models map)",
    )


def filter_models(models: dict, model_key: Optional[str]) -> dict:
    """Filter model dict to a single model if --model is specified."""
    if model_key is None:
        return models
    if model_key not in models:
        available = ", ".join(models.keys())
        print(
            f"Error: Model '{model_key}' not in config. Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)
    return {model_key: models[model_key]}
