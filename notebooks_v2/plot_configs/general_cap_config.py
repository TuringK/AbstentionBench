"""Paths and display settings for general-capability (PPL / QA) plots."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PPL_DATA_DIR = REPO_ROOT / "data" / "general_cap" / "ppl"
QA_DATA_DIR = REPO_ROOT / "data" / "general_cap" / "qa"

DEFAULT_PPL_OUTPUT_DIR = REPO_ROOT / "notebooks_v2" / "general_cap_outputs" / "ppl"
DEFAULT_QA_OUTPUT_DIR = REPO_ROOT / "notebooks_v2" / "general_cap_outputs" / "qa"

MODEL_FILES: dict[str, str] = {
    "qwen0_5": "Qwen 2.5 0.5B",
    "qwen1_5": "Qwen 2.5 1.5B",
    "qwen3": "Qwen 2.5 3B",
    "qwen7": "Qwen 2.5 7B",
    "tulu8": "Tulu 3.1 8B",
}

MODEL_ORDER = [
    "qwen0_5",
    "qwen1_5",
    "qwen3",
    "qwen7",
    "tulu8",
]

METHOD_ORDER = ["Vanilla", "DPO", "GRPO", "DoRA", "LoRA", "CAA"]

LANGUAGE_ORDER = [
    "English",
    "Arabic",
    "German",
    "Spanish",
    "Hindi",
    "Russian",
    "Chinese",
]

LANGUAGE_LABELS: dict[str, str] = {
    "English": "EN",
    "Arabic": "AR",
    "German": "DE",
    "Spanish": "ES",
    "Hindi": "HI",
    "Russian": "RU",
    "Chinese": "ZH",
}

DATASET_TO_LANGUAGE: dict[str, str] = {
    "PPL English": "English",
    "PPL Arabic": "Arabic",
    "PPL German": "German",
    "PPL Spanish": "Spanish",
    "PPL Hindi": "Hindi",
    "PPL Russian": "Russian",
    "PPL Chinese": "Chinese",
}

METHOD_COLORS: dict[str, str] = {
    "Vanilla": "#999999",
    "DPO": "#0072B2",
    "GRPO": "#009E73",
    "DoRA": "#CC79A7",
    "LoRA": "#E69F00",
    "CAA": "#D55E00",
}

COMPARISON_METHOD_COLOR = "#ee854a"

MODEL_COLORS: dict[str, str] = {
    "Qwen 2.5 0.5B": "#0072B2",
    "Qwen 2.5 1.5B": "#E69F00",
    "Qwen 2.5 3B": "#009E73",
    "Qwen 2.5 7B": "#CC79A7",
    "Tulu 3.1 8B": "#D55E00",
}

MODEL_SHORT_LABELS: dict[str, str] = {
    "Qwen 2.5 0.5B": "0.5B",
    "Qwen 2.5 1.5B": "1.5B",
    "Qwen 2.5 3B": "3B",
    "Qwen 2.5 7B": "7B",
    "Tulu 3.1 8B": "8B",
}

QA_BENCHMARK_ORDER = [
    "AgiEval",
    "CSQA2",
    "GPQA",
    "GSM8K",
    "MMLU Redux",
    "MuSR",
    "StrategyQA",
    "Belebele",
]

VANILLA_BASELINE_METHOD = "Vanilla"


def benchmark_slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def method_slug(name: str) -> str:
    return name.lower().replace(" ", "_")
