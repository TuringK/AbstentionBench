"""Explicit paths and model registry for multi-method plotting."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "v3_csv"

COEFFS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

MODEL_PARAMS = {
    "Qwen 2.5 0.5B": 0.5,
    "Qwen 2.5 1.5B": 1.5,
    "Qwen 2.5 3B": 3.0,
    "Qwen 2.5 7B": 7.0,
    "Tulu 3.1 8B": 8.0,
    "Gemma 3 1B": 1.0,
}


def sweep_csv(model_key: str, coeff: float) -> Path:
    coeff_str = f"{coeff:.1f}".replace(".", "_")
    return DATA_ROOT / f"{model_key}_sweep" / f"{model_key}_v3_sweep_{coeff_str}.csv"


def sweep_csvs(model_key: str, coeffs: list[float] | None = None) -> dict[float, Path]:
    coeffs = coeffs or COEFFS
    return {c: sweep_csv(model_key, c) for c in coeffs}


def rulebreakers_csv(model_key: str, coeff: float, layer: int) -> Path:
    coeff_str = f"{coeff:.1f}".replace(".", "_")
    return (
        DATA_ROOT
        / "rulenreakers"
        / f"{model_key}_v3_sweep_{coeff_str}_vec_{layer}_rulebreakers.csv"
    )


CAA_MODELS = {
    "qwen_0_5": {
        "display_name": "Qwen 2.5 0.5B",
        "params_b": 0.5,
        "sweep_csvs": sweep_csvs("qwen_0_5"),
        "rulebreakers_csv": rulebreakers_csv("qwen_0_5", 2.0, 14),
    },
    "qwen_1_5": {
        "display_name": "Qwen 2.5 1.5B",
        "params_b": 1.5,
        "sweep_csvs": sweep_csvs("qwen_1_5"),
        "rulebreakers_csv": rulebreakers_csv("qwen_1_5", 2.0, 15),
    },
    "qwen_3": {
        "display_name": "Qwen 2.5 3B",
        "params_b": 3.0,
        "sweep_csvs": sweep_csvs("qwen_3"),
        "rulebreakers_csv": rulebreakers_csv("qwen_3", 3.0, 19),
    },
    "qwen_7": {
        "display_name": "Qwen 2.5 7B",
        "params_b": 7.0,
        "sweep_csvs": sweep_csvs("qwen_7"),
        "rulebreakers_csv": rulebreakers_csv("qwen_7", 2.0, 16),
    },
    "gemma_1": {
        "display_name": "Gemma 3 1B",
        "params_b": 1.0,
        "sweep_csvs": sweep_csvs("gemma_1"),
        "rulebreakers_csv": rulebreakers_csv("gemma_1", 2.0, 12),
    },
    "tulu_8": {
        "display_name": "Tulu 3.1 8B",
        "params_b": 8.0,
        "sweep_csvs": sweep_csvs("tulu_8"),
        "rulebreakers_csv": rulebreakers_csv("tulu_8", 3.0, 16),
    },
}

DPO_MODELS = {
    "qwen_0_5": {
        "display_name": "Qwen 2.5 0.5B",
        "params_b": 0.5,
        "csv": REPO_ROOT / "data/dpo_v2/FinalAbstentionModels/qwen_2_5_0_5B_dpo_abstention/keyword_filtered.csv",
    },
    "qwen_1_5": {
        "display_name": "Qwen 2.5 1.5B",
        "params_b": 1.5,
        "csv": REPO_ROOT / "data/dpo_v2/FinalAbstentionModels/qwen_2_5_1_5B_dpo_abstention_batch_8/keyword_filtered.csv",
    },
    "qwen_3": {
        "display_name": "Qwen 2.5 3B",
        "params_b": 3.0,
        "csv": REPO_ROOT / "data/dpo_v2/FinalAbstentionModels/qwen2_5_3B_it_dpo_abstention_128_batch/keyword_filtered.csv",
    },
    "qwen_7": {
        "display_name": "Qwen 2.5 7B",
        "params_b": 7.0,
        "csv": REPO_ROOT / "data/dpo_v2/FinalAbstentionModels/qwen2_5_7B_it_dpo_abstention_high_lr/keyword_filtered.csv",
    },
    "gemma_1": {
        "display_name": "Gemma 3 1B",
        "params_b": 1.0,
        "csv": REPO_ROOT / "data/dpo_v2/FinalAbstentionModels/gemma_3_1b_lora_dpo_abstention/keyword_filtered.csv",
    },
    "tulu_8": {
        "display_name": "Tulu 3.1 8B",
        "params_b": 8.0,
        "csv": REPO_ROOT / "data/dpo_v2/FinalAbstentionModels/llama_3_1_tulu_8b_dpo_abstention/keyword_filtered.csv",
    },
}

GRPO_MODELS = {
    "qwen_0_5": {
        "display_name": "Qwen 2.5 0.5B",
        "params_b": 0.5,
        "csv": REPO_ROOT / "data/grpo/final/csvs/q0_5_res.csv",
    },
    "qwen_1_5": {
        "display_name": "Qwen 2.5 1.5B",
        "params_b": 1.5,
        "csv": REPO_ROOT / "data/grpo/final/csvs/q1_5_res.csv",
    },
    "qwen_3": {
        "display_name": "Qwen 2.5 3B",
        "params_b": 3.0,
        "csv": REPO_ROOT / "data/grpo/final/csvs/q3_0_res.csv",
    },
    "qwen_7": {
        "display_name": "Qwen 2.5 7B",
        "params_b": 7.0,
        "csv": REPO_ROOT / "data/grpo/final/csvs/q7_0_res.csv",
    },
    "gemma_1": {
        "display_name": "Gemma 3 1B",
        "params_b": 1.0,
        "csv": REPO_ROOT / "data/grpo/final/csvs/gemma_res.csv",
    },
    "tulu_8": {
        "display_name": "Tulu 3.1 8B",
        "params_b": 8.0,
        "csv": REPO_ROOT / "data/grpo/final/csvs/llama_res.csv",
    },
}

LORA_MODELS = {
    "qwen_0_5": {
        "display_name": "Qwen 2.5 0.5B",
        "params_b": 0.5,
        "csv": REPO_ROOT / "data/peft_csv_v2/PEFT/qwen2_5_0_5b_lora_abstention_GenericModel.csv",
    },
    "qwen_1_5": {
        "display_name": "Qwen 2.5 1.5B",
        "params_b": 1.5,
        "csv": REPO_ROOT / "data/peft_csv_v2/PEFT/qwen2_5_1_5b_lora_abstention_GenericModel.csv",
    },
    "qwen_3": {
        "display_name": "Qwen 2.5 3B",
        "params_b": 3.0,
        "csv": REPO_ROOT / "data/peft_csv_v2/PEFT/qwen2_5_3b_lora_abstention_GenericModel.csv",
    },
    "qwen_7": {
        "display_name": "Qwen 2.5 7B",
        "params_b": 7.0,
        "csv": REPO_ROOT / "data/peft_csv_v2/PEFT/qwen2_5_7b_lora_abstention_GenericModel.csv",
    },
    "gemma_1": {
        "display_name": "Gemma 3 1B",
        "params_b": 1.0,
        "csv": REPO_ROOT / "data/peft_csv_v2/PEFT/gemma_3_1b_lora_abstention_GenericModel.csv",
    },
    "tulu_8": {
        "display_name": "Tulu 3.1 8B",
        "params_b": 8.0,
        "csv": REPO_ROOT / "data/peft_csv_v2/PEFT/tulu_3_1_8b_lora_abstention_GenericModel.csv",
    },
}

DORA_MODELS = {
    "qwen_0_5": {
        "display_name": "Qwen 2.5 0.5B",
        "params_b": 0.5,
        "csv": REPO_ROOT / "data/peft_csv_v2/PEFT/qwen2_5_1_5b_dora_abstention_GenericModel.csv",
    },
    "qwen_1_5": {
        "display_name": "Qwen 2.5 1.5B",
        "params_b": 1.5,
        "csv": REPO_ROOT / "data/peft_csv_v2/PEFT/qwen2_5_1_5b_dora_abstention_GenericModel.csv",
    },
    "qwen_3": {
        "display_name": "Qwen 2.5 3B",
        "params_b": 3.0,
        "csv": REPO_ROOT / "data/peft_csv_v2/PEFT/qwen2_5_3b_dora_abstention_GenericModel.csv",
    },
    "qwen_7": {
        "display_name": "Qwen 2.5 7B",
        "params_b": 7.0,
        "csv": REPO_ROOT / "data/peft_csv_v2/PEFT/qwen2_5_7b_dora_abstention_GenericModel.csv",
    },
    "gemma_1": {
        "display_name": "Gemma 3 1B",
        "params_b": 1.0,
        "csv": REPO_ROOT / "data/peft_csv_v2/PEFT/gemma_3_1b_dora_abstention_GenericModel.csv",
    },
    "tulu_8": {
        "display_name": "Tulu 3.1 8B",
        "params_b": 8.0,
        "csv": REPO_ROOT / "data/peft_csv_v2/PEFT/tulu_3_1_8b_dora_abstention_GenericModel.csv",
    },
}

VANILLA_MODELS = {
    "qwen_0_5": {
        "display_name": "Qwen 2.5 0.5B",
        "params_b": 0.5,
        "csv": REPO_ROOT / "data/vanilla_retagged_new_detector_csv/qwen_2_5_0_5b_vanilla.csv",
    },
    "qwen_1_5": {
        "display_name": "Qwen 2.5 1.5B",
        "params_b": 1.5,
        "csv": REPO_ROOT / "data/vanilla_retagged_new_detector_csv/qwen_2_5_1_5b_vanilla.csv",
    },
    "qwen_3": {
        "display_name": "Qwen 2.5 3B",
        "params_b": 3.0,
        "csv": REPO_ROOT / "data/vanilla_retagged_new_detector_csv/qwen_2_5_3b_vanilla.csv",
    },
    "qwen_7": {
        "display_name": "Qwen 2.5 7B",
        "params_b": 7.0,
        "csv": REPO_ROOT / "data/vanilla_retagged_new_detector_csv/qwen_2_5_7b_vanilla.csv",
    },
    "gemma_1": {
        "display_name": "Gemma 3 1B",
        "params_b": 1.0,
        "csv": REPO_ROOT / "data/vanilla_retagged_new_detector_csv/gemma_3_1b_vanilla.csv",
    },
    "tulu_8": {
        "display_name": "Tulu 3.1 8B",
        "params_b": 8.0,
        "csv": REPO_ROOT / "data/vanilla_retagged_new_detector_csv/tulu_3_1_8b_vanilla.csv",
    },
}

# Registry of methods to plot. To add a new method:
#   1. Define a *_MODELS dict above (same shape as DPO_MODELS / LORA_MODELS).
#   2. Add an entry here with the display name, loader type, and models dict.
#   3. Pass METHODS into generate_method_plots() — no changes needed there for flat_csv
#      or caa_sweep loaders. For a new data format, add a loader and register its type.
METHODS = {
    "CAA": {"loader": "caa_sweep", "models": CAA_MODELS},
    "DPO": {"loader": "flat_csv", "models": DPO_MODELS},
    "GRPO": {"loader": "flat_csv", "models": GRPO_MODELS},
    "LoRA": {"loader": "flat_csv", "models": LORA_MODELS},
    "DoRA": {"loader": "flat_csv", "models": DORA_MODELS},
    "Vanilla": {"loader": "flat_csv", "models": VANILLA_MODELS},
}