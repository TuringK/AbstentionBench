#!/usr/bin/env bash
# Resume `caa_v3_all_sweep` without redoing completed work.
#
# Prerequisite: `source env.sh` (sets PROJECT_ROOT, PYTHON_BIN, etc.)
#
# This script matches the following resume point:
#   - Qwen2_5_7B_Instruct: from coeff 6.0 layer 18 through end of that coeff,
#     then coeffs 7.0–10.0 for all layers in the vector sweep
#   - Gemma3_1B_Instruct, Llama3_1_Tulu_3_1_8B: full coeff × layer sweeps
#
# Adjust QWEN7_MAX_LAYER if your `vec_layer_*.pt` set uses a different top index
# (should match the max from `data/vectors_caa_v3/Qwen2_5_7B_Instruct/`).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/experiment/caa/benchmark/caa_v3_all_sweep_recovery.yaml"
# Last layer index included in the coeff-6.0 tail (28-layer Qwen2.5-7B → 0..27).
QWEN7_MAX_LAYER="${QWEN7_MAX_LAYER:-27}"

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  echo "Error: PROJECT_ROOT is not set. Source env.sh from the repo root first." >&2
  exit 1
fi

if [[ "$ROOT" != "$PROJECT_ROOT" ]]; then
  echo "Warning: cd'd to $ROOT but PROJECT_ROOT=$PROJECT_ROOT; using current dir for config paths." >&2
fi

py=(python -u caa/run_experiment.py)

echo "=== 1/4 Qwen2.5-7B coeff 6.0, layers ${QWEN7_MAX_LAYER} only from 18 upward ==="
"${py[@]}" "$CONFIG" --model qwen2_5_7B_instruct --coeffs 6.0 --layers "18-${QWEN7_MAX_LAYER}"

echo "=== 2/4 Qwen2.5-7B coeffs 7.0–10.0, all layers in vector set (4D sweep) ==="
"${py[@]}" "$CONFIG" --model qwen2_5_7B_instruct --coeffs 7.0 8.0 9.0 10.0 --force-4d

echo "=== 3/4 Gemma3 1B, full v3 coeff sweep ==="
"${py[@]}" "$CONFIG" --model gemma_3_1B_instruct --force-4d

echo "=== 4/4 Llama3.1 Tulu 8B, full v3 coeff sweep ==="
"${py[@]}" "$CONFIG" --model allenai_llama_3_1_tulu_3_1_8B --force-4d

echo "Recovery submissions finished."
