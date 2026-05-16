#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"

: "${MODEL_PATH:?Please export MODEL_PATH to a local Qwen-VL model directory.}"
: "${DATA_PATH:?Please export DATA_PATH to the DI-Bench dataset root.}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/qwen_local_clean_all_scenes}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-bfloat16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SOURCE_MODE="${SOURCE_MODE:-full}"
BBOX_MODE="${BBOX_MODE:-visual}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/.cache}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
LIMIT="${LIMIT:-}"
FEW_SHOT_K="${FEW_SHOT_K:-0}"
FEW_SHOT_SCOPE="${FEW_SHOT_SCOPE:-same_task}"
FEW_SHOT_SELECTION="${FEW_SHOT_SELECTION:-first}"

mkdir -p "${OUT_DIR}"

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

for scene_dir in "${DATA_PATH}"/scene_*; do
    scene="$(basename "${scene_dir}")"
    report_path="${OUT_DIR}/${scene}.xlsx"

    if [[ -f "${report_path}" ]]; then
        echo "Skipping ${scene}: ${report_path} already exists."
        continue
    fi

    cmd=(
        python "${ROOT_DIR}/evaluate.py" \
        --model_path "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --scene "${scene}" \
        --report_path "${report_path}" \
        --task_type all \
        --bbox_mode "${BBOX_MODE}" \
        --source_mode "${SOURCE_MODE}" \
        --cache_dir "${CACHE_ROOT}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --temperature 0.0 \
        --top_p 1.0 \
        --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
        --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
        --max_num_seqs 1 \
        --dtype "${DTYPE}" \
        --limit_mm_per_prompt "${LIMIT_MM_PER_PROMPT}" \
        --enforce_eager \
        --seed 42
    )

    if [[ -n "${MAX_MODEL_LEN}" ]]; then
        cmd+=(--max_model_len "${MAX_MODEL_LEN}")
    fi
    if [[ -n "${LIMIT}" ]]; then
        cmd+=(--limit "${LIMIT}")
    fi
    if [[ "${FEW_SHOT_K}" -gt 0 ]]; then
        cmd+=(
            --few_shot_k "${FEW_SHOT_K}"
            --few_shot_scope "${FEW_SHOT_SCOPE}"
            --few_shot_selection "${FEW_SHOT_SELECTION}"
        )
    fi

    "${cmd[@]}"
done
