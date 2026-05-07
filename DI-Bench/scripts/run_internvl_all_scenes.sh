#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"

: "${MODEL_PATH:?Please export MODEL_PATH to a local InternVL3/InternVL3.5 model directory.}"
DATA_PATH="${DATA_PATH:-/path/to/Di-Bench}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/internvl_local_all_scenes}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/.cache}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
BBOX_MODE="${BBOX_MODE:-visual}"
SOURCE_MODE="${SOURCE_MODE:-full}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
MAX_TILES_PER_IMAGE="${MAX_TILES_PER_IMAGE:-12}"
LIMIT="${LIMIT:-}"

mkdir -p "${OUT_DIR}"

for scene_dir in "${DATA_PATH}"/scene_*; do
    scene="$(basename "${scene_dir}")"
    report_path="${OUT_DIR}/${scene}.xlsx"

    if [[ -f "${report_path}" ]]; then
        echo "Skipping ${scene}: ${report_path} already exists."
        continue
    fi

    cmd=(
        python "${ROOT_DIR}/evaluate.py"
        --backend internvl
        --model_path "${MODEL_PATH}"
        --data_path "${DATA_PATH}"
        --scene "${scene}"
        --report_path "${report_path}"
        --task_type all
        --bbox_mode "${BBOX_MODE}"
        --source_mode "${SOURCE_MODE}"
        --cache_dir "${CACHE_ROOT}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --temperature "${TEMPERATURE}"
        --top_p "${TOP_P}"
        --dtype "${DTYPE}"
        --device "${DEVICE}"
        --image_size "${IMAGE_SIZE}"
        --max_tiles_per_image "${MAX_TILES_PER_IMAGE}"
        --seed 42
    )

    if [[ -n "${LIMIT}" ]]; then
        cmd+=(--limit "${LIMIT}")
    fi

    "${cmd[@]}"
done
