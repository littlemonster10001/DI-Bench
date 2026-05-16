#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${PROJECT_ROOT}/DI-Bench/evaluate.py}"
: "${DATA_PATH:?Please export DATA_PATH to the DI-Bench dataset root.}"
SCENE="${SCENE:-scene_001}"
: "${MODEL_PATH:?Please export MODEL_PATH to a local Qwen-VL model directory.}"
MODEL_NAME="$(basename "${MODEL_PATH}")"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/${MODEL_NAME}_all_tasks_smoke_${SCENE}}"

BBOX_MODE="${BBOX_MODE:-visual}"
SOURCE_MODE="${SOURCE_MODE:-full}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

TASKS=(
  Image_Retrieval
  Image_Level_Cross_View_Matching
  Object_Level_Cross_View_Matching
  building_damage_assessment
  Building_Damage_Counting
  Road_Damage_Reasoning
  poi_alignment
  population_estimation
  height_alignment
  height_comparison
  Area_Estimation
  Length_Estimation
  Distance_Estimation
  route_planning
  uav_landing_assessment
)

if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
fi
if [[ -n "${CONDA_ENV}" ]]; then
  conda activate "${CONDA_ENV}"
fi

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "${OUT_DIR}"

echo "Model: ${MODEL_PATH}"
echo "Scene: ${SCENE}"
echo "Output: ${OUT_DIR}"
echo "VLLM_WORKER_MULTIPROC_METHOD: ${VLLM_WORKER_MULTIPROC_METHOD}"
echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
if [[ -n "${MAX_MODEL_LEN}" ]]; then
  echo "Max model len: ${MAX_MODEL_LEN}"
else
  echo "Max model len: model default"
fi

for TASK in "${TASKS[@]}"; do
  echo "========== Testing task: ${TASK} =========="

  EXTRA_ARGS=()
  if [[ -n "${MAX_MODEL_LEN}" ]]; then
    EXTRA_ARGS+=(--max_model_len "${MAX_MODEL_LEN}")
  fi

  python "${EVAL_SCRIPT}" \
    --backend qwen \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --scene "${SCENE}" \
    --task_type "${TASK}" \
    --limit 1 \
    --report_path "${OUT_DIR}/${TASK}.xlsx" \
    --bbox_mode "${BBOX_MODE}" \
    --source_mode "${SOURCE_MODE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --dtype "${DTYPE}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --limit_mm_per_prompt "${LIMIT_MM_PER_PROMPT}" \
    "${EXTRA_ARGS[@]}"
done

echo "All task smoke tests finished."
echo "Reports:"
ls -lh "${OUT_DIR}"
