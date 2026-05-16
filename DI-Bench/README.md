# DI-Bench Local

This directory is a lightweight local evaluation package for DI-Bench.

It currently supports:

- `Qwen-VL` local inference through `vllm`
- `InternVL3` local inference
- `InternVL3.5` local inference

The intended use case is to evaluate one local model over the DI-Bench scenes and export per-scene `.xlsx` reports.

## Directory Layout

```text
DI-Bench/
├── benchmark/                # dataset loading, evaluation, xlsx export
├── model_adapters/           # local model backends
├── task_prompters/           # task-specific prompt and visual overlay builders
├── scripts/                  # batch run scripts
├── evaluate.py               # main entry
├── rendering.py              # overlay helpers
└── requirements.txt
```

## Installation

Use the existing `Di_bench` conda environment, or install the local package dependencies:

```bash
cd DI-Bench
pip install -r requirements.txt
```

Notes:

- `Qwen-VL` runs here depend on `vllm`.
- `InternVL3` and `InternVL3.5` runs here use Hugging Face `transformers` local loading, not `vllm`.
- `geopandas` is required because DI-Bench route and road tasks read local `.gpkg` files.

## Dataset Path

The examples below assume the dataset is located at:

```text
/path/to/DI-Bench-dataset
```

Set paths for your local environment:

```bash
export DATA_PATH=/path/to/DI-Bench-dataset
export MODEL_PATH=/path/to/local/model
export OUT_DIR=outputs
```

## Qwen-VL Usage

Run one scene:

```bash
python evaluate.py \
  --backend qwen \
  --model_path "${MODEL_PATH}" \
  --data_path "${DATA_PATH}" \
  --scene scene_001 \
  --report_path "${OUT_DIR}/di_bench_qwen_scene_001.xlsx" \
  --task_type all \
  --bbox_mode visual \
  --source_mode full \
  --tensor_parallel_size 1 \
  --dtype bfloat16 \
  --limit_mm_per_prompt 16
```

Run all scenes:

```bash
export DATA_PATH=/path/to/DI-Bench-dataset
export MODEL_PATH=/path/to/Qwen-VL
bash scripts/run_qwen_all_scenes.sh
```

Run a one-sample smoke test for every task type:

```bash
export DATA_PATH=/path/to/DI-Bench-dataset
export MODEL_PATH=/path/to/Qwen-VL
bash scripts/run_qwen_all_tasks_smoke.sh
```

For large Qwen-VL checkpoints, configure tensor parallelism and sequence length through environment variables:

```bash
export DATA_PATH=/path/to/DI-Bench-dataset
export MODEL_PATH=/path/to/Qwen-VL
export TENSOR_PARALLEL_SIZE=4
export MAX_MODEL_LEN=32768
bash scripts/run_qwen_all_tasks_smoke.sh
```

The Qwen scripts set `VLLM_WORKER_MULTIPROC_METHOD=spawn` by default to avoid CUDA re-initialization errors in vLLM worker subprocesses.

## InternVL3 / InternVL3.5 Usage

This package includes a local adapter for the following model families:

- `InternVL3-2B`
- `InternVL3-8B`
- `InternVL3.5-4B`
- `InternVL3.5-8B`

All of them use the same backend:

```text
--backend internvl
```

Run one scene:

```bash
python evaluate.py \
  --backend internvl \
  --model_path "${MODEL_PATH}" \
  --data_path "${DATA_PATH}" \
  --scene scene_001 \
  --report_path "${OUT_DIR}/di_bench_internvl3_scene_001.xlsx" \
  --task_type all \
  --bbox_mode visual \
  --source_mode full \
  --dtype bfloat16 \
  --device cuda \
  --image_size 448 \
  --max_tiles_per_image 12
```

Replace `--model_path` with any local `InternVL3` or `InternVL3.5` checkpoint path, for example:

```text
/path/to/InternVL3-8B
/path/to/InternVL3_5-4B
/path/to/InternVL3_5-8B
```

Run all scenes:

```bash
export DATA_PATH=/path/to/DI-Bench-dataset
export MODEL_PATH=/path/to/InternVL3_5-4B
bash scripts/run_internvl_all_scenes.sh
```

## Main Arguments

Shared:

- `--model_path`: local checkpoint directory
- `--data_path`: dataset root or one scene directory
- `--scene`: one scene name when `data_path` is the dataset root
- `--report_path`: output `.xlsx`
- `--task_type`: task filter, default `all`
- `--bbox_mode`: `visual` or `raw`
- `--source_mode`: default `full`
- `--limit`: optional sample cap for smoke tests

Qwen-specific:

- `--tensor_parallel_size`
- `--gpu_memory_utilization`
- `--max_num_seqs`
- `--max_model_len`
- `--limit_mm_per_prompt`
- `--enforce_eager`

InternVL-specific:

- `--image_size`
- `--max_tiles_per_image`
- `--device`

## Smoke Test

For a minimal single-task smoke test, add:

```bash
--limit 1
```

To check all DI-Bench task prompters and the Qwen inference path, use:

```bash
export DATA_PATH=/path/to/DI-Bench-dataset
export MODEL_PATH=/path/to/Qwen-VL
bash scripts/run_qwen_all_tasks_smoke.sh
```

Common environment variables supported by the run scripts:

- `DATA_PATH`: dataset root containing `scene_*` directories
- `MODEL_PATH`: local checkpoint directory
- `OUT_DIR`: output report directory
- `BBOX_MODE`: `visual` or `raw`
- `SOURCE_MODE`: `full` or `none`
- `MAX_NEW_TOKENS`: generation length
- `TENSOR_PARALLEL_SIZE`: vLLM tensor parallel size for Qwen
- `GPU_MEMORY_UTILIZATION`: vLLM GPU memory fraction for Qwen
- `MAX_MODEL_LEN`: optional Qwen maximum context length override
- `LIMIT`: optional per-scene sample cap for all-scene scripts
