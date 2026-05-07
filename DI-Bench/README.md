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
├── scripts/                  # batch run scripts
├── evaluate.py               # main entry
├── prompting.py              # local prompt and image construction
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
/path/to/Di-Bench
```

## Qwen-VL Usage

Run one scene:

```bash
conda activate di_bench

python evaluate.py \
  --backend qwen \
  --model_path /path/to/Qwen3-VL-2B-Instruct \
  --data_path /path/to/Di-Bench \
  --scene scene_001 \
  --report_path /path/to/outputs/di_bench_qwen_scene_001.xlsx \
  --task_type all \
  --bbox_mode visual \
  --source_mode full \
  --tensor_parallel_size 1 \
  --dtype bfloat16 \
  --limit_mm_per_prompt 16
```

Run all scenes:

```bash
conda activate di_bench

export MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct
bash scripts/run_qwen_all_scenes.sh
```

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
conda activate di_bench

python evaluate.py \
  --backend internvl \
  --model_path /path/to/InternVL3-2B \
  --data_path /path/to/Di-Bench \
  --scene scene_001 \
  --report_path /path/to/outputs/di_bench_internvl3_scene_001.xlsx \
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
conda activate di_bench

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
- `--limit_mm_per_prompt`
- `--enforce_eager`

InternVL-specific:

- `--image_size`
- `--max_tiles_per_image`
- `--device`

## Smoke Test

For a minimal smoke test, add:

```bash
--limit 1
```

That is the fastest way to verify that the model path, dataset path, and runtime environment are correct before launching all scenes.
