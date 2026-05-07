#!/usr/bin/env python3
import argparse
import os
import random

from benchmark import BenchmarkEvaluator, BenchmarkRunner, resolve_scene_dir
from model_adapters import InternVLLocalAdapter, QwenVLLMAdapter

from utils import set_project_cache_root


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight local DI-Bench entrypoint for Qwen-VL and InternVL.")
    parser.add_argument("--backend", default="qwen", choices=["qwen", "internvl"])
    parser.add_argument("--model_path", required=True, help="Local model path.")
    parser.add_argument("--data_path", required=True, help="Either the dataset root or a specific scene directory.")
    parser.add_argument("--scene", default=None, help="Scene name such as scene_001 when --data_path is the dataset root.")
    parser.add_argument("--report_path", default=None, help="Output .xlsx path.")
    parser.add_argument("--output_path", default=None, help="Compatibility alias for --report_path.")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--task_type", default="all", help="Task filter. Default: all.")
    parser.add_argument("--bbox_mode", default="visual", choices=["visual", "raw"])
    parser.add_argument("--source_mode", default="full", choices=["full", "none"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_num_seqs", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--disable_mm_preprocessor_cache", action="store_true")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--limit_mm_per_prompt", type=int, default=None)
    parser.add_argument("--few_shot_k", type=int, default=0)
    parser.add_argument("--few_shot_scope", default="same_task")
    parser.add_argument("--few_shot_selection", default="first")
    parser.add_argument("--save_details", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--max_tiles_per_image", type=int, default=12)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def main():
    args = parse_args()
    report_path = args.report_path or args.output_path
    if not report_path:
        raise ValueError("One of --report_path or --output_path is required.")

    set_seed(args.seed)
    if args.cache_dir:
        set_project_cache_root(os.path.abspath(args.cache_dir))
    os.environ["DI_BENCH_SOURCE_MODE"] = args.source_mode
    scene_dir = resolve_scene_dir(args.data_path, args.scene)

    if args.backend == "qwen":
        adapter = QwenVLLMAdapter(
            model_path=args.model_path,
            bbox_mode=args.bbox_mode,
            source_mode=args.source_mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            seed=args.seed,
            max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager,
            limit_mm_per_prompt=args.limit_mm_per_prompt,
            disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
        )
    else:
        adapter = InternVLLocalAdapter(
            model_path=args.model_path,
            bbox_mode=args.bbox_mode,
            source_mode=args.source_mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            dtype=args.dtype,
            device="cpu" if str(args.device).lower() == "cpu" else "cuda",
            image_size=args.image_size,
            max_tiles_per_image=args.max_tiles_per_image,
        )
    runner = BenchmarkRunner(
        model_adapter=adapter,
        evaluator=BenchmarkEvaluator(),
        few_shot_k=args.few_shot_k,
        few_shot_scope=args.few_shot_scope,
        few_shot_selection=args.few_shot_selection,
    )
    scene_info, _, summary = runner.run(
        scene_dir=scene_dir,
        output_path=report_path,
        task_type=args.task_type,
        limit=args.limit,
    )
    print(f"Saved report to {report_path}")
    print(f"Scene: {scene_info.get('scene_id')} | Accuracy: {summary['overall_accuracy']:.4f}")


if __name__ == "__main__":
    main()
