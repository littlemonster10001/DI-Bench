#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark import resolve_scene_dir
from benchmark.dataset import load_scene_questions
from utils import BenchmarkPrompter, set_project_cache_root


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect DI-Bench prompter outputs for every task type.")
    parser.add_argument("--data_path", default=os.environ.get("DATA_PATH", "data/DI-Bench"))
    parser.add_argument("--scene", default="scene_001", help="Scene name, or all for all scenes.")
    parser.add_argument("--task_type", nargs="+", default=["all"], help="Task filters, or all.")
    parser.add_argument("--samples_per_task", type=int, default=1)
    parser.add_argument("--bbox_mode", default="visual", choices=["visual", "raw", "text"])
    parser.add_argument("--source_mode", default="full", choices=["full", "none", "rgb_only"])
    parser.add_argument("--out_dir", default="outputs/prompter_inspect_all")
    parser.add_argument("--max_prompt_chars", type=int, default=20000)
    return parser.parse_args()


def strip_file_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return parsed.path
    return uri


def scene_dirs(data_path: str, scene: str):
    data_root = Path(data_path)
    if scene == "all":
        return sorted(path for path in data_root.glob("scene_*") if (path / "questions.json").is_file())
    return [resolve_scene_dir(data_path, scene)]


def selected_tasks(task_filters: list[str], items: list[dict]):
    if len(task_filters) == 1 and task_filters[0].lower() == "all":
        return sorted({str(item.get("task_type", "")) for item in items})
    return task_filters


def sample_items_for_task(items: list[dict], task_type: str, limit: int):
    task_items = [item for item in items if str(item.get("task_type")) == task_type]
    return task_items[: max(1, limit)]


def split_content(messages):
    prompt_parts = []
    image_paths = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            prompt_parts.append(content)
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                prompt_parts.append(block.get("text", ""))
            elif block.get("type") == "image":
                image_paths.append(strip_file_uri(block.get("image", "")))
    return "\n\n".join(part for part in prompt_parts if part).strip(), image_paths


def copy_images(image_paths: list[str], sample_dir: Path):
    images_dir = sample_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, image_path in enumerate(image_paths, start=1):
        src = Path(image_path)
        dst = images_dir / f"image_{idx:02d}{src.suffix or '.jpg'}"
        shutil.copy2(src, dst)
        width = height = None
        try:
            with Image.open(src) as image:
                width, height = image.size
        except Exception:
            pass
        records.append({"source": str(src), "saved": str(dst), "width": width, "height": height})
    return records


def write_preview(sample_dir: Path, item: dict, prompt_text: str, images: list[dict], max_prompt_chars: int):
    display_prompt = prompt_text
    truncated = False
    if len(display_prompt) > max_prompt_chars:
        display_prompt = display_prompt[:max_prompt_chars]
        truncated = True

    lines = [
        f"# {item.get('scene_id')} / {item.get('question_id')} / {item.get('task_type')}",
        "",
        f"- Question type: `{item.get('question_type')}`",
        f"- Image count: `{len(images)}`",
        f"- Prompt chars: `{len(prompt_text)}`",
        "",
        "## Images",
        "",
    ]
    for image in images:
        image_name = Path(image["saved"]).name
        lines.append(f"- `{image_name}` ({image.get('width')} x {image.get('height')})")
        lines.append(f"![{image_name}](images/{image_name})")
        lines.append("")

    lines.extend(["## Prompt", "", "```text", display_prompt])
    if truncated:
        lines.append(f"\n...[truncated, total_chars={len(prompt_text)}]")
    lines.extend(["```", ""])
    (sample_dir / "input_preview.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    output_root = Path(args.out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    set_project_cache_root(str(output_root / "_cache"))

    prompter = BenchmarkPrompter(bbox_mode=args.bbox_mode, source_mode=args.source_mode)
    rows = []
    by_task = defaultdict(lambda: {"ok": 0, "failed": 0})

    for scene_dir in scene_dirs(args.data_path, args.scene):
        scene_info, items = load_scene_questions(scene_dir, task_type="all")
        tasks = selected_tasks(args.task_type, items)
        for task_type in tasks:
            samples = sample_items_for_task(items, task_type, args.samples_per_task)
            if not samples:
                rows.append(
                    {
                        "scene": scene_dir.name,
                        "task_type": task_type,
                        "question_id": "",
                        "status": "missing",
                        "image_count": 0,
                        "prompt_chars": 0,
                        "sample_dir": "",
                        "error": "No sample for task in scene",
                    }
                )
                by_task[task_type]["failed"] += 1
                continue

            for sample_index, item in enumerate(samples):
                item = dict(item)
                sample_dir = output_root / scene_dir.name / task_type / str(item.get("question_id"))
                sample_dir.mkdir(parents=True, exist_ok=True)
                try:
                    messages = prompter.get_messages(str(scene_dir), item)
                    prompt_text, image_paths = split_content(messages)
                    image_records = copy_images(image_paths, sample_dir)

                    (sample_dir / "messages.json").write_text(
                        json.dumps(messages, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    (sample_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
                    (sample_dir / "item.json").write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
                    (sample_dir / "metadata.json").write_text(
                        json.dumps(
                            {
                                "scene": scene_dir.name,
                                "task_type": task_type,
                                "question_id": item.get("question_id"),
                                "sample_index": sample_index,
                                "bbox_mode": args.bbox_mode,
                                "source_mode": args.source_mode,
                                "image_count": len(image_records),
                                "prompt_chars": len(prompt_text),
                                "images": image_records,
                                "ground_truth": item.get("ground_truth"),
                                "options": item.get("options"),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    write_preview(sample_dir, item, prompt_text, image_records, args.max_prompt_chars)

                    rows.append(
                        {
                            "scene": scene_dir.name,
                            "task_type": task_type,
                            "question_id": item.get("question_id"),
                            "status": "ok",
                            "image_count": len(image_records),
                            "prompt_chars": len(prompt_text),
                            "sample_dir": str(sample_dir),
                            "error": "",
                        }
                    )
                    by_task[task_type]["ok"] += 1
                    print(f"OK {scene_dir.name} {task_type} {item.get('question_id')}: {sample_dir}")
                except Exception as exc:
                    error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                    (sample_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    rows.append(
                        {
                            "scene": scene_dir.name,
                            "task_type": task_type,
                            "question_id": item.get("question_id"),
                            "status": "failed",
                            "image_count": 0,
                            "prompt_chars": 0,
                            "sample_dir": str(sample_dir),
                            "error": error_text,
                        }
                    )
                    by_task[task_type]["failed"] += 1
                    print(f"FAILED {scene_dir.name} {task_type} {item.get('question_id')}: {error_text}")

    summary_path = output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene", "task_type", "question_id", "status", "image_count", "prompt_chars", "sample_dir", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    (output_root / "summary_by_task.json").write_text(json.dumps(by_task, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = [row for row in rows if row["status"] != "ok"]
    print(f"\nSaved inspection root: {output_root}")
    print(f"Summary CSV: {summary_path}")
    print(f"OK: {len(rows) - len(failures)} | Failed/missing: {len(failures)}")


if __name__ == "__main__":
    main()
