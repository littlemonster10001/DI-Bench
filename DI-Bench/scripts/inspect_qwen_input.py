#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image
from transformers import AutoProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark import resolve_scene_dir
from benchmark.dataset import load_scene_questions
from utils import BenchmarkPrompter, set_project_cache_root


DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect the exact Qwen-VL prompt and rendered images before model inference."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="Local Qwen2.5/3 VL model directory.")
    parser.add_argument("--data_path", default=os.environ.get("DATA_PATH", "data/DI-Bench"))
    parser.add_argument("--scene", default="scene_001")
    parser.add_argument("--task_type", default="all", help="Task filter. Use all to keep all tasks.")
    parser.add_argument("--question_id", default=None, help="Inspect a specific question_id, e.g. 101.")
    parser.add_argument("--index", type=int, default=0, help="Index within the selected task/filter.")
    parser.add_argument("--bbox_mode", default="visual", choices=["visual", "raw", "text"])
    parser.add_argument("--source_mode", default="full", choices=["full", "none", "rgb_only"])
    parser.add_argument("--out_dir", default="outputs/qwen_input_inspect")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--skip_processor", action="store_true", help="Only dump messages/images, skip Qwen chat template.")
    return parser.parse_args()


def strip_file_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return parsed.path
    return uri


def select_item(scene_dir: Path, task_type: str, question_id: str | None, index: int):
    scene_info, items = load_scene_questions(scene_dir, task_type=task_type)
    if question_id is not None:
        matched = [item for item in items if str(item.get("question_id")) == str(question_id)]
        if not matched:
            raise ValueError(f"question_id={question_id} not found under {scene_dir}")
        return scene_info, matched[0], items

    if not items:
        raise ValueError(f"No questions matched task_type={task_type!r} under {scene_dir}")
    safe_index = min(max(index, 0), len(items) - 1)
    return scene_info, items[safe_index], items


def normalize_messages_and_copy_images(messages, output_dir: Path):
    normalized_messages = []
    image_records = []
    image_inputs = []
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for message in messages:
        content_blocks = message.get("content", [])
        if isinstance(content_blocks, str):
            normalized_messages.append({"role": message["role"], "content": content_blocks})
            continue

        normalized_content = []
        for block in content_blocks:
            block_type = block.get("type")
            if block_type == "image":
                src_path = Path(strip_file_uri(block["image"]))
                dst_path = images_dir / f"image_{len(image_records) + 1:02d}{src_path.suffix or '.jpg'}"
                shutil.copy2(src_path, dst_path)

                with Image.open(src_path) as image:
                    image_rgb = image.convert("RGB")
                    image_inputs.append(image_rgb.copy())
                    width, height = image_rgb.size

                image_records.append(
                    {
                        "source": str(src_path),
                        "saved": str(dst_path),
                        "width": width,
                        "height": height,
                    }
                )
                normalized_content.append({"type": "image"})
            elif block_type == "text":
                normalized_content.append({"type": "text", "text": block.get("text", "")})
            else:
                raise ValueError(f"Unsupported content block type: {block_type}")

        normalized_messages.append({"role": message["role"], "content": normalized_content})

    return normalized_messages, image_inputs, image_records


def extract_user_prompt(messages):
    parts = []
    for message in messages:
        if message.get("role") != "user":
            continue
        for block in message.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
    return "\n\n".join(part for part in parts if part).strip()


def write_preview_markdown(output_dir: Path, item: dict, user_prompt: str, image_records: list[dict]):
    lines = [
        f"# Qwen Input Inspection: {item.get('scene_id')} / {item.get('question_id')}",
        "",
        f"- Task type: `{item.get('task_type')}`",
        f"- Question type: `{item.get('question_type')}`",
        f"- Image count: `{len(image_records)}`",
        "",
        "## Images",
        "",
    ]
    for image in image_records:
        image_name = Path(image["saved"]).name
        lines.append(f"- `{image_name}` ({image['width']} x {image['height']})")
        lines.append(f"![{image_name}](images/{image_name})")
        lines.append("")

    lines.extend(["## User Prompt", "", "```text", user_prompt, "```", ""])
    (output_dir / "input_preview.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    scene_dir = resolve_scene_dir(args.data_path, args.scene)
    scene_info, item, _ = select_item(scene_dir, args.task_type, args.question_id, args.index)

    output_dir = Path(args.out_dir) / scene_dir.name / str(item.get("question_id"))
    output_dir.mkdir(parents=True, exist_ok=True)
    set_project_cache_root(str(output_dir / "cache"))

    prompter = BenchmarkPrompter(bbox_mode=args.bbox_mode, source_mode=args.source_mode)
    messages = prompter.get_messages(str(scene_dir), item)
    normalized_messages, image_inputs, image_records = normalize_messages_and_copy_images(messages, output_dir)
    user_prompt = extract_user_prompt(messages)

    chat_prompt = None
    if not args.skip_processor:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
        chat_prompt = processor.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        (output_dir / "qwen_chat_prompt.txt").write_text(chat_prompt, encoding="utf-8")

    (output_dir / "messages.json").write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "normalized_messages.json").write_text(
        json.dumps(normalized_messages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "user_prompt.txt").write_text(user_prompt, encoding="utf-8")
    (output_dir / "item.json").write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "scene_info.json").write_text(json.dumps(scene_info, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_path": args.model_path,
                "scene_dir": str(scene_dir),
                "scene": scene_dir.name,
                "question_id": item.get("question_id"),
                "task_type": item.get("task_type"),
                "bbox_mode": args.bbox_mode,
                "source_mode": args.source_mode,
                "image_count": len(image_records),
                "images": image_records,
                "chat_prompt_chars": len(chat_prompt) if chat_prompt is not None else None,
                "user_prompt_chars": len(user_prompt),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_preview_markdown(output_dir, item, user_prompt, image_records)

    print(f"Saved inspection output to: {output_dir}")
    print(f"Question: {scene_dir.name}/{item.get('question_id')} task={item.get('task_type')}")
    print(f"Images: {len(image_records)}")
    print(f"User prompt: {len(user_prompt)} chars")
    if chat_prompt is not None:
        print(f"Qwen chat prompt: {len(chat_prompt)} chars")


if __name__ == "__main__":
    main()
