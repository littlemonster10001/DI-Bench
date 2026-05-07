from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from prompting import QwenBenchmarkPrompter, SYSTEM_PROMPT


_DEFAULT_CACHE_ROOT = Path(__file__).resolve().parent / ".cache"
_PROJECT_CACHE_ROOT = os.path.abspath(os.environ.get("DI_BENCH_CACHE_ROOT", str(_DEFAULT_CACHE_ROOT)))
_LAST_RUN_CACHE_DIR: str | None = None


def set_project_cache_root(cache_root: str):
    global _PROJECT_CACHE_ROOT
    _PROJECT_CACHE_ROOT = os.path.abspath(cache_root)
    os.environ["DI_BENCH_CACHE_ROOT"] = _PROJECT_CACHE_ROOT
    os.makedirs(_PROJECT_CACHE_ROOT, exist_ok=True)


def get_project_cache_dir() -> str:
    os.makedirs(_PROJECT_CACHE_ROOT, exist_ok=True)
    return _PROJECT_CACHE_ROOT


def create_run_cache_dir(prefix: str = "run") -> str:
    global _LAST_RUN_CACHE_DIR
    root = Path(get_project_cache_dir())
    run_dir = root / f"{prefix}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _LAST_RUN_CACHE_DIR = str(run_dir)
    return str(run_dir)


def clear_project_cache():
    global _LAST_RUN_CACHE_DIR
    if _LAST_RUN_CACHE_DIR:
        shutil.rmtree(_LAST_RUN_CACHE_DIR, ignore_errors=True)
        _LAST_RUN_CACHE_DIR = None


class BenchmarkPrompter:
    def __init__(self, system_prompt: str | None = None, bbox_mode: str = "visual", source_mode: str = "full"):
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.prompter = QwenBenchmarkPrompter(bbox_mode=bbox_mode, source_mode=source_mode)

    def get_messages(self, data_dir: str, item: dict[str, Any]):
        scene_dir = Path(data_dir)
        bundle = self.prompter.build(scene_dir, item)
        scene_id = str(item.get("scene_id") or bundle.get("scene_info", {}).get("scene_id") or scene_dir.name)
        question_id = str(item.get("question_id", "unknown"))
        cache_dir = Path(get_project_cache_dir()) / "prompt_images" / scene_id / question_id
        cache_dir.mkdir(parents=True, exist_ok=True)

        content = []
        for idx, image in enumerate(bundle["images"], start=1):
            image_path = cache_dir / f"image_{idx:02d}.jpg"
            image.convert("RGB").save(image_path, format="JPEG", quality=95)
            content.append({"type": "image", "image": f"file://{image_path.resolve()}"})

        content.append({"type": "text", "text": bundle["prompt"]})
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]
