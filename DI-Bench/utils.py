from __future__ import annotations

import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


_DEFAULT_CACHE_ROOT = Path(__file__).resolve().parent / ".cache"
_PROJECT_CACHE_ROOT = os.path.abspath(os.environ.get("DI_BENCH_CACHE_ROOT", str(_DEFAULT_CACHE_ROOT)))
_LAST_RUN_CACHE_DIR: str | None = None
_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))


DEFAULT_SYSTEM_PROMPT = (
    "You are an intelligent assistant for disaster scenarios. "
    "Please answer the questions accurately based on the provided scenario information and image references."
)


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


def _load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf",
    ]
    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return ImageFont.truetype(str(candidate_path), size)
    return ImageFont.load_default()


def _save_drawn_image(image: Image.Image, image_path: str, output_dir: str, prefix: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    source = Path(image_path)
    output_path = Path(output_dir) / f"{prefix}_{source.stem}.jpg"
    image.convert("RGB").save(output_path, format="JPEG", quality=95)
    return str(output_path)


def draw_bboxes_on_image(image_path: str, options_on_this_image: list[tuple], output_dir: str) -> str:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font(max(20, min(image.size) // 160))
    line_width = max(3, min(18, min(image.size) // 250))

    for label, bbox in options_on_this_image:
        if not bbox or len(bbox) != 4:
            continue
        xmin, ymin, xmax, ymax = [int(round(value)) for value in bbox]
        draw.rectangle([xmin, ymin, xmax, ymax], outline="#e53935", width=line_width)

        text = str(label).replace("Option", "").strip()
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w = right - left
        text_h = bottom - top
        pad_x = max(8, line_width * 2)
        pad_y = max(6, line_width)
        box_y = max(0, ymin - text_h - pad_y * 2 - line_width)
        draw.rounded_rectangle(
            [xmin, box_y, xmin + text_w + pad_x * 2, box_y + text_h + pad_y * 2],
            radius=8,
            fill="#e53935",
        )
        draw.text((xmin + pad_x - left, box_y + pad_y - top), text, fill="white", font=font)

    return _save_drawn_image(image, image_path, output_dir, "viz_bbox")


def draw_polygons_on_image(
    image_path: str,
    polygons: list[list[list[float]]],
    output_dir: str,
    outline_color: str = "red",
    width_ratio: float = 0.005,
) -> str:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(3, min(18, int(min(image.size) * width_ratio)))

    for polygon in polygons or []:
        points = [(int(round(x)), int(round(y))) for x, y in polygon]
        if len(points) >= 2:
            draw.polygon(points, outline=outline_color, width=line_width)

    return _save_drawn_image(image, image_path, output_dir, "viz_poly")


def draw_routes_on_image(image_path: str, routes: list[dict[str, Any]], output_dir: str, width_ratio: float = 0.005) -> str:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font(max(18, min(image.size) // 170))
    line_width = max(3, min(18, int(min(image.size) * width_ratio)))
    palette = ["#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa", "#00acc1"]

    for idx, route in enumerate(routes or []):
        coords = route.get("pixel_coordinates", [])
        if len(coords) < 2:
            continue
        label = str(route.get("label", chr(ord("A") + idx)))
        color = palette[idx % len(palette)]
        points = [(int(round(x)), int(round(y))) for x, y in coords]
        draw.line(points, fill=color, width=line_width, joint="curve")

        label_index = min(len(points) - 1, max(1, len(points) // 3))
        label_x, label_y = points[label_index]
        label_x = max(0, min(image.size[0] - 56, label_x + (idx % 3) * 12))
        label_y = max(0, min(image.size[1] - 42, label_y + (idx // 3) * 12))
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w = right - left
        text_h = bottom - top
        draw.rounded_rectangle(
            [label_x, label_y, label_x + text_w + 18, label_y + text_h + 14],
            radius=8,
            fill="white",
            outline=color,
            width=2,
        )
        draw.text((label_x + 9 - left, label_y + 7 - top), label, font=font, fill=color)

    return _save_drawn_image(image, image_path, output_dir, "viz_routes")


def get_image_size(image_path: str, default: tuple[int, int] = (1024, 1024)) -> tuple[int, int]:
    try:
        with Image.open(image_path) as image:
            return image.size
    except Exception:
        return default


def _norm_value(value: float, max_value: int) -> float:
    denom = max(1, max_value - 1)
    return max(0.0, min(1.0, float(value) / denom))


def normalize_point(point: list[float] | tuple[float, float], image_size: tuple[int, int]) -> list[float]:
    width, height = image_size
    if len(point) < 2:
        return []
    return [_norm_value(point[0], width), _norm_value(point[1], height)]


def normalize_bbox(bbox: list[float] | tuple[float, float, float, float], image_size: tuple[int, int]) -> list[float]:
    width, height = image_size
    if len(bbox) != 4:
        return []
    return [
        _norm_value(bbox[0], width),
        _norm_value(bbox[1], height),
        _norm_value(bbox[2], width),
        _norm_value(bbox[3], height),
    ]


def normalize_points(points: list[list[float]], image_size: tuple[int, int]) -> list[list[float]]:
    return [normalized for point in points or [] if (normalized := normalize_point(point, image_size))]


def format_normalized_point(point: list[float] | tuple[float, float], image_size: tuple[int, int], precision: int = 4) -> str:
    normalized = normalize_point(point, image_size)
    if not normalized:
        return "[]"
    return "[" + ", ".join(f"{value:.{precision}f}" for value in normalized) + "]"


def format_normalized_bbox(bbox: list[float] | tuple[float, float, float, float], image_size: tuple[int, int], precision: int = 4) -> str:
    normalized = normalize_bbox(bbox, image_size)
    if not normalized:
        return "[]"
    return "[" + ", ".join(f"{value:.{precision}f}" for value in normalized) + "]"


def format_normalized_points(points: list[list[float]], image_size: tuple[int, int], precision: int = 4) -> str:
    normalized = normalize_points(points, image_size)
    return " -> ".join(
        "(" + ", ".join(f"{value:.{precision}f}" for value in point) + ")"
        for point in normalized
    )


def sanitize_coordinate_text(text: str) -> str:
    if not text:
        return text
    sanitized = re.sub(
        r"pixel coordinates\s*\[[^\]]+\]",
        "the normalized coordinates specified above",
        text,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(
        r"BBox\s*\[[^\]]+\]",
        "the normalized BBox specified above",
        sanitized,
        flags=re.IGNORECASE,
    )
    return sanitized


class BenchmarkPrompter:
    def __init__(self, system_prompt: str | None = None, bbox_mode: str = "visual", source_mode: str = "full"):
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.bbox_mode = bbox_mode
        self.source_mode = source_mode

    def get_messages(self, data_dir: str, item: dict[str, Any]):
        from task_prompters import get_prompter, get_prompter_v2

        os.environ["DI_BENCH_SOURCE_MODE"] = self.source_mode
        task_type = self._resolve_task_type(item.get("task_type", "default"))
        item_for_prompt = dict(item)
        item_for_prompt["instruction"] = sanitize_coordinate_text(str(item.get("instruction", "")))
        try:
            prompter = get_prompter_v2(task_type)
        except ValueError:
            prompter = get_prompter(task_type)

        user_content = prompter.build_content(data_dir, item_for_prompt)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _resolve_task_type(self, task_type: str) -> str:
        if self.bbox_mode not in {"text", "raw"}:
            return task_type

        text_task_map = {
            "Object_Level_Cross_View_Matching": "Object_Level_Cross_View_Matching_Text",
            "building_damage_assessment": "building_damage_assessment_Text",
            "Building_Damage_Counting": "Building_Damage_Counting_Text",
            "poi_alignment": "poi_alignment_Text",
            "population_estimation": "population_estimation_Text",
            "height_comparison": "height_comparison_Text",
            "Area_Estimation": "Area_Estimation_Text",
            "Length_Estimation": "Length_Estimation_Text",
            "Distance_Estimation": "Distance_Estimation_Text",
            "route_planning": "route_planning_Text",
            "uav_landing_assessment": "uav_landing_assessment_Text",
        }
        return text_task_map.get(task_type, task_type)
