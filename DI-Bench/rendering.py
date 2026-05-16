from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def open_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


@lru_cache(maxsize=1)
def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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


def _draw_badge(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, fill: str):
    font = load_font(28)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width = right - left
    height = bottom - top
    pad_x = 14
    pad_y = 10
    rect = [x, y, x + width + pad_x * 2, y + height + pad_y * 2]
    draw.rounded_rectangle(rect, radius=12, fill=fill)
    draw.text((x + pad_x - left, y + pad_y - top), text, font=font, fill="white")


def add_corner_label(image: Image.Image, label: str, fill: str = "#1565c0") -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    _draw_badge(draw, 16, 16, label, fill)
    return canvas


def draw_bboxes(image: Image.Image, labeled_bboxes: list[tuple[str, list[int]]], color: str = "#e53935") -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = load_font(24)
    width = max(3, min(10, min(canvas.size) // 300))
    for label, bbox in labeled_bboxes:
        xmin, ymin, xmax, ymax = [int(v) for v in bbox]
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=width)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w = right - left
        text_h = bottom - top
        rect = [xmin, max(0, ymin - text_h - 18), xmin + text_w + 18, max(0, ymin - 4)]
        draw.rounded_rectangle(rect, radius=8, fill=color)
        draw.text((rect[0] + 9 - left, rect[1] + 6 - top), label, font=font, fill="white")
    return canvas


def draw_point(image: Image.Image, xy: list[int] | tuple[int, int], label: str = "P", color: str = "#e53935") -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    x, y = int(xy[0]), int(xy[1])
    radius = max(10, min(canvas.size) // 90)
    width = max(3, min(8, min(canvas.size) // 350))
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, width=width, fill="white")
    draw.line([x - radius * 2, y, x + radius * 2, y], fill=color, width=width)
    draw.line([x, y - radius * 2, x, y + radius * 2], fill=color, width=width)
    _draw_badge(draw, min(x + radius + 8, canvas.size[0] - 80), max(8, y - radius - 18), label, color)
    return canvas


def draw_polygons(image: Image.Image, polygons: list[list[list[float]]], color: str = "#e53935") -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width = max(3, min(10, min(canvas.size) // 300))
    for polygon in polygons:
        points = [(int(round(x)), int(round(y))) for x, y in polygon]
        if len(points) >= 2:
            draw.polygon(points, outline=color, width=width)
    return canvas


def draw_routes(image: Image.Image, routes: list[dict]) -> Image.Image:
    palette = ["#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa", "#00acc1"]
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = load_font(24)
    width = max(3, min(12, min(canvas.size) // 300))
    for idx, route in enumerate(routes):
        points = route.get("pixel_coordinates", [])
        if len(points) < 2:
            continue
        label = route.get("label", chr(ord("A") + idx))
        color = palette[idx % len(palette)]
        path = [(int(round(x)), int(round(y))) for x, y in points]
        draw.line(path, fill=color, width=width, joint="curve")
        label_index = min(len(path) - 1, max(1, len(path) // 3))
        start_x, start_y = path[label_index]
        start_x = max(0, min(canvas.size[0] - 48, start_x + (idx % 3) * 12))
        start_y = max(0, min(canvas.size[1] - 36, start_y + (idx // 3) * 12))
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        draw.rounded_rectangle(
            [start_x, start_y, start_x + (right - left) + 18, start_y + (bottom - top) + 14],
            radius=8,
            fill="white",
            outline=color,
            width=2,
        )
        draw.text((start_x + 9 - left, start_y + 7 - top), label, font=font, fill=color)
    return canvas


def crop_bbox_with_margin(image: Image.Image, bbox: list[int], margin_ratio: float = 0.08) -> Image.Image:
    width, height = image.size
    xmin, ymin, xmax, ymax = [int(v) for v in bbox]
    margin_x = int((xmax - xmin) * margin_ratio)
    margin_y = int((ymax - ymin) * margin_ratio)
    left = max(0, xmin - margin_x)
    top = max(0, ymin - margin_y)
    right = min(width, xmax + margin_x)
    bottom = min(height, ymax + margin_y)
    return image.crop((left, top, right, bottom)).convert("RGB")
