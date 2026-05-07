from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from PIL import Image

from rendering import add_corner_label, crop_bbox_with_margin, draw_bboxes, draw_point, draw_polygons, draw_routes, open_rgb_image


SYSTEM_PROMPT = (
    "You are evaluating DI-Bench from images and structured scene context. "
    "Follow the question exactly. "
    "For single-choice questions, answer with only one option letter such as A. "
    "For multiple-choice or multiple-selection questions, answer with only the option letters in alphabetical order, "
    "separated by commas, such as A,C. Do not add explanation unless the prompt explicitly asks for it."
)


class QwenBenchmarkPrompter:
    def __init__(self, bbox_mode: str = "visual", source_mode: str = "full"):
        self.bbox_mode = bbox_mode
        self.source_mode = source_mode
        self._vector_cache: dict[Path, gpd.GeoDataFrame] = {}

    def build(self, scene_dir: Path, item: dict) -> dict:
        images, image_notes = self._build_images(scene_dir, item)
        prompt = self._build_prompt(item, image_notes)
        return {"system_prompt": SYSTEM_PROMPT, "prompt": prompt, "images": images, "image_notes": image_notes}

    def _build_prompt(self, item: dict, image_notes: list[str]) -> str:
        scene_info = item.get("scene_info", {})
        lines = []
        if self.source_mode == "full" and scene_info:
            lines.extend(
                [
                    "Scene context:",
                    f"- Scene ID: {scene_info.get('scene_id', '')}",
                    f"- Location: {scene_info.get('location', '')}",
                    f"- Date: {scene_info.get('timestamp', '')}",
                    f"- Disaster type: {scene_info.get('disaster_type', '')}",
                    f"- Description: {scene_info.get('description', '')}",
                    "",
                ]
            )

        if image_notes:
            lines.append("Images are provided in this order:")
            for idx, note in enumerate(image_notes, start=1):
                lines.append(f"{idx}. {note}")
            lines.append("")

        task_hint = self._task_hint(item)
        if task_hint:
            lines.append(task_hint)
            lines.append("")

        lines.append(f"Task type: {item.get('task_type')}")
        lines.append(f"Question: {str(item.get('instruction', '')).strip()}")
        lines.append("")
        lines.append("Options:")
        for key, value in item.get("options", {}).items():
            if isinstance(value, dict):
                display = value.get("label", key)
            else:
                display = value
            lines.append(f"{key}: {display}")
        lines.append("")

        qtype = str(item.get("question_type", "single_choice"))
        if qtype == "single_choice":
            lines.append("Return only one option letter.")
        elif qtype in {"multiple_choice", "multiple_selection"}:
            lines.append("Return only the option letters, in alphabetical order, separated by commas.")
        else:
            lines.append("Return a short answer only.")

        ref = item.get("ref", {})
        rois = ref.get("rois")
        if rois:
            lines.append(f"Named landing zones or regions of interest: {', '.join(str(x) for x in rois)}.")
        return "\n".join(lines).strip()

    def _task_hint(self, item: dict) -> str:
        task_type = str(item.get("task_type"))
        hints = {
            "Image_Retrieval": "Each candidate image is labeled with its option letter.",
            "Image_Level_Cross_View_Matching": "The first image is the aerial query. The remaining images are candidate matches labeled by option.",
            "Object_Level_Cross_View_Matching": "The first image is the ground query. The remaining images are cropped aerial candidate regions labeled by option.",
            "poi_alignment": "The target point is marked as P and candidate regions are labeled on the image.",
            "height_alignment": "The target point is marked as P on both the RGB image and the DSM image.",
            "height_comparison": "Regions A-D are marked on both the RGB image and the DSM image.",
            "building_damage_assessment": "The target building region is boxed on the paired pre-disaster and post-disaster images.",
            "Building_Damage_Counting": "The counting area is boxed on the paired pre-disaster and post-disaster images.",
            "population_estimation": "The target region is boxed on all aligned source images.",
            "Area_Estimation": "The target polygon is outlined on the aligned images.",
            "Length_Estimation": "The target polygon is outlined on the aligned images.",
            "Distance_Estimation": "Both target polygons are outlined on the aligned images.",
            "uav_landing_assessment": "Candidate regions are labeled on the images.",
            "route_planning": "Candidate routes are overlaid on the route image with different labels and colors.",
            "Road_Damage_Reasoning": "The target road is highlighted on the pre-disaster and post-disaster images.",
        }
        return hints.get(task_type, "")

    def _build_images(self, scene_dir: Path, item: dict) -> tuple[list[Image.Image], list[str]]:
        task_type = str(item.get("task_type"))
        builder = getattr(self, f"_build_{task_type}", None)
        if builder is not None:
            return builder(scene_dir, item)
        return self._build_default(scene_dir, item)

    def _load_rel_image(self, scene_dir: Path, rel_path: str) -> Image.Image:
        return open_rgb_image(scene_dir / rel_path)

    def _build_default(self, scene_dir: Path, item: dict):
        images = []
        notes = []
        for idx, rel_path in enumerate(item.get("ref", {}).get("images", []), start=1):
            image = self._load_rel_image(scene_dir, rel_path)
            images.append(add_corner_label(image, f"Image {idx}"))
            notes.append(f"Image {idx}")
        return images, notes

    def _build_Image_Retrieval(self, scene_dir: Path, item: dict):
        images = []
        notes = []
        image_paths = item.get("ref", {}).get("images", [])
        option_keys = list(item.get("options", {}).keys())
        for idx, rel_path in enumerate(image_paths):
            label = option_keys[idx] if idx < len(option_keys) else chr(ord("A") + idx)
            image = self._load_rel_image(scene_dir, rel_path)
            images.append(add_corner_label(image, label))
            notes.append(f"Option {label}")
        return images, notes

    def _build_Image_Level_Cross_View_Matching(self, scene_dir: Path, item: dict):
        ref = item.get("ref", {})
        query = add_corner_label(self._load_rel_image(scene_dir, ref["query_image"]), "Query", fill="#6a1b9a")
        images = [query]
        notes = ["Query aerial image"]
        option_keys = list(item.get("options", {}).keys())
        for idx, rel_path in enumerate(ref.get("candidate_images", [])):
            label = option_keys[idx] if idx < len(option_keys) else chr(ord("A") + idx)
            images.append(add_corner_label(self._load_rel_image(scene_dir, rel_path), label))
            notes.append(f"Candidate {label}")
        return images, notes

    def _build_Object_Level_Cross_View_Matching(self, scene_dir: Path, item: dict):
        ref = item.get("ref", {})
        images = [add_corner_label(self._load_rel_image(scene_dir, ref["query_image"]), "Query", fill="#6a1b9a")]
        notes = ["Ground-view query image"]
        for key, value in item.get("options", {}).items():
            full_image = self._load_rel_image(scene_dir, value["image_path"])
            crop = crop_bbox_with_margin(full_image, value["bbox"])
            images.append(add_corner_label(crop, key))
            notes.append(f"{key}: {value.get('label', key)}")
        return images, notes

    def _build_poi_alignment(self, scene_dir: Path, item: dict):
        ref = item.get("ref", {})
        image = self._load_rel_image(scene_dir, ref["images"][0])
        regions = [(region.get("label", "?"), region["bbox"]) for region in ref.get("regions", [])]
        image = draw_bboxes(image, regions)
        image = draw_point(image, ref["target_poi"]["pixel_coords"], label="P")
        return [add_corner_label(image, "Main")], ["POI alignment image"]

    def _build_population_estimation(self, scene_dir: Path, item: dict):
        ref = item.get("ref", {})
        labels = ["Pre", "Post", "Population"]
        bbox = ref.get("bbox")
        images = []
        notes = []
        for idx, rel_path in enumerate(ref.get("images", [])):
            image = self._load_rel_image(scene_dir, rel_path)
            if bbox and self.bbox_mode == "visual":
                image = draw_bboxes(image, [("Target", bbox)])
            label = labels[idx] if idx < len(labels) else f"Image {idx + 1}"
            images.append(add_corner_label(image, label))
            notes.append(label)
        return images, notes

    def _build_height_alignment(self, scene_dir: Path, item: dict):
        ref = item.get("ref", {})
        labels = ["RGB", "DSM"]
        images = []
        notes = []
        for idx, rel_path in enumerate(ref.get("images", [])):
            image = draw_point(self._load_rel_image(scene_dir, rel_path), ref["target_poi"]["pixel_coords"], label="P")
            label = labels[idx] if idx < len(labels) else f"Image {idx + 1}"
            images.append(add_corner_label(image, label))
            notes.append(label)
        return images, notes

    def _build_height_comparison(self, scene_dir: Path, item: dict):
        ref = item.get("ref", {})
        boxes = [(entry["id"], entry["bbox"]) for entry in ref.get("bboxes", [])]
        labels = ["RGB", "DSM"]
        images = []
        notes = []
        for idx, rel_path in enumerate(ref.get("images", [])):
            image = draw_bboxes(self._load_rel_image(scene_dir, rel_path), boxes)
            label = labels[idx] if idx < len(labels) else f"Image {idx + 1}"
            images.append(add_corner_label(image, label))
            notes.append(label)
        return images, notes

    def _build_building_damage_assessment(self, scene_dir: Path, item: dict):
        return self._build_bbox_pair(scene_dir, item, ["Pre", "Post"])

    def _build_Building_Damage_Counting(self, scene_dir: Path, item: dict):
        return self._build_bbox_pair(scene_dir, item, ["Pre", "Post"])

    def _build_bbox_pair(self, scene_dir: Path, item: dict, labels: list[str]):
        ref = item.get("ref", {})
        bbox = ref.get("bbox")
        images = []
        notes = []
        for idx, rel_path in enumerate(ref.get("images", [])):
            image = self._load_rel_image(scene_dir, rel_path)
            if bbox and self.bbox_mode == "visual":
                image = draw_bboxes(image, [("Target", bbox)])
            label = labels[idx] if idx < len(labels) else f"Image {idx + 1}"
            images.append(add_corner_label(image, label))
            notes.append(label)
        return images, notes

    def _build_Area_Estimation(self, scene_dir: Path, item: dict):
        return self._build_polygon_pair(scene_dir, item, ["Pre", "Post"])

    def _build_Length_Estimation(self, scene_dir: Path, item: dict):
        return self._build_polygon_pair(scene_dir, item, ["Pre", "Post"])

    def _build_Distance_Estimation(self, scene_dir: Path, item: dict):
        return self._build_polygon_pair(scene_dir, item, ["Post", "Pre"])

    def _build_polygon_pair(self, scene_dir: Path, item: dict, labels: list[str]):
        ref = item.get("ref", {})
        polygons = ref.get("bboxes", [])
        images = []
        notes = []
        for idx, rel_path in enumerate(ref.get("images", [])):
            image = self._load_rel_image(scene_dir, rel_path)
            if polygons and self.bbox_mode == "visual":
                image = draw_polygons(image, polygons)
            label = labels[idx] if idx < len(labels) else f"Image {idx + 1}"
            images.append(add_corner_label(image, label))
            notes.append(label)
        return images, notes

    def _build_uav_landing_assessment(self, scene_dir: Path, item: dict):
        ref = item.get("ref", {})
        regions = [(region.get("label", "?"), region["bbox"]) for region in ref.get("regions", [])]
        images = []
        notes = []
        for idx, rel_path in enumerate(ref.get("images", [])):
            image = self._load_rel_image(scene_dir, rel_path)
            if regions and self.bbox_mode == "visual":
                image = draw_bboxes(image, regions)
            label = "Pre" if idx == 0 else "Post"
            images.append(add_corner_label(image, label))
            notes.append(label)
        return images, notes

    def _build_route_planning(self, scene_dir: Path, item: dict):
        ref = item.get("ref", {})
        image_paths = ref.get("images", [])
        routes = item.get("context_paths", [])
        images = []
        notes = []
        for idx, rel_path in enumerate(image_paths):
            image = self._load_rel_image(scene_dir, rel_path)
            if idx == len(image_paths) - 1 and routes and self.bbox_mode == "visual":
                image = draw_routes(image, routes)
                label = "Post + Routes"
            else:
                label = "Pre" if idx == 0 else "Post"
            images.append(add_corner_label(image, label))
            notes.append(label)
        return images, notes

    def _build_Road_Damage_Reasoning(self, scene_dir: Path, item: dict):
        ref = item.get("ref", {})
        road_pixels = self._target_road_pixels(scene_dir, item)
        images = []
        notes = []
        for idx, rel_path in enumerate(ref.get("images", [])):
            image = self._load_rel_image(scene_dir, rel_path)
            if road_pixels and self.bbox_mode == "visual":
                image = draw_routes(image, [{"label": "Road", "pixel_coordinates": road_pixels}])
            label = "Pre" if idx == 0 else "Post"
            images.append(add_corner_label(image, label))
            notes.append(label)
        return images, notes

    def _target_road_pixels(self, scene_dir: Path, item: dict):
        ref = item.get("ref", {})
        road_info = ref.get("road", {})
        gpkg_rel = road_info.get("gpkg_path")
        target_road_id = str(ref.get("target_road_id", ""))
        if not gpkg_rel or not target_road_id:
            return []

        gpkg_path = scene_dir / gpkg_rel
        gdf = self._read_vector(gpkg_path)
        if gdf.empty:
            return []

        matched = gdf[gdf["osm_id"].astype(str) == target_road_id]
        if matched.empty:
            return []

        row = matched.iloc[0]
        width, height = self._load_rel_image(scene_dir, ref["images"][-1]).size
        min_x = float(row["patch_min_x"])
        min_y = float(row["patch_min_y"])
        max_x = float(row["patch_max_x"])
        max_y = float(row["patch_max_y"])
        if max_x == min_x or max_y == min_y:
            return []

        geometry = row.geometry
        coords = []
        for part in getattr(geometry, "geoms", [geometry]):
            for x, y in part.coords:
                px = (float(x) - min_x) / (max_x - min_x) * width
                py = (max_y - float(y)) / (max_y - min_y) * height
                coords.append([px, py])
        return coords

    def _read_vector(self, path: Path):
        if path not in self._vector_cache:
            self._vector_cache[path] = gpd.read_file(path)
        return self._vector_cache[path]
