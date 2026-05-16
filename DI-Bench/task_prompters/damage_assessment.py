import os
from typing import Dict, Any, List
from PIL import Image
from .base_prompter import BaseTaskPrompter
from utils import (
    draw_bboxes_on_image,
    draw_routes_on_image,
    format_normalized_bbox,
    format_normalized_points,
    get_image_size,
)

class BuildingDamageAssessmentPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        bbox = ref.get('bbox', None)
        instruction = item.get('instruction', "")
        options = item.get('options', {})
        cache_dir = "./cache_v2"

        if bbox:
            os.makedirs(cache_dir, exist_ok=True)

        content = []

        # 1. 加载图像 (图1保持原样，图2画红框)
        for idx, path in enumerate(image_paths):
            full_path = os.path.join(data_dir, path)

            # idx == 1 表示这是 Post-disaster image
            if idx == 1 and bbox:
                opts_list = [("Target", bbox)]
                drawn_image_path = draw_bboxes_on_image(full_path, opts_list, cache_dir)
                
                content.append({
                    "type": "image",
                    "image": f"file://{os.path.abspath(drawn_image_path)}",
                })
            else:
                content.append({
                    "type": "image",
                    "image": f"file://{full_path}",
                })

        # 2. 构建 Prompt 文本
        context_info = "Visual Reference:\n"
        if len(image_paths) >= 2:
            context_info += "- Image 1: Pre-disaster Image (Reference)\n"
            if bbox:
                context_info += "- Image 2: Post-disaster Image with a Red Bounding Box marking the target area\n"
            else:
                context_info += "- Image 2: Post-disaster Image\n"
        else:
            context_info += "- Image 1: Post-disaster Image for analysis\n"

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: Building Damage Assessment\n\n"
            f"{context_info}\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Please strictly focus on the target area to answer the question. Output only the option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content


class BuildingDamageAssessmentTextPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        bbox = ref.get('bbox', None)
        instruction = item.get('instruction', "")
        options = item.get('options', {})

        content = []

        # 1. 直接加载原图，不画框
        for path in image_paths:
            full_path = os.path.join(data_dir, path)
            content.append({
                "type": "image",
                "image": f"file://{full_path}",
            })

        # 2. 构建 Prompt 文本，以文本形式注入 BBox
        context_info = "Visual Reference:\n"
        if len(image_paths) >= 2:
            context_info += "- Image 1: Pre-disaster Image (Reference)\n"
            context_info += "- Image 2: Post-disaster Image\n"
        else:
            context_info += "- Image 1: Post-disaster Image for analysis\n"

        if bbox:
            bbox_image_path = os.path.join(data_dir, image_paths[-1]) if image_paths else ""
            norm_bbox = format_normalized_bbox(bbox, get_image_size(bbox_image_path))
            context_info += f"\nNote: The target area is located at normalized BBox {norm_bbox} in [xmin, ymin, xmax, ymax] format.\n"

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: Building Damage Assessment\n\n"
            f"{context_info}\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Please strictly focus on the target area to answer the question. Output only the option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content


class BuildingDamageCountingPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        bbox = ref.get('bbox', None)
        instruction = item.get('instruction', "")
        options = item.get('options', {})
        cache_dir = "./cache_v2"

        if bbox:
            os.makedirs(cache_dir, exist_ok=True)

        content = []

        # 判断 bbox 是否接近全图 (例如 [0, 0, 1023, 1023])。如果是，其实不需要画框
        is_full_image = False
        if bbox and bbox[0] == 0 and bbox[1] == 0 and bbox[2] > 1000 and bbox[3] > 1000:
             is_full_image = True

        for idx, path in enumerate(image_paths):
            full_path = os.path.join(data_dir, path)

            if idx == 1 and bbox and not is_full_image:
                opts_list = [("Target", bbox)]
                drawn_image_path = draw_bboxes_on_image(full_path, opts_list, cache_dir)
                
                content.append({
                    "type": "image",
                    "image": f"file://{os.path.abspath(drawn_image_path)}",
                })
            else:
                content.append({
                    "type": "image",
                    "image": f"file://{full_path}",
                })

        context_info = "Visual Reference:\n"
        if len(image_paths) >= 2:
            context_info += "- Image 1: Pre-disaster Image (Reference)\n"
            if bbox and not is_full_image:
                context_info += "- Image 2: Post-disaster Image with a Red Bounding Box marking the counting area\n"
            else:
                context_info += "- Image 2: Post-disaster Image (Target Area: Full Image)\n"
        else:
            context_info += "- Image 1: Post-disaster Image for analysis\n"

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: Building Damage Counting\n\n"
            f"{context_info}\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Please carefully count within the specified area. Output only the option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content


class BuildingDamageCountingTextPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        bbox = ref.get('bbox', None)
        instruction = item.get('instruction', "")
        options = item.get('options', {})

        content = []

        for path in image_paths:
            full_path = os.path.join(data_dir, path)
            content.append({
                "type": "image",
                "image": f"file://{full_path}",
            })

        context_info = "Visual Reference:\n"
        if len(image_paths) >= 2:
            context_info += "- Image 1: Pre-disaster Image (Reference)\n"
            context_info += "- Image 2: Post-disaster Image\n"
        else:
            context_info += "- Image 1: Post-disaster Image for analysis\n"

        if bbox:
             bbox_image_path = os.path.join(data_dir, image_paths[-1]) if image_paths else ""
             norm_bbox = format_normalized_bbox(bbox, get_image_size(bbox_image_path))
             context_info += f"\nNote: The target counting area is bounded by normalized BBox {norm_bbox} in [xmin, ymin, xmax, ymax] format.\n"

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: Building Damage Counting\n\n"
            f"{context_info}\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Please carefully count within the specified area. Output only the option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content


class RoadDamageReasoningPrompter(BaseTaskPrompter):
    def __init__(self):
        # Cache for GeoDataFrame to avoid reloading the same file multiple times
        self.gpkg_cache = {}

    def get_road_route(self, data_dir: str, gpkg_rel_path: str, patch_id: str, target_osm_id: str, image_size=(1024, 1024)):
        try:
            import geopandas as gpd
        except ImportError:
            return [], "Error: geopandas is required for this task."

        full_gpkg_path = os.path.join(data_dir, gpkg_rel_path)

        if full_gpkg_path not in self.gpkg_cache:
            if not os.path.exists(full_gpkg_path):
                return [], "Error: GPKG file not found."
            self.gpkg_cache[full_gpkg_path] = gpd.read_file(full_gpkg_path)

        roads_gdf = self.gpkg_cache[full_gpkg_path]
        patch_roads = roads_gdf[roads_gdf['patch_id'].astype(str) == str(patch_id)]
        target_road = patch_roads[patch_roads['osm_id'].astype(str) == str(target_osm_id)]

        if target_road.empty:
            return [], "Error: Target road not found."

        row = target_road.iloc[0]
        geom = row.geometry
        if geom.geom_type == 'LineString':
            world_coords = list(geom.coords)
        elif geom.geom_type == 'MultiLineString':
            world_coords = []
            for line in geom.geoms:
                world_coords.extend(list(line.coords))
        else:
            return [], f"Error: Unsupported geometry type {geom.geom_type}."

        min_x = float(row["patch_min_x"])
        min_y = float(row["patch_min_y"])
        max_x = float(row["patch_max_x"])
        max_y = float(row["patch_max_y"])
        width, height = image_size
        if max_x == min_x or max_y == min_y:
            return [], "Error: Invalid patch bounds."

        coords = []
        for x, y in world_coords:
            px = (float(x) - min_x) / (max_x - min_x) * width
            py = (max_y - float(y)) / (max_y - min_y) * height
            coords.append([px, py])

        if len(coords) > 15:
            step = max(1, len(coords) // 15)
            sampled_coords = coords[::step][:15]
            if sampled_coords[-1] != coords[-1]:
                sampled_coords.append(coords[-1])
        else:
            sampled_coords = coords

        formatted_coords = format_normalized_points(sampled_coords, image_size)
        return coords, f"Normalized trajectory: {formatted_coords}"

    def get_road_trajectory(self, data_dir: str, gpkg_rel_path: str, patch_id: str, target_osm_id: str) -> str:
        _, trajectory_str = self.get_road_route(data_dir, gpkg_rel_path, patch_id, target_osm_id)
        return trajectory_str

    def build_content(self, data_dir: str, item: dict) -> list:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        instruction = item.get('instruction', "")
        options = item.get('options', {})
        road_info = ref.get('road', {})
        gpkg_path = road_info.get('gpkg_path', "")
        patch_id = road_info.get('patch_id', "")
        target_road_id = ref.get('target_road_id', "")
        cache_dir = "./cache_v2"

        route_coords = []
        trajectory_str = "Trajectory: Information unavailable."
        if gpkg_path and patch_id and target_road_id:
            image_size = (1024, 1024)
            if image_paths:
                first_image_path = os.path.join(data_dir, image_paths[0])
                if os.path.exists(first_image_path):
                    with Image.open(first_image_path) as first_image:
                        image_size = first_image.size
            route_coords, trajectory_str = self.get_road_route(
                data_dir,
                gpkg_path,
                patch_id,
                target_road_id,
                image_size=image_size,
            )

        content = []

        # 1. Load Images and draw the target road on both pre/post-disaster images.
        if route_coords:
            os.makedirs(cache_dir, exist_ok=True)
            route_items = [{"label": "Target Road", "pixel_coordinates": route_coords}]
        else:
            route_items = []

        for path in image_paths:
            full_path = os.path.join(data_dir, path)
            image_path = full_path
            if route_items:
                image_path = os.path.abspath(draw_routes_on_image(full_path, route_items, cache_dir))
            content.append({
                "type": "image",
                "image": f"file://{image_path}",
            })

        # 2. Construct Prompt
        context_info = "Visual Reference:\n"
        if len(image_paths) >= 2:
            if route_items:
                context_info += "- Image 1: Pre-disaster Satellite Image with the target road highlighted\n"
                context_info += "- Image 2: Post-disaster Satellite Image with the same target road highlighted\n"
            else:
                context_info += "- Image 1: Pre-disaster Satellite Image\n"
                context_info += "- Image 2: Post-disaster Satellite Image\n"
        else:
            if route_items:
                context_info += "- Image 1: Post-disaster Satellite Image with the target road highlighted\n"
            else:
                context_info += "- Image 1: Post-disaster Satellite Image\n"

        target_info = (
            f"\nTarget Road Definition:\n"
            f"The target road segment is highlighted in the images and defined by the following normalized [x, y] coordinates tracing its path:\n"
            f"{trajectory_str}\n"
        )

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: Road Damage Reasoning\n\n"
            f"{context_info}"
            f"{target_info}\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Please focus on the highlighted target road and compare its state between Image 1 and Image 2. Use the trajectory coordinates only as auxiliary location information. Select the option that best describes its damage. Output only the option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content
