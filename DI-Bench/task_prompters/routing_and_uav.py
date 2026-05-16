import os
from typing import Dict, Any, List
from .base_prompter import BaseTaskPrompter
from utils import (
    draw_routes_on_image,
    draw_bboxes_on_image,
    format_normalized_bbox,
    format_normalized_points,
    get_image_size,
)

class RoutePlanningPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        instruction = item.get('instruction', "")
        options = item.get('options', {})
        context_paths = item.get('context_paths', [])
        cache_dir = "./cache_v2"
        os.makedirs(cache_dir, exist_ok=True)

        content = []

        # 1. 图像处理: 在最后一张图像 (通常是 Post-disaster) 上画出所有的候选路径
        for idx, path in enumerate(image_paths):
            full_path = os.path.join(data_dir, path)
            
            if idx == len(image_paths) - 1 and context_paths:
                drawn_image_path = draw_routes_on_image(full_path, context_paths, cache_dir)
                content.append({
                    "type": "image",
                    "image": f"file://{os.path.abspath(drawn_image_path)}"
                })
            else:
                content.append({
                    "type": "image",
                    "image": f"file://{full_path}"
                })

        # 2. 构建 Prompt
        context_info = "Visual Reference:\n"
        if len(image_paths) >= 2:
            context_info += "- Image 1: Pre-disaster Image\n"
            context_info += "- Image 2: Post-disaster Image with candidate routes drawn in different colors\n"
        else:
            context_info += "- Image 1: Post-disaster Image with candidate routes drawn in different colors\n"

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: Route Planning\n\n"
            f"{context_info}\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Please trace the colored paths in the image and evaluate their safety and efficiency. Output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content

class RoutePlanningTextPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        instruction = item.get('instruction', "")
        options = item.get('options', {})
        context_paths = item.get('context_paths', [])

        content = []

        # 1. 加载原图
        for path in image_paths:
            full_path = os.path.join(data_dir, path)
            content.append({
                "type": "image",
                "image": f"file://{full_path}"
            })

        # 2. 将所有路径点转换为归一化坐标文本
        coord_image_path = os.path.join(data_dir, image_paths[-1]) if image_paths else ""
        image_size = get_image_size(coord_image_path)
        paths_text = "Candidate Routes (normalized coordinates):\n"
        for p_data in context_paths:
            label = p_data.get('label', '')
            coords = p_data.get('pixel_coordinates', [])
            
            pts = format_normalized_points(coords, image_size)
            paths_text += f"- Route {label}: {pts}\n"

        context_info = "Visual Reference:\n"
        if len(image_paths) >= 2:
            context_info += "- Image 1: Pre-disaster Image\n"
            context_info += "- Image 2: Post-disaster Image\n"
        else:
            context_info += "- Image 1: Post-disaster Image\n"

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: Route Planning\n\n"
            f"Please note that coordinates are normalized: top-left is [0, 0] and bottom-right is [1, 1].\n"
            f"{context_info}\n"
            f"{paths_text}\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Please map the normalized trajectory coordinates to the visual content in the Post-disaster image. Identify which path is the safest and most efficient. Output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content

class UAVLandingAssessmentPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        regions = ref.get('regions', [])
        instruction = item.get('instruction', "")
        options = item.get('options', {})
        cache_dir = "./cache_v2"
        os.makedirs(cache_dir, exist_ok=True)

        content = []

        # 1. 图像处理: 在最后一张图上画出降落区的 BBox
        for idx, path in enumerate(image_paths):
            full_path = os.path.join(data_dir, path)
            
            if idx == len(image_paths) - 1 and regions:
                opts_list = []
                for reg in regions:
                    label = reg.get('label', 'Zone')
                    bbox = reg.get('bbox', [])
                    if len(bbox) == 4:
                        opts_list.append((label, bbox))
                
                drawn_image_path = draw_bboxes_on_image(full_path, opts_list, cache_dir)
                content.append({
                    "type": "image",
                    "image": f"file://{os.path.abspath(drawn_image_path)}"
                })
            else:
                content.append({
                    "type": "image",
                    "image": f"file://{full_path}"
                })

        context_info = "Visual Reference:\n"
        if len(image_paths) >= 2:
            context_info += "- Image 1: Pre-disaster Image\n"
            context_info += "- Image 2: Post-disaster Image with the target landing zone marked by a Red Bounding Box\n"
        else:
            context_info += "- Image 1: Post-disaster Image with the target landing zone marked by a Red Bounding Box\n"

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: UAV Landing Assessment\n\n"
            f"{context_info}\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Please carefully analyze the surface conditions within the marked landing zone. Output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content

class UAVLandingAssessmentTextPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        regions = ref.get('regions', [])
        instruction = item.get('instruction', "")
        options = item.get('options', {})

        content = []

        # 1. 加载原图
        for path in image_paths:
            full_path = os.path.join(data_dir, path)
            content.append({
                "type": "image",
                "image": f"file://{full_path}"
            })

        # 2. 坐标转换
        coord_image_path = os.path.join(data_dir, image_paths[-1]) if image_paths else ""
        image_size = get_image_size(coord_image_path)
        zones_text = "Target Landing Zones:\n"
        for reg in regions:
            label = reg.get('label', 'Zone')
            bbox = reg.get('bbox', [])
            zones_text += f"- {label} is located at normalized BBox {format_normalized_bbox(bbox, image_size)}\n"

        context_info = "Visual Reference:\n"
        if len(image_paths) >= 2:
            context_info += "- Image 1: Pre-disaster Image\n"
            context_info += "- Image 2: Post-disaster Image\n"
        else:
            context_info += "- Image 1: Post-disaster Image\n"

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: UAV Landing Assessment\n\n"
            f"Please note that coordinates are normalized: top-left is [0, 0] and bottom-right is [1, 1].\n"
            f"{context_info}\n"
            f"{zones_text}\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Locate the landing zone based on the provided normalized BBox coordinates and analyze its surface conditions. Output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content
