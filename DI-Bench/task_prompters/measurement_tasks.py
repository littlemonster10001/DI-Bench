import os
from typing import Dict, Any, List
from .base_prompter import BaseTaskPrompter
from utils import draw_polygons_on_image, format_normalized_points, get_image_size

class AreaEstimationPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        bboxes = ref.get('bboxes', [])
        gsd = ref.get('gsd_meters_per_pixel', 0.6)
        instruction = item.get('instruction', "")
        options = item.get('options', {})
        cache_dir = "./cache_v2"
        os.makedirs(cache_dir, exist_ok=True)

        content = []

        for idx, path in enumerate(image_paths):
            full_path = os.path.join(data_dir, path)
            
            if idx == len(image_paths) - 1 and bboxes:
                drawn_image_path = draw_polygons_on_image(full_path, bboxes, cache_dir)
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
            context_info += "- Image 2: Post-disaster Image with the target object highlighted by a red polygon\n"
        else:
            context_info += "- Image 1: Image with the target object highlighted by a red polygon\n"

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: Area Estimation\n\n"
            f"{context_info}\n"
            f"CRITICAL: The Ground Sample Distance (GSD) of this image is {gsd} meters per pixel.\n\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Please estimate the floor area of the building highlighted with the red polygon outline in the image. Output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content

class AreaEstimationTextPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        bboxes = ref.get('bboxes', [])
        gsd = ref.get('gsd_meters_per_pixel', 0.6)
        instruction = item.get('instruction', "")
        options = item.get('options', {})

        content = []

        for path in image_paths:
            full_path = os.path.join(data_dir, path)
            content.append({
                "type": "image",
                "image": f"file://{full_path}"
            })

        coord_text = ""
        coord_image_path = os.path.join(data_dir, image_paths[-1]) if image_paths else ""
        image_size = get_image_size(coord_image_path)
        if bboxes:
            poly = bboxes[0]
            pts = format_normalized_points(poly, image_size)
            coord_text = f"The building you need to analyze is defined by the following normalized contour:\n[{pts}]"

        context_info = "Visual Reference:\n"
        if len(image_paths) >= 2:
            context_info += "- Image 1: Pre-disaster Image\n"
            context_info += "- Image 2: Post-disaster Image\n"
        else:
            context_info += "- Image 1: Image for analysis\n"

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: Area Estimation\n\n"
            f"{context_info}\n"
            f"Please note that coordinates are normalized: top-left is [0, 0] and bottom-right is [1, 1].\n"
            f"CRITICAL: The Ground Sample Distance (GSD) of this image is {gsd} meters per pixel.\n"
            f"{coord_text}\n\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Please locate the building based on the coordinates and estimate its floor area. Output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content

class LengthEstimationPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        bboxes = ref.get('bboxes', [])
        gsd = ref.get('gsd_meters_per_pixel', 0.6)
        instruction = item.get('instruction', "")
        options = item.get('options', {})
        cache_dir = "./cache_v2"
        os.makedirs(cache_dir, exist_ok=True)

        content = []

        for idx, path in enumerate(image_paths):
            full_path = os.path.join(data_dir, path)
            
            if idx == len(image_paths) - 1 and bboxes:
                drawn_image_path = draw_polygons_on_image(full_path, bboxes, cache_dir)
                content.append({
                    "type": "image",
                    "image": f"file://{os.path.abspath(drawn_image_path)}"
                })
            else:
                content.append({
                    "type": "image",
                    "image": f"file://{full_path}"
                })

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: Length Estimation\n\n"
            f"CRITICAL: The Ground Sample Distance (GSD) of this image is {gsd} meters per pixel.\n\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Please observe the object highlighted by the red polygon. Estimate the length of its longest side based on the GSD. Output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content

class LengthEstimationTextPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        bboxes = ref.get('bboxes', [])
        gsd = ref.get('gsd_meters_per_pixel', 0.6)
        instruction = item.get('instruction', "")
        options = item.get('options', {})

        content = []

        for path in image_paths:
            full_path = os.path.join(data_dir, path)
            content.append({
                "type": "image",
                "image": f"file://{full_path}"
            })

        coord_text = ""
        coord_image_path = os.path.join(data_dir, image_paths[-1]) if image_paths else ""
        image_size = get_image_size(coord_image_path)
        if bboxes:
            poly = bboxes[0]
            pts = format_normalized_points(poly, image_size)
            coord_text = f"The target object is defined by the following normalized contour:\n[{pts}]"

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: Length Estimation\n\n"
            f"Please note that coordinates are normalized: top-left is [0, 0] and bottom-right is [1, 1].\n"
            f"CRITICAL: The Ground Sample Distance (GSD) of this image is {gsd} meters per pixel.\n"
            f"{coord_text}\n\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Estimate the length of the longest side of the defined object based on the GSD. Output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content

class DistanceEstimationPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        bboxes = ref.get('bboxes', [])
        gsd = ref.get('gsd_meters_per_pixel', 0.6)
        instruction = item.get('instruction', "")
        options = item.get('options', {})
        cache_dir = "./cache_v2"
        os.makedirs(cache_dir, exist_ok=True)

        content = []

        for idx, path in enumerate(image_paths):
            full_path = os.path.join(data_dir, path)
            
            if idx == len(image_paths) - 1 and bboxes:
                drawn_image_path = draw_polygons_on_image(full_path, bboxes, cache_dir)
                content.append({
                    "type": "image",
                    "image": f"file://{os.path.abspath(drawn_image_path)}"
                })
            else:
                content.append({
                    "type": "image",
                    "image": f"file://{full_path}"
                })

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: Distance Estimation\n\n"
            f"CRITICAL: The Ground Sample Distance (GSD) of this image is {gsd} meters per pixel.\n\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Please observe the objects highlighted by the red polygons in the image. Estimate the real-world distance between them based on the provided GSD. Output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content

class DistanceEstimationTextPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        bboxes = ref.get('bboxes', [])
        gsd = ref.get('gsd_meters_per_pixel', 0.6)
        instruction = item.get('instruction', "")
        options = item.get('options', {})

        content = []

        for path in image_paths:
            full_path = os.path.join(data_dir, path)
            content.append({
                "type": "image",
                "image": f"file://{full_path}"
            })

        coord_text = ""
        coord_image_path = os.path.join(data_dir, image_paths[-1]) if image_paths else ""
        image_size = get_image_size(coord_image_path)
        if bboxes and len(bboxes) >= 2:
            obj1_pts = format_normalized_points(bboxes[0], image_size)
            obj2_pts = format_normalized_points(bboxes[1], image_size)
            coord_text = (
                f"Object A is located within the polygon defined by normalized coordinates:\n[{obj1_pts}]\n"
                f"Object B is located within the polygon defined by normalized coordinates:\n[{obj2_pts}]"
            )

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: Distance Estimation\n\n"
            f"Please note that coordinates are normalized: top-left is [0, 0] and bottom-right is [1, 1].\n"
            f"CRITICAL: The Ground Sample Distance (GSD) of this image is {gsd} meters per pixel.\n"
            f"{coord_text}\n\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Based on their normalized coordinates and the given GSD, calculate or estimate the real-world distance between Object A and Object B. Output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content
