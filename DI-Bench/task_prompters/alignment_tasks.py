import os
from typing import Dict, Any, List
from .base_prompter import BaseTaskPrompter
from utils import draw_bboxes_on_image, format_normalized_bbox, format_normalized_point, get_image_size, normalize_bbox

class POIAlignmentPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_list = ref.get('images', [])
        target_poi = ref.get('target_poi', {})
        regions = ref.get('regions', [])
        instruction = item.get('instruction', "")
        options = item.get('options', {})
        cache_dir = "./cache_v2"
        os.makedirs(cache_dir, exist_ok=True)

        content = []

        poi_coords = target_poi.get('pixel_coords', [])
        poi_type = target_poi.get('fclass', 'unknown')
        ref_image_path = os.path.join(data_dir, image_list[0]) if image_list else ""
        image_size = get_image_size(ref_image_path)
        norm_poi = format_normalized_point(poi_coords, image_size)

        # 1. 图像处理 (画框模式)
        if image_list:
            rel_path = image_list[0]
            full_path = os.path.join(data_dir, rel_path)

            opts_list = []
            for reg in regions:
                label = reg.get('label', '')
                bbox = reg.get('bbox', [])
                if len(bbox) == 4:
                    opts_list.append((label, bbox))

            if opts_list:
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

        # 2. 构建 Prompt
        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: POI Alignment\n\n"
            f"Target POI Definition:\n"
            f"- Feature Class: {poi_type}\n"
            f"- Normalized Coordinates: {norm_poi}\n\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: The candidate regions are marked with Red Bounding Boxes and labels in the image. "
            f"Please output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content


class POIAlignmentTextPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_list = ref.get('images', [])
        target_poi = ref.get('target_poi', {})
        regions = ref.get('regions', [])
        instruction = item.get('instruction', "")
        options = item.get('options', {})

        content = []

        poi_coords = target_poi.get('pixel_coords', [])
        poi_type = target_poi.get('fclass', 'unknown')
        ref_image_path = os.path.join(data_dir, image_list[0]) if image_list else ""
        image_size = get_image_size(ref_image_path)
        norm_poi = format_normalized_point(poi_coords, image_size)

        # 1. 直接加载原图
        if image_list:
            rel_path = image_list[0]
            full_path = os.path.join(data_dir, rel_path)
            content.append({
                "type": "image",
                "image": f"file://{full_path}"
            })

        # 2. 准备选项和坐标的映射
        region_dict = {reg.get('label', ''): reg.get('bbox', []) for reg in regions}
        
        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            opt_label = options[key]
            bbox = region_dict.get(opt_label, "Unknown BBox")
            bbox_text = format_normalized_bbox(bbox, image_size) if isinstance(bbox, list) else "Unknown BBox"
            options_str += f"{key}: {opt_label}, located at normalized BBox {bbox_text}\n"

        prompt_text = (
            f"Task: POI Alignment\n\n"
            f"Target POI Definition:\n"
            f"- Feature Class: {poi_type}\n"
            f"- Normalized Coordinates: {norm_poi}\n\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Please match the Target POI coordinates with the normalized bounding boxes provided in the options. "
            f"Output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content


class PopulationEstimationPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        bbox = ref.get('bbox', None)
        instruction = item.get('instruction', "")
        options = item.get('options', {})
        cache_dir = "./cache_v2"
        os.makedirs(cache_dir, exist_ok=True)

        content = []
        bbox_ref_image_path = os.path.join(data_dir, image_paths[0]) if image_paths else ""
        bbox_ref_size = get_image_size(bbox_ref_image_path)
        norm_bbox_values = normalize_bbox(bbox, bbox_ref_size) if bbox else []

        # 1. 加载图像并画框
        for idx, path in enumerate(image_paths):
            full_path = os.path.join(data_dir, path)
            
            if norm_bbox_values:
                current_width, current_height = get_image_size(full_path)
                scaled_bbox = [
                    norm_bbox_values[0] * max(1, current_width - 1),
                    norm_bbox_values[1] * max(1, current_height - 1),
                    norm_bbox_values[2] * max(1, current_width - 1),
                    norm_bbox_values[3] * max(1, current_height - 1),
                ]
                opts_list = [("Target", scaled_bbox)]
                # 因为 TIF 格式 Image.open 读取后直接画框保存可能会变样，这里兼容一下 RGB 转换
                try:
                    from PIL import Image
                    img = Image.open(full_path)
                    # 强制转为 RGB，防止 16-bit TIF 或单通道图无法用红色画笔
                    if img.mode != 'RGB':
                        # 如果是TIF热力图，转换RGB后可能会丢失原始数据精度，但仅用于视觉显示没问题
                        pass
                except Exception:
                    pass

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

        # 2. 构建 Prompt
        context_info = "Visual Reference:\n"
        if len(image_paths) == 3:
            context_info += "- Image 1: Pre-disaster RGB Image\n"
            context_info += "- Image 2: Post-disaster RGB Image\n"
            context_info += "- Image 3: Population Distribution Heatmap\n"
        
        if norm_bbox_values:
            context_info += "Note: The target area is marked with a Red Bounding Box across all images.\n"

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: Population Estimation\n\n"
            f"{context_info}\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Please estimate based on the heatmap and output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content


class PopulationEstimationTextPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        bbox = ref.get('bbox', None)
        instruction = item.get('instruction', "")
        options = item.get('options', {})

        content = []

        # 1. 加载原图
        for path in image_paths:
            full_path = os.path.join(data_dir, path)
            content.append({
                "type": "image",
                "image": f"file://{full_path}",
            })

        # 2. 构建 Prompt
        context_info = "Visual Reference:\n"
        if len(image_paths) == 3:
            context_info += "- Image 1: Pre-disaster RGB Image\n"
            context_info += "- Image 2: Post-disaster RGB Image\n"
            context_info += "- Image 3: Population Distribution Heatmap\n"
        
        if bbox:
            bbox_image_path = os.path.join(data_dir, image_paths[0]) if image_paths else ""
            norm_bbox = format_normalized_bbox(bbox, get_image_size(bbox_image_path))
            context_info += f"Note: The target area is located at normalized BBox {norm_bbox} in [xmin, ymin, xmax, ymax] format.\n"

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: Population Estimation\n\n"
            f"{context_info}\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Please map the provided normalized BBox coordinates to the heatmap to estimate the population. Output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content


class HeightAlignmentPrompter(BaseTaskPrompter):
    # 此任务通常没有 bbox，所以只需要一个类即可兼顾
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        target_poi = ref.get('target_poi', {})
        instruction = item.get('instruction', "")
        options = item.get('options', {})

        content = []

        # 1. 加载原图
        for path in image_paths:
            full_path = os.path.join(data_dir, path)
            content.append({
                "type": "image",
                "image": f"file://{full_path}",
            })

        # 2. 构建 Prompt
        poi_coords = target_poi.get('pixel_coords', [])
        point_image_path = os.path.join(data_dir, image_paths[0]) if image_paths else ""
        norm_poi = format_normalized_point(poi_coords, get_image_size(point_image_path))
        
        context_info = "Visual Reference:\n"
        if len(image_paths) == 2:
            context_info += "- Image 1: Post-disaster RGB Image\n"
            context_info += "- Image 2: Digital Elevation Model (DSM) Heatmap\n"
            
        context_info += f"Target Point Normalized Coordinates: {norm_poi}\n"

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: Height Alignment\n\n"
            f"{context_info}\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content


class HeightComparisonPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        bboxes = ref.get('bboxes', [])
        instruction = item.get('instruction', "")
        options = item.get('options', {})
        cache_dir = "./cache_v2"
        os.makedirs(cache_dir, exist_ok=True)

        content = []
        opts_list = []
        for entry in bboxes:
            label = entry.get('id', entry.get('label', 'Region'))
            bbox = entry.get('bbox', [])
            if len(bbox) == 4:
                opts_list.append((label, bbox))

        for idx, path in enumerate(image_paths):
            full_path = os.path.join(data_dir, path)
            if opts_list:
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
        if len(image_paths) == 2:
            context_info += "- Image 1: Post-disaster RGB Image with candidate regions marked\n"
            context_info += "- Image 2: Digital Elevation Model (DSM) with the same candidate regions marked\n"
        else:
            context_info += "- Candidate regions are marked with labels on the provided images\n"

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: Height Comparison\n\n"
            f"{context_info}\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Compare the elevations of the marked regions using the DSM reference. "
            f"Output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content


class HeightComparisonTextPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        image_paths = ref.get('images', [])
        bboxes = ref.get('bboxes', [])
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
        if len(image_paths) == 2:
            context_info += "- Image 1: Post-disaster RGB Image\n"
            context_info += "- Image 2: Digital Elevation Model (DSM)\n"
        else:
            context_info += "- Images are provided in the listed order\n"

        bbox_image_path = os.path.join(data_dir, image_paths[0]) if image_paths else ""
        image_size = get_image_size(bbox_image_path)
        bbox_text = "Candidate Regions (normalized bounding boxes in [xmin, ymin, xmax, ymax] format):\n"
        for entry in bboxes:
            label = entry.get('id', entry.get('label', 'Region'))
            bbox_text += f"- Region {label}: {format_normalized_bbox(entry.get('bbox', []), image_size)}\n"

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            options_str += f"{key}: {options[key]}\n"

        prompt_text = (
            f"Task: Height Comparison\n\n"
            f"{context_info}\n"
            f"{bbox_text}\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Instructions: Use the region coordinates and DSM reference to compare elevations. "
            f"Output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content
