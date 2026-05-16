import os
from typing import Dict, Any, List
from .base_prompter import BaseTaskPrompter
from utils import draw_bboxes_on_image, format_normalized_bbox, get_image_size

class ImageLevelCrossViewMatchingPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        query_img = ref.get('query_image', "")
        candidates = ref.get('candidate_images', [])
        instruction = item.get('instruction', "")
        options = item.get('options', {})

        content = []

        # 1. 加载 Query Image (将作为 Image 1)
        if query_img:
            full_query_path = os.path.join(data_dir, query_img)
            content.append({
                "type": "image",
                "image": f"file://{full_query_path}",
            })

        # 2. 依次加载 Candidate Images (将作为 Image 2, Image 3...)
        for path in candidates:
            full_path = os.path.join(data_dir, path)
            content.append({
                "type": "image",
                "image": f"file://{full_path}",
            })

        # 3. 构建 Prompt
        context_info = (
            "Visual Reference:\n"
            "- Image 1: Query Image (Street View or Reference)\n"
            "- Image 2 onwards: Candidate Images\n\n"
        )

        options_str = ""
        sorted_keys = sorted(options.keys())
        for idx, key in enumerate(sorted_keys):
            # 同样显式映射：A -> Image 2, B -> Image 3, etc.
            options_str += f"{key}: {options[key]} (corresponds to Image {idx + 2})\n"

        prompt_text = (
            f"Task: Cross-View Matching (Image Level)\n\n"
            f"{context_info}"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Please output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content


class ObjectLevelCrossViewMatchingPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = []
        cache_dir = "./cache_v2"
        os.makedirs(cache_dir, exist_ok=True)

        ref = item.get('ref', {})
        query_img = ref.get('query_image', "")
        instruction = item.get('instruction', "")
        options = item.get('options', {})

        # 1. 加载 Query Image
        if query_img:
            full_query_path = os.path.join(data_dir, query_img)
            content.append({
                "type": "image",
                "image": f"file://{full_query_path}",
            })

        # 2. 按 satellite image_path 对 Option BBox 进行分组，避免重复画图
        image_groups = {}
        sorted_keys = sorted(options.keys())

        for key in sorted_keys:
            opt = options[key]
            path = opt.get('image_path', "")
            bbox = opt.get('bbox', [])

            if path:
                if path not in image_groups:
                    image_groups[path] = []
                # 记录这幅图上要画哪些框及其对应的 Option 字母
                image_groups[path].append((key, bbox))

        # 3. 绘制带有红框的候选图，并记录其在 Prompt 中的索引
        sat_image_path_to_index = {}

        for rel_path, opts_list in image_groups.items():
            full_orig_path = os.path.join(data_dir, rel_path)
            # 调用 utils 中的绘图函数，框上会打上 A/B/C/D 标签
            drawn_image_path = draw_bboxes_on_image(full_orig_path, opts_list, cache_dir)

            content.append({
                "type": "image",
                "image": f"file://{os.path.abspath(drawn_image_path)}",
            })
            # 记录这幅图对应的是第几张候选图 (content 列表长度减 1)
            sat_image_path_to_index[rel_path] = len(content) - 1

        # 4. 构建 Prompt 文本
        context_info = (
            "Visual Reference:\n"
            "- Image 1: Query Image (Street View)\n"
            "- Subsequent Images: Candidate Satellite Images with red bounding boxes labeled by option letters\n\n"
        )

        options_str = ""
        for key in sorted_keys:
            opt = options[key]
            sat_path = opt.get('image_path', "")
            
            # 明确告诉模型这个选项在第几张图里，降低识别难度
            if sat_path in sat_image_path_to_index:
                img_idx = sat_image_path_to_index[sat_path] + 1
                options_str += f"{key}: BBox {key} in Image {img_idx}\n"
            else:
                options_str += f"{key}: BBox {key}\n"

        prompt_text = (
            f"Task: Cross-View Matching (Object Level)\n\n"
            f"{context_info}"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Please output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content


class ObjectLevelCrossViewMatchingTextPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = []

        ref = item.get('ref', {})
        query_img = ref.get('query_image', "")
        instruction = item.get('instruction', "")
        options = item.get('options', {})

        # 1. 加载 Query Image
        if query_img:
            full_query_path = os.path.join(data_dir, query_img)
            content.append({
                "type": "image",
                "image": f"file://{full_query_path}",
            })

        # 2. 收集并去重候选图像
        unique_candidate_paths = []
        for key, opt in options.items():
            path = opt.get('image_path', "")
            if path and path not in unique_candidate_paths:
                unique_candidate_paths.append(path)

        # 3. 按顺序加载候选图像，并建立映射 (path -> Image Index)
        sat_image_path_to_index = {}
        for path in unique_candidate_paths:
            full_path = os.path.join(data_dir, path)
            content.append({
                "type": "image",
                "image": f"file://{full_path}",
            })
            sat_image_path_to_index[path] = len(content) - 1

        # 4. 构建 Prompt 文本
        context_info = (
            "Visual Reference:\n"
            "- Image 1: Query Image (Street View)\n"
            "- Subsequent Images: Candidate Satellite Images\n\n"
            "Note: The bounding boxes (BBox) are normalized to [0, 1] and represented as [xmin, ymin, xmax, ymax] for the corresponding image.\n\n"
        )

        options_str = ""
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            opt = options[key]
            sat_path = opt.get('image_path', "")
            bbox = opt.get('bbox', [])
            image_size = get_image_size(os.path.join(data_dir, sat_path)) if sat_path else (1024, 1024)
            norm_bbox = format_normalized_bbox(bbox, image_size)
            
            # 将路径映射回 Image X
            if sat_path in sat_image_path_to_index:
                img_idx = sat_image_path_to_index[sat_path] + 1
                options_str += f"{key}: Candidate BBox {key} is in Image {img_idx}, with normalized BBox {norm_bbox}\n"
            else:
                options_str += f"{key}: Candidate BBox {key}, with normalized BBox {norm_bbox}\n"

        prompt_text = (
            f"Task: Cross-View Matching (Object Level)\n\n"
            f"{context_info}"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Please output only the correct option letter."
        )

        content.append({"type": "text", "text": prompt_text})
        return content
