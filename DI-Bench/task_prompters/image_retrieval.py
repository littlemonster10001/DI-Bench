import os
from typing import Dict, Any, List
from .base_prompter import BaseTaskPrompter

class ImageRetrievalPrompter(BaseTaskPrompter):
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ref = item.get('ref', {})
        images = ref.get('images', [])
        instruction = item.get('instruction', "")
        options = item.get('options', {})

        content = []
        
        # 1. 加载所有图像
        for idx, path in enumerate(images):
            full_path = os.path.join(data_dir, path)
            content.append({
                "type": "image",
                "image": f"file://{full_path}",
            })

        # 2. 构建 Options 文本并映射到对应的 Image 序号
        options_str = ""
        sorted_keys = sorted(options.keys())
        for idx, key in enumerate(sorted_keys):
            # 模型默认称呼传入的第一张图为 Image 1, 第二张为 Image 2, 以此类推
            # 因此这里显式告诉模型 Option 对应的物理图像编号
            options_str += f"{key}: {options[key]} (corresponds to Image {idx + 1})\n"

        # 3. 构建完整的 Text Prompt
        prompt_text = (
            f"Task: Image Retrieval\n\n"
            f"Question: {instruction}\n\n"
            f"Options:\n"
            f"{options_str}\n"
            f"Please output only the correct option letter(s)."
        )

        content.append({"type": "text", "text": prompt_text})
        return content
