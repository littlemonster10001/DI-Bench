import os
from typing import Dict, Any, List

class BaseTaskPrompter:
    """抽象基类，定义了 V2 任务策略类的通用接口"""
    def build_content(self, data_dir: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement this method")
