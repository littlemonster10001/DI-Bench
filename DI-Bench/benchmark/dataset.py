import json
from pathlib import Path


def resolve_scene_dir(data_path: str, scene: str | None = None) -> Path:
    data_root = Path(data_path)
    if (data_root / "questions.json").is_file():
        return data_root
    if scene is None:
        raise ValueError("When --data_path points to the dataset root, --scene is required.")
    scene_dir = data_root / scene
    if not (scene_dir / "questions.json").is_file():
        raise FileNotFoundError(f"questions.json not found under {scene_dir}")
    return scene_dir


def load_scene_questions(scene_dir: Path, task_type: str = "all", limit: int | None = None):
    payload = json.loads((scene_dir / "questions.json").read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Unexpected questions.json format in {scene_dir}")

    scene_blob = payload[0]
    scene_info = scene_blob.get("scene_info", {})
    questions = scene_blob.get("questions", [])
    selected = []

    for item in questions:
        current_type = str(item.get("task_type", ""))
        if task_type.lower() != "all" and current_type != task_type:
            continue
        record = dict(item)
        record["scene_info"] = scene_info
        record["scene_id"] = scene_info.get("scene_id", scene_dir.name)
        record["unique_id"] = f"{record['scene_id']}_{record.get('question_id')}"
        selected.append(record)

    if limit is not None:
        selected = selected[:limit]
    return scene_info, selected
