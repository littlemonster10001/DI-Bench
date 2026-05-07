import json
from pathlib import Path

from openpyxl import Workbook


def _json_cell(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_workbook(output_path: str, scene_info: dict, results: list[dict], summary: dict):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    summary_ws = workbook.active
    summary_ws.title = "summary"
    summary_ws.append(["field", "value"])
    summary_ws.append(["scene_id", scene_info.get("scene_id", "")])
    summary_ws.append(["location", scene_info.get("location", "")])
    summary_ws.append(["timestamp", scene_info.get("timestamp", "")])
    summary_ws.append(["disaster_type", scene_info.get("disaster_type", "")])
    summary_ws.append(["requested_items", summary["requested_items"]])
    summary_ws.append(["evaluated_items", summary["evaluated_items"]])
    summary_ws.append(["overall_accuracy", summary["overall_accuracy"]])
    summary_ws.append(["overall_avg_score", summary["overall_avg_score"]])
    summary_ws.append([])
    summary_ws.append(["task_type", "accuracy", "avg_score", "count", "correct"])
    for task_name, stats in summary["per_task"].items():
        summary_ws.append([task_name, stats["accuracy"], stats["avg_score"], stats["count"], stats["correct"]])

    details_ws = workbook.create_sheet("details")
    details_ws.append(
        [
            "scene_id",
            "question_id",
            "task_type",
            "question_type",
            "instruction",
            "ground_truth",
            "model_raw_response",
            "extracted_answer",
            "is_correct",
            "score",
            "prompt",
            "meta",
        ]
    )
    for row in results:
        details_ws.append(
            [
                row.get("scene_id"),
                row.get("question_id"),
                row.get("task_type"),
                row.get("question_type"),
                row.get("instruction"),
                _json_cell(row.get("ground_truth")),
                row.get("model_raw_response"),
                _json_cell(row.get("extracted_answer")),
                row.get("is_correct"),
                row.get("score"),
                row.get("prompt"),
                _json_cell(row.get("meta")),
            ]
        )

    workbook.save(output_file)
