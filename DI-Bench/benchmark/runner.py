from collections import defaultdict

from tqdm import tqdm

from model_adapters.base import BaseModelAdapter

from .dataset import load_scene_questions
from .reporter import write_workbook


class BenchmarkRunner:
    def __init__(
        self,
        model_adapter,
        evaluator,
        few_shot_k: int = 0,
        few_shot_scope: str = "same_task",
        few_shot_selection: str = "first",
    ):
        self.model_adapter = model_adapter
        self.evaluator = evaluator
        self.few_shot_k = few_shot_k
        self.few_shot_scope = few_shot_scope
        self.few_shot_selection = few_shot_selection

    def run(
        self,
        scene_dir,
        output_path: str,
        task_type: str = "all",
        limit: int | None = None,
        save_outputs: bool = True,
        return_summary: bool = False,
    ):
        scene_info, questions = load_scene_questions(scene_dir, task_type=task_type, limit=limit)
        if not questions:
            raise ValueError("No questions matched the requested task filter.")

        results = []
        stats = defaultdict(lambda: {"count": 0, "correct": 0, "score_sum": 0.0})

        for item in tqdm(questions, desc="Evaluating", dynamic_ncols=True):
            item = dict(item)
            item["_few_shot_examples"] = self._select_few_shot_examples(questions, item)
            bundle = self.model_adapter.generate(item=item, scene_dir=scene_dir)
            response = bundle["text"]
            prediction = self.evaluator.normalize_prediction(item, response)
            is_correct, score = self.evaluator.evaluate_response(item, prediction)

            task_name = item.get("task_type", "unknown")
            stats[task_name]["count"] += 1
            stats[task_name]["correct"] += int(is_correct)
            stats[task_name]["score_sum"] += float(score)

            results.append(
                {
                    "scene_id": item.get("scene_id"),
                    "question_id": item.get("question_id"),
                    "task_type": task_name,
                    "question_type": item.get("question_type"),
                    "instruction": item.get("instruction"),
                    "ground_truth": item.get("ground_truth"),
                    "model_raw_response": response,
                    "extracted_answer": prediction,
                    "is_correct": is_correct,
                    "score": round(float(score), 4),
                    "prompt": bundle.get("prompt"),
                    "meta": bundle.get("meta", {}),
                }
            )

        evaluated_items = len(results)
        total_correct = sum(task["correct"] for task in stats.values())
        total_score = sum(task["score_sum"] for task in stats.values())
        summary = {
            "requested_items": len(questions),
            "evaluated_items": evaluated_items,
            "overall_accuracy": round(total_correct / evaluated_items, 4) if evaluated_items else 0.0,
            "overall_avg_score": round(total_score / evaluated_items, 4) if evaluated_items else 0.0,
            "per_task": {},
        }
        for task_name, task_stats in stats.items():
            count = task_stats["count"]
            summary["per_task"][task_name] = {
                "accuracy": round(task_stats["correct"] / count, 4) if count else 0.0,
                "avg_score": round(task_stats["score_sum"] / count, 4) if count else 0.0,
                "count": count,
                "correct": task_stats["correct"],
            }

        if save_outputs:
            write_workbook(output_path, scene_info, results, summary)
        if return_summary:
            return scene_info, results, summary
        return scene_info, results, summary

    def _select_few_shot_examples(self, all_items, current_item):
        if self.few_shot_k <= 0:
            return []

        candidates = []
        for item in all_items:
            if item.get("unique_id") == current_item.get("unique_id"):
                continue
            if self.few_shot_scope == "same_task" and item.get("task_type") != current_item.get("task_type"):
                continue
            candidates.append(item)

        if self.few_shot_selection == "random":
            import random

            rng = random.Random(42)
            rng.shuffle(candidates)
        return [dict(example) for example in candidates[: self.few_shot_k]]
