import re
from difflib import SequenceMatcher


LETTER_SET = "ABCDEFGH"


class BenchmarkEvaluator:
    def extract_single_choice(self, response: str) -> str:
        if not response:
            return "FAILED"

        text = response.strip()
        patterns = [
            r"(?:final answer|correct answer|answer|prediction|option)\s*[:：]?\s*([A-H])\b",
            r"^\s*[\(\[\*]*([A-H])[\)\]\*]*\s*$",
            r"\b([A-H])\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return "FAILED"

    def extract_multi_choice(self, response: str) -> str:
        if not response:
            return "FAILED"

        text = response.strip()
        cue_patterns = [
            r"(?:final answer|correct answer|selected options|answer|prediction|output is)\s*[:：]?\s*([A-H][,\sA-Hand]*)",
            r"^\s*([A-H](?:[,\s]+[A-H])*)\s*$",
            r"^\s*([A-H]+)\s*$",
        ]
        for pattern in cue_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            parsed = self._normalize_letter_block(match.group(1))
            if parsed:
                return parsed

        option_blocks = re.findall(r"^\s*([A-H])\s*:", text, re.IGNORECASE | re.MULTILINE)
        if option_blocks:
            return "".join(sorted(dict.fromkeys(letter.upper() for letter in option_blocks)))

        parsed = self._normalize_letter_block(text)
        return parsed or "FAILED"

    def _normalize_letter_block(self, text: str) -> str:
        matches = re.findall(r"\b([A-H])\b", text, re.IGNORECASE)
        if not matches:
            tight = re.fullmatch(rf"[{LETTER_SET}]+", text.strip(), re.IGNORECASE)
            if not tight:
                return ""
            matches = list(tight.group(0))
        return "".join(sorted(dict.fromkeys(letter.upper() for letter in matches)))

    def normalize_prediction(self, item: dict, response: str):
        qtype = str(item.get("question_type", "single_choice"))
        if qtype == "single_choice":
            return self.extract_single_choice(response)
        if qtype in {"multiple_choice", "multiple_selection"}:
            return self.extract_multi_choice(response)
        return (response or "").strip()

    def evaluate_response(self, item: dict, prediction):
        gt = item.get("ground_truth", {}).get("answer")
        qtype = str(item.get("question_type", "single_choice"))
        if gt is None:
            return False, 0.0

        if qtype == "single_choice":
            expected = str(gt).strip().upper()
            pred = str(prediction).strip().upper()
            return expected == pred, 1.0 if expected == pred else 0.0

        if qtype in {"multiple_choice", "multiple_selection"}:
            if isinstance(gt, str):
                expected = "".join(sorted(dict.fromkeys(re.findall(r"[A-H]", gt.upper()))))
            else:
                expected = "".join(sorted(dict.fromkeys(str(x).strip().upper() for x in gt)))
            pred = str(prediction).strip().upper()
            return expected == pred, 1.0 if expected == pred else 0.0

        pred_text = str(prediction).strip().lower()
        gt_text = str(gt).strip().lower()
        if gt_text and gt_text in pred_text:
            return True, 1.0
        score = SequenceMatcher(None, pred_text, gt_text).ratio()
        return score > 0.3, score
