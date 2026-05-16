import inspect
import os
from urllib.parse import urlparse

from PIL import Image
from transformers import AutoProcessor

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams

from model_adapters.base import BaseModelAdapter

from utils import BenchmarkPrompter


class QwenVLLMAdapter(BaseModelAdapter):
    def __init__(
        self,
        model_path: str,
        bbox_mode: str = "visual",
        source_mode: str = "full",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        max_num_seqs: int = 1,
        seed: int = 42,
        max_model_len: int | None = None,
        enforce_eager: bool = False,
        limit_mm_per_prompt: int | None = None,
        disable_mm_preprocessor_cache: bool = False,
    ):
        self.prompter = BenchmarkPrompter(bbox_mode=bbox_mode, source_mode=source_mode)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        self.source_mode = source_mode

        llm_kwargs = {
            "model": model_path,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "seed": seed,
        }
        optional_kwargs = {
            "max_model_len": max_model_len,
            "enforce_eager": enforce_eager,
            "limit_mm_per_prompt": limit_mm_per_prompt,
            "disable_mm_preprocessor_cache": disable_mm_preprocessor_cache,
        }
        llm_kwargs.update(self._filter_supported_kwargs(optional_kwargs))
        self.llm = LLM(**llm_kwargs)

    def generate(self, item, scene_dir):
        scene_dir = str(scene_dir)
        prompt_bundle = self._build_prompt_bundle(scene_dir, item)
        prompt = self._build_chat_prompt(prompt_bundle["messages"])
        llm_input = {"prompt": prompt}
        if prompt_bundle["images"]:
            llm_input["multi_modal_data"] = {"image": prompt_bundle["images"]}

        outputs = self.llm.generate([llm_input], sampling_params=self.sampling_params, use_tqdm=False)
        output = outputs[0].outputs[0]
        return {
            "text": output.text.strip(),
            "prompt": prompt_bundle["readable_prompt"],
            "meta": {
                "num_images": len(prompt_bundle["images"]),
                "finish_reason": getattr(output, "finish_reason", None),
                "few_shot_count": len(item.get("_few_shot_examples", [])),
                "source_mode": self.source_mode,
            },
        }

    def _build_prompt_bundle(self, scene_dir: str, item: dict):
        os.environ["DI_BENCH_SOURCE_MODE"] = self.source_mode
        current_messages = self.prompter.get_messages(scene_dir, item)
        messages = [current_messages[0]]

        for example in item.get("_few_shot_examples", []) or []:
            example_messages = self.prompter.get_messages(scene_dir, example)
            messages.append(example_messages[-1])
            messages.append({"role": "assistant", "content": self._format_few_shot_answer(example)})

        messages.append(current_messages[-1])
        prompt, image_inputs = self._build_prompt_and_images(messages)
        return {
            "messages": messages,
            "images": image_inputs,
            "readable_prompt": prompt,
        }

    def _build_chat_prompt(self, messages):
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _build_prompt_and_images(self, messages):
        normalized_messages = []
        image_inputs = []

        for message in messages:
            content_blocks = message.get("content", [])
            if isinstance(content_blocks, str):
                normalized_messages.append({"role": message["role"], "content": content_blocks})
                continue

            normalized_content = []
            for block in content_blocks:
                block_type = block.get("type")
                if block_type == "image":
                    image_path = self._strip_file_scheme(block["image"])
                    image_inputs.append(Image.open(image_path).convert("RGB"))
                    normalized_content.append({"type": "image"})
                    continue
                if block_type == "text":
                    normalized_content.append({"type": "text", "text": block["text"]})
                    continue
                if block_type == "video":
                    raise NotImplementedError("Video inputs are not supported in qwen_local_clean.")
                raise ValueError(f"Unsupported content block type: {block_type}")
            normalized_messages.append({"role": message["role"], "content": normalized_content})

        prompt = self.processor.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt, image_inputs

    def _format_few_shot_answer(self, item):
        ground_truth = item.get("ground_truth", {})
        answer = ground_truth.get("answer") if isinstance(ground_truth, dict) else ground_truth
        if isinstance(answer, (list, tuple)):
            return ", ".join(str(value) for value in answer)
        if answer is None:
            return ""
        return str(answer)

    def _strip_file_scheme(self, uri: str):
        parsed = urlparse(uri)
        if parsed.scheme == "file":
            return parsed.path
        return uri

    def _filter_supported_kwargs(self, candidate_kwargs: dict):
        supported = set(inspect.signature(LLM.__init__).parameters.keys())
        return {key: value for key, value in candidate_kwargs.items() if value is not None and key in supported}
