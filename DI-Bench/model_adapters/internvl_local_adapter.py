import os
from contextlib import nullcontext

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from transformers.generation import GenerationMixin

from model_adapters.base import BaseModelAdapter

from utils import BenchmarkPrompter


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    grid_w = target_width // image_size

    for idx in range(blocks):
        box = (
            (idx % grid_w) * image_size,
            (idx // grid_w) * image_size,
            ((idx % grid_w) + 1) * image_size,
            ((idx // grid_w) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _load_image_pixels(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def _ensure_generation_mixin(model, model_name):
    if not hasattr(model, "generate"):
        if not hasattr(model, "prepare_inputs_for_generation"):
            raise AttributeError(
                f"{model_name} does not expose generate() and is missing prepare_inputs_for_generation()."
            )

        model_cls = model.__class__
        if not getattr(model_cls, "_di_bench_generation_mixin_patched", False):
            for attr_name, attr_value in GenerationMixin.__dict__.items():
                if attr_name.startswith("__"):
                    continue
                if hasattr(model_cls, attr_name):
                    continue
                setattr(model_cls, attr_name, attr_value)
            model_cls._di_bench_generation_mixin_patched = True

    if getattr(model, "generation_config", None) is None:
        model_config = getattr(model, "config", None)
        if model_config is not None:
            model.generation_config = GenerationConfig.from_model_config(model_config)
        else:
            model.generation_config = GenerationConfig()


class InternVLLocalAdapter(BaseModelAdapter):
    def __init__(
        self,
        model_path,
        bbox_mode="visual",
        max_new_tokens=256,
        temperature=0.0,
        top_p=1.0,
        dtype="bfloat16",
        device="cuda",
        image_size=448,
        max_tiles_per_image=12,
        source_mode="full",
    ):
        self.model_path = model_path
        self.prompter = BenchmarkPrompter(bbox_mode=bbox_mode, source_mode=source_mode)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self.image_size = image_size
        self.max_tiles_per_image = max_tiles_per_image
        self.torch_dtype = self._resolve_torch_dtype(dtype)

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if self.device != "cpu":
            model_kwargs["use_flash_attn"] = True

        hf_device_map = os.environ.get("HF_DEVICE_MAP", "").strip().lower()
        if self.device != "cpu" and hf_device_map and hf_device_map not in {"none", "single"}:
            model_kwargs["device_map"] = hf_device_map

        self.model = AutoModel.from_pretrained(self.model_path, **model_kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)

        _ensure_generation_mixin(self.model, "InternVL chat model")
        language_model = getattr(self.model, "language_model", None)
        if language_model is not None:
            _ensure_generation_mixin(language_model, "InternVL language model")

        if self.device == "cpu":
            self.model = self.model.to("cpu")
        elif "device_map" not in model_kwargs:
            self.model = self.model.to("cuda")

    def generate(self, item, scene_dir):
        messages = self.prompter.get_messages(str(scene_dir), item)
        question, image_paths = self._build_question_and_images(messages)
        pixel_values, num_patches_list = self._build_pixel_values(image_paths)

        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0 or self.top_p < 1.0,
        }
        if generation_config["do_sample"]:
            generation_config["temperature"] = self.temperature
            generation_config["top_p"] = self.top_p

        model_device = self._get_model_device()
        autocast_context = self._get_autocast_context(model_device)

        chat_kwargs = {
            "tokenizer": self.tokenizer,
            "pixel_values": pixel_values,
            "question": question,
            "generation_config": generation_config,
        }
        if num_patches_list:
            chat_kwargs["num_patches_list"] = num_patches_list

        with torch.no_grad():
            with autocast_context:
                response = self.model.chat(**chat_kwargs)

        text = response[0] if isinstance(response, tuple) else response
        return {
            "text": str(text).strip(),
            "prompt": question,
            "meta": {
                "num_images": len(image_paths),
                "dtype": str(self.torch_dtype),
                "device": str(model_device),
            },
        }

    def _build_question_and_images(self, messages):
        text_parts = []
        image_paths = []

        for message in messages:
            content_blocks = message.get("content", [])
            if isinstance(content_blocks, str):
                if content_blocks.strip():
                    text_parts.append(content_blocks.strip())
                continue

            for block in content_blocks:
                block_type = block.get("type")
                if block_type == "text":
                    block_text = block.get("text", "").strip()
                    if block_text:
                        text_parts.append(block_text)
                    continue
                if block_type == "image":
                    image_path = self._strip_file_scheme(block["image"])
                    image_paths.append(image_path)
                    if len(image_paths) == 1:
                        text_parts.append("<image>")
                    else:
                        text_parts.append(f"Image-{len(image_paths)}: <image>")
                    continue
                if block_type == "video":
                    raise NotImplementedError("Video inputs are not supported in the current InternVL adapter.")
                raise ValueError(f"Unsupported content block type: {block_type}")

        question = "\n".join(part for part in text_parts if part).strip()
        return question, image_paths

    def _build_pixel_values(self, image_paths):
        if not image_paths:
            return None, None

        patch_tensors = []
        num_patches_list = []
        for image_path in image_paths:
            pixel_values = _load_image_pixels(
                image_path,
                input_size=self.image_size,
                max_num=self.max_tiles_per_image,
            )
            num_patches_list.append(int(pixel_values.size(0)))
            patch_tensors.append(pixel_values)

        merged = torch.cat(patch_tensors, dim=0).to(self._get_model_device(), dtype=self.torch_dtype)
        return merged, num_patches_list

    def _get_model_device(self):
        if self.device == "cpu":
            return torch.device("cpu")
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            return model_device
        return torch.device("cuda")

    def _get_autocast_context(self, model_device):
        if model_device.type != "cuda":
            return nullcontext()
        if self.torch_dtype not in {torch.float16, torch.bfloat16}:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.torch_dtype)

    def _strip_file_scheme(self, uri):
        if uri.startswith("file://"):
            return uri[7:]
        return uri

    def _resolve_torch_dtype(self, dtype):
        dtype_key = str(dtype).lower()
        if dtype_key in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype_key in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype_key in {"fp32", "float32", "float"}:
            return torch.float32
        if dtype_key == "auto":
            return torch.bfloat16 if self.device != "cpu" else torch.float32
        raise ValueError(f"Unsupported dtype for InternVL local adapter: {dtype}")
