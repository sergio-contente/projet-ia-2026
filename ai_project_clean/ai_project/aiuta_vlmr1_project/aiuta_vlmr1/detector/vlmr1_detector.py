"""vlmr1_detector.py — VLM-R1 detector adapter.
Refactored from benchmark_ovd.py::run_single_inference()."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import AbstractDetector, Detection, DetectionResult
from .output_parser import OutputParser
from .prompt_templates import PromptBuilder
from ..utils.model_loader import ModelLoader


class VLMr1Detector(AbstractDetector):
    def __init__(self, config):
        self._config = config
        self._loader = ModelLoader.get_instance(config.model)

    def _run_forward(
        self,
        messages: list,
    ) -> tuple[str, DetectionResult]:
        from qwen_vl_utils import process_vision_info

        proc = self._loader.processor
        t_pre0 = time.perf_counter()
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(messages)
        inputs = proc(
            text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt"
        ).to(self._loader.device)
        t_pre = time.perf_counter() - t_pre0

        t_gen0 = time.perf_counter()
        with torch.inference_mode():
            gen = self._loader.model.generate(
                **inputs,
                max_new_tokens=self._config.model.max_new_tokens,
                do_sample=False,
            )
        t_gen = time.perf_counter() - t_gen0

        t_parse0 = time.perf_counter()
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        raw = proc.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        parsed = OutputParser.parse_full(raw)
        dets = []
        if parsed.bboxes:
            for item in parsed.bboxes:
                if isinstance(item, dict) and "bbox_2d" in item and "label" in item:
                    try:
                        dets.append(Detection(
                            bbox=[float(b) for b in item["bbox_2d"]],
                            label=item["label"].strip().lower(),
                            reasoning=parsed.reasoning_text,
                        ))
                    except (ValueError, TypeError):
                        pass
        t_parse = time.perf_counter() - t_parse0
        total = t_pre + t_gen + t_parse

        return raw, DetectionResult(
            detections=dets,
            raw_output=raw,
            reasoning_text=parsed.reasoning_text,
            json_valid=parsed.json_valid,
            latency_sec=total,
            preprocess_latency_sec=t_pre,
            generate_latency_sec=t_gen,
            parse_latency_sec=t_parse,
        )

    def detect(self, image_path: str, target_categories: list[str],
               kg_context: str | None = None) -> DetectionResult:
        builder = PromptBuilder().set_categories(target_categories)
        if kg_context:
            builder.set_kg_context(kg_context).set_scene_type("indoor")
        system, user_prompt = builder.build()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{Path(image_path).resolve()}"},
                {"type": "text", "text": user_prompt},
            ]},
        ]
        _, result = self._run_forward(messages)
        return result

    def detect_from_observation(self, observation: np.ndarray,
                                 target_categories: list[str],
                                 kg_context: str | None = None) -> DetectionResult:
        pil_image = Image.fromarray(observation.astype(np.uint8))
        builder = PromptBuilder().set_categories(target_categories)
        if kg_context:
            builder.set_kg_context(kg_context).set_scene_type("indoor")
        system, user_prompt = builder.build()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": user_prompt},
            ]},
        ]
        _, result = self._run_forward(messages)
        return result
