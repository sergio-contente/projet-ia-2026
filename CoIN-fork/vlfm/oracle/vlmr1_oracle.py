# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from PIL import Image

from aiuta_vlmr1.utils.model_loader import ModelLoader


class VLMr1Oracle:
    """
    Replaces VLMOracle (LLaVA) when running with VLM-R1.
    Answers yes/no/IDK questions about the instance_imagegoal using the same
    VLM as the detector via ModelLoader singleton.
    Interface: set_instance_image(img, target) + answer(question) -> str
    """

    def __init__(self) -> None:
        self._instance_image: np.ndarray | None = None
        self._target_object: str = ""
        self._loader = None  # lazy — loaded when needed

    def set_instance_image(self, instance_image: np.ndarray, target_object: str) -> None:
        self._instance_image = instance_image.astype(np.uint8)
        self._target_object = target_object
        print(f"[VLMr1Oracle] Instance image set for target: {target_object}")

    def answer(self, question: str) -> str:
        """Answer a question about the instance_imagegoal. Returns 'yes', 'no', or 'I don't know'."""
        if self._instance_image is None:
            return "I don't know"
        try:
            if self._loader is None:
                if ModelLoader._instances:
                    key = next(iter(ModelLoader._instances))
                    self._loader = ModelLoader._instances[key]
                else:
                    return "I don't know"

            pil_image = Image.fromarray(self._instance_image)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                pil_image.save(tmp_path, format="JPEG", quality=95)
                img_url = f"file://{os.path.abspath(tmp_path)}"

                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer only with yes, no, or I don't know.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": img_url,
                                "min_pixels": 256 * 28 * 28,
                                "max_pixels": 512 * 28 * 28,
                            },
                            {"type": "text", "text": question},
                        ],
                    },
                ]

                from qwen_vl_utils import process_vision_info

                proc = self._loader.processor
                text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                img_in, vid_in = process_vision_info(messages)
                inputs = proc(
                    text=[text],
                    images=img_in,
                    videos=vid_in,
                    padding=True,
                    return_tensors="pt",
                ).to(self._loader.device)

                with torch.inference_mode():
                    gen = self._loader.model.generate(
                        **inputs,
                        max_new_tokens=16,
                        do_sample=False,
                        use_cache=False,
                    )

                trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, gen)]
                raw = proc.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                raw_l = raw.strip().lower()

                print(f"[VLMr1Oracle] Q: {question!r} → A: {raw_l!r}")

                if raw_l.startswith("yes"):
                    return "yes"
                if raw_l.startswith("no"):
                    return "no"
                return "I don't know"
            finally:
                try:
                    os.unlink(tmp_path)
                except FileNotFoundError:
                    pass
        except Exception as e:
            print(f"[VLMr1Oracle] Error: {e}")
            return "I don't know"

    def reset(self) -> None:
        self._instance_image = None
        self._target_object = ""
