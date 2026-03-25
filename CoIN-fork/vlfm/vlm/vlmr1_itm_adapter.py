from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from aiuta_vlmr1.utils.model_loader import ModelLoader


class VLMr1ITMAdapter:
    """
    Adapter em formato semelhante ao `BLIP2ITMClient`.

    Interface:
      - `.cosine(image: np.ndarray, txt: str) -> float`

    Implementa um VLM-R1 forward pass com `max_new_tokens=1` e devolve
    `P(token="yes")` como score ITM.
    """

    def __init__(self, *, model_config: Any) -> None:
        loader = ModelLoader.get_instance(model_config)
        self._loader = loader
        self._model = loader.model
        self._processor = loader.processor
        self._device = loader.device

        # Cache dos ids de tokens para "yes/no" (com variações de tokenização).
        tokenizer = getattr(self._processor, "tokenizer", None)
        self._token_id_yes = None
        if tokenizer is not None:
            yes_candidates = [" yes", "Yes", " yes,", " yes.", "Yes,"]
            ids: list[int] = []
            for c in yes_candidates:
                enc = tokenizer.encode(c, add_special_tokens=False)
                if enc:
                    ids.append(int(enc[0]))
            self._token_id_yes = ids[0] if ids else None

    def _to_uint8_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image
        img = image
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        Score ITM em [0,1], com fallback robusto:
        - se falhar no forward pass => retorna 0.5
        """
        try:
            if self._token_id_yes is None:
                # Não conseguimos calcular probabilidade com precisão.
                return 0.5

            rgb = self._to_uint8_rgb(image)
            pil_image = Image.fromarray(rgb)

            question = f"Does this image show {txt}? Answer only yes or no."
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers with only yes or no.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": question},
                    ],
                },
            ]

            proc = self._processor
            from qwen_vl_utils import process_vision_info

            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            img_in, vid_in = process_vision_info(messages)

            inputs = proc(
                text=[text],
                images=img_in,
                videos=vid_in,
                padding=True,
                return_tensors="pt",
            ).to(self._device)

            with torch.inference_mode():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                if not getattr(out, "scores", None):
                    return 0.5

                # `scores[0]` corresponde ao passo de geração 1.
                logits = out.scores[0][0]  # (vocab,)
                probs = F.softmax(logits, dim=-1)
                prob_yes = float(probs[self._token_id_yes].detach().cpu().item())
                # Normalização defensiva.
                if not np.isfinite(prob_yes):
                    return 0.5
                return float(max(0.0, min(1.0, prob_yes)))
        except Exception:
            return 0.5

