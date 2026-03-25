from __future__ import annotations

from typing import Any

from vlfm.mapping.vlmr1_frontier_map import VLMr1FrontierMap
from vlfm.policy.itm_policy import ITMPolicyV2
from vlfm.vlm.vlmr1_itm_adapter import VLMr1ITMAdapter


class VLMNavITMPolicy(ITMPolicyV2):
    """
    Versão do ITMPolicyV2 que usa VLM-R1 local (via adapter) em vez do BLIP2ITM (porta 12182).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if not getattr(self, "_use_vlmr1", False):
            return

        vlmr1_bridge = getattr(self, "_vlmr1_bridge", None)
        if vlmr1_bridge is None:
            return

        model_config = getattr(vlmr1_bridge.config, "model", None)
        if model_config is None:
            return

        adapter = VLMr1ITMAdapter(model_config=model_config)
        self._itm = adapter

        # Pode não ser usado no ITMPolicyV2, mas mantém compatibilidade caso outro caminho
        # dependa de `_frontier_map`.
        self._frontier_map = VLMr1FrontierMap(adapter)

