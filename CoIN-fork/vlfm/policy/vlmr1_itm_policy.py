from __future__ import annotations

import os
from typing import Any

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

    def _update_value_map(self) -> None:
        """
        ITMPolicyV2 chama _update_value_map() em TODOS os steps.

        Em VLM-R1, isso é caro; então aplicamos throttling e só atualizamos a
        trajetória do agente nos steps intermediários.
        """
        update_every = int(os.environ.get("VLMR1_VALUE_MAP_UPDATE_EVERY", "3"))
        update_every = max(1, update_every)

        if getattr(self, "_num_steps", 0) % update_every != 0:
            try:
                self._value_map.update_agent_traj(
                    self._observations_cache["robot_xy"],
                    self._observations_cache["robot_heading"],
                )
            except Exception:
                # Evita crash no caso de cache incompleta durante warmup.
                pass
            return

        super()._update_value_map()

