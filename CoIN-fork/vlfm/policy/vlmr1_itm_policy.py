from __future__ import annotations

import os
from typing import Any

import cv2

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
        Throttled value map update para VLM-R1.
        Quando há detecções não confirmadas no detection_cloud, injeta score alto
        na posição dessas detecções para guiar a exploração em direção a elas.
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
                pass
            return

        # Primeiro faz o update normal via ITM adapter
        super()._update_value_map()

        # Depois injeta score alto para detecções não confirmadas
        # Isso garante que o agente explore em direção a objetos vistos mas não confirmados
        try:
            obj_map = getattr(self, "_object_map", None)
            if obj_map is None:
                return

            target = getattr(self, "_target_object", "").split("|")[0].strip().lower()
            if not target:
                return

            detection_cloud = getattr(obj_map, "detection_cloud", {})
            if target not in detection_cloud:
                return

            cloud = detection_cloud[target]
            if cloud is None or len(cloud) == 0:
                return

            # Pega a posição 2D média das detecções não confirmadas
            positions_2d = cloud[:, :2]  # (N, 2) — x, y no frame episódico
            centroid = positions_2d.mean(axis=0)  # (2,)

            # Injeta um score alto no value map nessa posição
            # Simulamos um depth frame artificial apontando para o centróide
            robot_xy = self._observations_cache.get("robot_xy")
            if robot_xy is None:
                return

            # Usa o value map diretamente para marcar a posição como valiosa
            # Convertendo coordenadas do mundo para pixels do mapa
            map_size = self._value_map.size
            ppm = self._value_map.pixels_per_meter
            origin = map_size // 2

            px = int(origin + centroid[0] * ppm)
            py = int(origin - centroid[1] * ppm)  # y invertido no mapa

            px = max(0, min(map_size - 1, px))
            py = max(0, min(map_size - 1, py))

            # Marca uma região ao redor do centróide com score alto
            radius_px = max(5, int(1.0 * ppm))  # 1 metro de raio
            cv2.circle(
                self._value_map._value_map[:, :, 0],
                (px, py),
                radius_px,
                0.85,  # score alto mas não máximo
                -1,   # filled
            )

        except Exception as e:
            print(f"[VLMr1ITMPolicy] detection_cloud injection error: {e}")

