from __future__ import annotations

import os
from typing import Any

import numpy as np

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
            print(f"[DEBUG inject] obj_map={type(obj_map).__name__ if obj_map else None}")
            if obj_map is None:
                return
            target = getattr(self, "_target_object", "").split("|")[0].strip().lower()
            detection_cloud = getattr(obj_map, "detection_cloud", {})
            print(f"[DEBUG inject] target={target!r}, dc_keys={list(detection_cloud.keys())}")
            if target not in detection_cloud:
                return
            cloud = detection_cloud[target]
            if cloud is None or len(cloud) == 0:
                return
            print(f"[DEBUG inject] cloud shape={cloud.shape}, injecting...")
            positions_2d = cloud[:, :2]
            centroid = positions_2d.mean(axis=0)
            map_size = self._value_map.size
            ppm = self._value_map.pixels_per_meter
            # Usar a mesma fórmula de sort_waypoints para garantir consistência
            # sort_waypoints faz:
            #   px = int(-x * ppm) + origin[0]  → row no array (flipud)
            #   py = int(-y * ppm) + origin[1]  → col no array
            #   point_px = (value_map.shape[0] - px, py)
            origin_arr = self._value_map._episode_pixel_origin  # array [row_origin, col_origin]
            row_in_map = int(self._value_map._value_map.shape[0] - (int(-centroid[0] * ppm) + origin_arr[0]))
            col_in_map = int(int(-centroid[1] * ppm) + origin_arr[1])
            row_in_map = max(0, min(map_size - 1, row_in_map))
            col_in_map = max(0, min(map_size - 1, col_in_map))

            # Injeção com gradiente gaussiano — score decresce com distância ao centróide
            # Isso preserva informação direcional: frontier mais próximo da cama tem score maior
            # Criar grade de coordenadas do mapa
            ys, xs = np.ogrid[0:map_size, 0:map_size]
            # Distância de cada pixel ao centróide em pixels
            dist_px = np.sqrt((xs - col_in_map) ** 2 + (ys - row_in_map) ** 2)
            # Sigma = raio de injeção em pixels (controla suavidade do gradiente)
            sigma_px = max(30, int(2.0 * ppm))  # 2 metros de sigma
            # Score gaussiano: 0.85 no centro, decai para 0 nas bordas
            gaussian = 0.85 * np.exp(-(dist_px**2) / (2 * sigma_px**2))
            # Aplicar apenas onde o novo score é maior que o existente (não sobrescrever stop confirmados)
            current = self._value_map._value_map[:, :, 0]
            self._value_map._value_map[:, :, 0] = np.maximum(current, gaussian.astype(np.float32))
            print(f"[DEBUG inject] gaussian sigma={sigma_px}px, max_injected={gaussian.max():.3f}")
        except Exception as e:
            print(f"[VLMr1ITMPolicy] detection_cloud injection error: {e}")
            import traceback; traceback.print_exc()

    def _sort_frontiers_by_value(self, observations, frontiers):
        print(f"[DEBUG frontiers] n={len(frontiers)}, frontiers={frontiers[:3]}")
        print(f"[DEBUG frontiers] detection centroid injetado anteriormente")
        result = super()._sort_frontiers_by_value(observations, frontiers)
        print(f"[DEBUG frontiers] best_value={result[1][0] if result[1] else 'N/A'}")
        return result

