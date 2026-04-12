"""
Angular steering for vLLM via forward hooks and ``llm.apply_model``.

Adapted from the Angular steering reference implementation (Vu & Nguyen, 2025;
see ``angular-steering/vllm_angular_steering.py`` in this repo's submodule).

Requires ``enforce_eager=True`` on ``vllm.LLM``. Set
``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` in the environment before constructing
``LLM`` when using hook registration (``apply_model``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AngularSteeringOperator:
    """Applies the Angular rotation in the (u1, u2) plane on hidden states."""

    def __init__(self, first_direction: np.ndarray, second_direction: np.ndarray):
        self.first_direction = torch.from_numpy(first_direction).float()
        self.second_direction = torch.from_numpy(second_direction).float()

        self.b1 = self.first_direction / self.first_direction.norm()
        self.b2 = self.second_direction - (self.second_direction @ self.b1) * self.b1
        self.b2 = self.b2 / self.b2.norm()

        self.proj_matrix = torch.outer(self.b1, self.b1) + torch.outer(self.b2, self.b2)

        self._device_cache: Dict[Tuple, Dict[str, torch.Tensor]] = {}
        self._rotation_cache: Dict[Tuple, torch.Tensor] = {}

    def _get_device_tensors(self, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        cache_key = (device, dtype)
        if cache_key not in self._device_cache:
            self._device_cache[cache_key] = {
                "proj_matrix": self.proj_matrix.to(device=device, dtype=dtype),
                "b1": self.b1.to(device=device, dtype=dtype),
                "b2": self.b2.to(device=device, dtype=dtype),
            }
        return self._device_cache[cache_key]

    def _get_rotation_vector(
        self, theta: float, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        theta_normalized = theta % 360
        cache_key = (device, dtype, theta_normalized)

        if cache_key not in self._rotation_cache:
            cached = self._get_device_tensors(device, dtype)
            theta_rad = torch.tensor(theta_normalized * torch.pi / 180.0)
            self._rotation_cache[cache_key] = (
                torch.cos(theta_rad) * cached["b1"]
                + torch.sin(theta_rad) * cached["b2"]
            )

        return self._rotation_cache[cache_key]

    def steer(
        self,
        hidden_states: torch.Tensor,
        target_degree: float,
        adaptive_mode: int = 1,
    ) -> torch.Tensor:
        device = hidden_states.device
        dtype = hidden_states.dtype

        cached = self._get_device_tensors(device, dtype)
        proj_matrix = cached["proj_matrix"]
        first_dir = cached["b1"]

        v_theta = self._get_rotation_vector(target_degree, device, dtype)

        proj_h = hidden_states @ proj_matrix.T
        r = proj_h.norm(dim=-1, keepdim=True)

        if adaptive_mode == 0:
            return hidden_states - proj_h + r * v_theta

        if adaptive_mode == 1:
            alignment = hidden_states @ first_dir
            mask = (alignment > 0).unsqueeze(-1)
            steered = hidden_states - proj_h + r * v_theta
            return torch.where(mask, steered, hidden_states)

        raise ValueError(f"Unknown adaptive_mode: {adaptive_mode}. Supported: 0, 1")

    def clear_cache(self) -> None:
        self._device_cache.clear()
        self._rotation_cache.clear()

    def clear_rotation_cache(self) -> None:
        self._rotation_cache.clear()


def _detect_prefill_decode_phase(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> bool:
    try:
        from vllm.forward_context import get_forward_context

        forward_ctx = get_forward_context()
        attn_metadata = forward_ctx.attn_metadata

        if isinstance(attn_metadata, dict):
            attn_meta = next(iter(attn_metadata.values())) if attn_metadata else None
        else:
            attn_meta = attn_metadata

        if attn_meta is not None:
            max_query_len = getattr(attn_meta, "max_query_len", None)
            if max_query_len is not None:
                return max_query_len == 1

            if hasattr(attn_meta, "num_decode_tokens"):
                return attn_meta.num_decode_tokens > 0

            if hasattr(attn_meta, "num_prefill_tokens"):
                return attn_meta.num_prefill_tokens == 0

    except Exception as e:
        logger.debug("Metadata detection failed for %s: %s", layer_name, e)

    return False


def create_steering_hook(
    operator: AngularSteeringOperator,
    state: Dict[str, Any],
    layer_name: str,
    prompt_only: bool = False,
) -> Callable[..., Any]:
    _layer_name = layer_name
    _initial_operator = operator

    def hook_fn(module, input_tuple, output):
        import builtins

        target_degree = state.get("target_degree", 0.0)
        adaptive_mode = state.get("adaptive_mode", 1)
        enabled = state.get("enabled", True)

        if not enabled:
            return output

        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        if prompt_only:
            is_decode = _detect_prefill_decode_phase(hidden_states, _layer_name)
            if is_decode:
                return output

        current_operator = getattr(builtins, "_steering_operator", _initial_operator)

        last_theta = state.get("last_theta", None)
        if last_theta is not None and last_theta != target_degree:
            current_operator.clear_rotation_cache()
        state["last_theta"] = target_degree

        steered = current_operator.steer(
            hidden_states=hidden_states,
            target_degree=target_degree,
            adaptive_mode=adaptive_mode,
        )

        if rest is not None:
            return (steered,) + rest
        return steered

    return hook_fn


def clear_hooks(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if hasattr(module, "_forward_hooks") and module._forward_hooks:
            count += len(module._forward_hooks)
            module._forward_hooks.clear()
    return count


def _layer_config_to_directions(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Support AbstentionBench ``extract_angular`` npy entries (``mode``, directions)."""
    if "first_direction" not in config or "second_direction" not in config:
        raise KeyError(
            "Each layer config must contain 'first_direction' and 'second_direction' "
            f"(keys found: {list(config.keys())})"
        )
    return config["first_direction"], config["second_direction"]


class AngularSteering:
    """Load ``*_steering_config.npy``, register hooks via ``llm.apply_model``."""

    def __init__(self, llm):
        self.llm = llm
        self.steering_configs: Dict[str, AngularSteeringOperator] = {}
        self.hooks_registered = False

        self._target_degree = 0.0
        self._adaptive_mode = 1
        self._enabled = False

    @staticmethod
    def load_steering_config_from_npy(config_file: str) -> Dict[str, Dict[str, Any]]:
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Steering config not found: {config_file}")

        config_dict = np.load(str(config_path), allow_pickle=True).item()
        logger.info("Loaded steering config from %s (%d layer entries)", config_file, len(config_dict))
        return config_dict

    def load_config_from_file(self, config_file: str) -> None:
        config_dict = self.load_steering_config_from_npy(config_file)

        self.steering_configs = {}
        for layer_name, cfg in config_dict.items():
            u1, u2 = _layer_config_to_directions(cfg)
            self.steering_configs[layer_name] = AngularSteeringOperator(u1, u2)

        logger.info("Created %d Angular steering operators", len(self.steering_configs))

    def apply_steering(
        self,
        target_degree: float = 0.0,
        adaptive_mode: int = 1,
        prompt_only: bool = False,
    ) -> Dict[str, Any]:
        if not self.steering_configs:
            raise ValueError("No steering configurations loaded. Call load_config_from_file() first.")

        self._target_degree = target_degree
        self._adaptive_mode = adaptive_mode
        self._enabled = True

        first_layer_name = next(iter(self.steering_configs))
        shared_operator = self.steering_configs[first_layer_name]
        target_layers = list(self.steering_configs.keys())

        def register_hooks_fn(model: nn.Module) -> int:
            import builtins

            if not hasattr(builtins, "_steering_state"):
                builtins._steering_state = {}

            builtins._steering_state["target_degree"] = target_degree
            builtins._steering_state["adaptive_mode"] = adaptive_mode
            builtins._steering_state["enabled"] = True
            builtins._steering_state["is_first_pass"] = True
            builtins._steering_state["last_theta"] = None
            builtins._steering_operator = shared_operator

            clear_hooks(model)

            count = 0
            module_dict = dict(model.named_modules())

            for layer_name in target_layers:
                if layer_name in module_dict:
                    module = module_dict[layer_name]
                    hook = create_steering_hook(
                        operator=shared_operator,
                        state=builtins._steering_state,
                        layer_name=layer_name,
                        prompt_only=prompt_only,
                    )
                    module.register_forward_hook(hook)
                    count += 1

            return count

        results = self.llm.apply_model(register_hooks_fn)
        self.hooks_registered = True

        logger.info(
            "Registered Angular steering hooks: %s (degree=%s, adaptive_mode=%s, prompt_only=%s)",
            results,
            target_degree,
            adaptive_mode,
            prompt_only,
        )
        return {"hooks_registered": results}

    def update_steering(
        self,
        target_degree: Optional[float] = None,
        adaptive_mode: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        if not self.hooks_registered:
            raise ValueError("Hooks not registered. Call apply_steering() first.")

        if target_degree is not None:
            self._target_degree = target_degree
        if adaptive_mode is not None:
            self._adaptive_mode = adaptive_mode
        if enabled is not None:
            self._enabled = enabled

        def update_state_fn(model: nn.Module) -> bool:
            import builtins

            if hasattr(builtins, "_steering_state"):
                if target_degree is not None:
                    builtins._steering_state["target_degree"] = target_degree
                if adaptive_mode is not None:
                    builtins._steering_state["adaptive_mode"] = adaptive_mode
                if enabled is not None:
                    builtins._steering_state["enabled"] = enabled
            return True

        self.llm.apply_model(update_state_fn)

    def remove_steering(self) -> None:
        def remove_hooks_fn(model: nn.Module) -> int:
            return clear_hooks(model)

        count = self.llm.apply_model(remove_hooks_fn)
        self.hooks_registered = False
        self._enabled = False
        logger.info("Removed %s Angular steering hooks", count)

    def set_degree(self, target_degree: float) -> None:
        self.update_steering(target_degree=target_degree, enabled=True)

    @property
    def target_degree(self) -> float:
        return self._target_degree

    @property
    def adaptive_mode(self) -> int:
        return self._adaptive_mode

    @property
    def enabled(self) -> bool:
        return self._enabled
