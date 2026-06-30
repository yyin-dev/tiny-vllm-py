from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DebugCollector:
    records: dict[str, Any] = field(default_factory=dict)
    stage: str = "unset"

    def set_prefill(self) -> None:
        self.stage = "prefill"

    def set_decode_step(self, step: int) -> None:
        self.stage = f"decode{step}"

    def record(self, layer_idx: int, name: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            value = value.detach().clone()

        self.records[self._key(layer_idx, name)] = value

    def get(self, stage: str, layer_idx: int, name: str) -> Any:
        return self.records[f"{stage}.{layer_idx}.{name}"]

    def _key(self, layer_idx: int, name: str) -> str:
        return f"{self.stage}.{layer_idx}.{name}"
