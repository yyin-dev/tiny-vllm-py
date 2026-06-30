import torch
from torch import Tensor
from jaxtyping import Float


class LayerKVCache:
    def __init__(self):
        self.ks = torch.tensor([])
        self.vs = torch.tensor([])

    def append_kvs(
        self,
        ks: Float[Tensor, "... num_heads seq_len head_dim"],
        vs: Float[Tensor, "... num_heads seq_len head_dim"],
    ):
        assert ks.shape[-2] == vs.shape[-2]

        if self.ks.numel() == 0:
            self.ks = ks
            self.vs = vs
        else:
            self.ks = torch.cat([self.ks, ks], dim=-2)
            self.vs = torch.cat([self.vs, vs], dim=-2)

    def get_kv_prefix(
        self,
    ) -> tuple[
        Float[Tensor, "... num_heads seq_len head_dim"],
        Float[Tensor, "... num_heads seq_len head_dim"],
    ]:
        return (self.ks, self.vs)

    def current_length(self) -> int:
        if self.ks.numel() == 0:
            return 0

        return self.ks.shape[-2]

    def is_empty(self) -> bool:
        return self.current_length() == 0


class KVCache:
    def __init__(self):
        self.layers: dict[int, LayerKVCache] = {}

    def append_kvs(
        self,
        layer_idx,
        ks: Float[Tensor, "... num_heads seq_len head_dim"],
        vs: Float[Tensor, "... num_heads seq_len head_dim"],
    ):
        if layer_idx not in self.layers:
            self.layers[layer_idx] = LayerKVCache()

        self.layers[layer_idx].append_kvs(ks, vs)

    def get_kv_prefix(self, layer_idx: int) -> tuple[
        Float[Tensor, "... num_heads seq_len head_dim"],
        Float[Tensor, "... num_heads seq_len head_dim"],
    ]:
        if layer_idx not in self.layers:
            self.layers[layer_idx] = LayerKVCache()

        return self.layers[layer_idx].get_kv_prefix()

    def current_length(self, layer_idx) -> int:
        if layer_idx not in self.layers:
            self.layers[layer_idx] = LayerKVCache()

        return self.layers[layer_idx].current_length()

    def is_empty(self, layer_idx) -> bool:
        if layer_idx not in self.layers:
            self.layers[layer_idx] = LayerKVCache()

        return self.layers[layer_idx].is_empty()
