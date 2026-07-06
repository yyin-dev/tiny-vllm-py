from abc import ABC, abstractmethod
from kv_cache import RequestKVCache
import torch


class EngineModel(ABC):
    """
    Model that can be used by [Engine] for continuous batching.
    """

    @abstractmethod
    def prefill(
        self, prompts_and_kvs: list[tuple[torch.Tensor, RequestKVCache]]
    ) -> list[torch.Tensor]:
        """
        Each prompt tensor should have shape (1, seq_len).
        Returns a (1, 1) tensor, representing one token, for each item in the input list.

        `prefill` appends k/v to `RequestKVCache`.
        """
        pass

    @abstractmethod
    def decode(
        self, prev_tokens_and_kvs: list[tuple[torch.Tensor, RequestKVCache]]
    ) -> list[torch.Tensor]:
        """
        Each input tensor should have shape (1, 1).
        Returns a (1, 1) tensor, representing one token, for each item in the input list.

        `decode` appends new k/v to `RequestKVCache`.
        """
        pass
