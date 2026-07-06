import os
import sys

import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from kv_cache import RequestKVCache
from util import assert_allclose_tight


def assert_request_kv_cache_equal(
    actual: RequestKVCache,
    expected: RequestKVCache,
    num_layers: int,
) -> None:
    for layer_idx in range(num_layers):
        actual_k, actual_v = actual.get_kv_prefix(layer_idx)
        expected_k, expected_v = expected.get_kv_prefix(layer_idx)

        assert actual_k.shape == expected_k.shape
        assert actual_v.shape == expected_v.shape
        assert_allclose_tight(actual_k, expected_k)
        assert_allclose_tight(actual_v, expected_v)


def run_singleton_prefill(
    model, prompt: torch.Tensor
) -> tuple[torch.Tensor, RequestKVCache]:
    cache = RequestKVCache()
    [prefill_token] = model.prefill([(prompt, cache)])
    return prefill_token, cache


# Verify that running in batch matches running as singletons
def test_continuous_batching_prefill_matches_singletons(local_model_cpu, tokenizer):
    model = local_model_cpu
    device = "cpu"

    prompts = [
        "Hi!",
        "A really really really really long prompt",
    ]
    encoded = tokenizer(prompts)
    prompt_tensors = [torch.tensor([ids], device=device) for ids in encoded.input_ids]

    batched_caches = [RequestKVCache() for _ in prompt_tensors]
    batched_prefill_tokens = model.prefill(
        list(zip(prompt_tensors, batched_caches, strict=True))
    )

    assert len(batched_prefill_tokens) == len(prompt_tensors)

    for prompt, batched_token, batched_cache in zip(
        prompt_tensors, batched_prefill_tokens, batched_caches, strict=True
    ):
        singleton_token, singleton_cache = run_singleton_prefill(model, prompt)

        assert torch.equal(batched_token, singleton_token)
        assert_request_kv_cache_equal(
            batched_cache,
            singleton_cache,
            num_layers=len(model.layers),
        )
