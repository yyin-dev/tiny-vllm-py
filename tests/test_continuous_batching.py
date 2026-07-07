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


def run_singleton_decode(
    model, prev_token: torch.Tensor, cache: RequestKVCache
) -> tuple[torch.Tensor, RequestKVCache]:
    [decode_token] = model.decode([(prev_token, cache)])
    return decode_token, cache


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


def test_continuous_batching_decode_matches_singletons(local_model_cpu, tokenizer):
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
    batched_decode_tokens = model.decode(
        list(zip(batched_prefill_tokens, batched_caches, strict=True))
    )

    assert len(batched_decode_tokens) == len(prompt_tensors)

    for prompt, batched_decode_token, batched_cache in zip(
        prompt_tensors, batched_decode_tokens, batched_caches, strict=True
    ):
        singleton_prefill_token, singleton_cache = run_singleton_prefill(model, prompt)
        singleton_decode_token, singleton_cache = run_singleton_decode(
            model, singleton_prefill_token, singleton_cache
        )

        assert torch.equal(batched_decode_token, singleton_decode_token)
        assert_request_kv_cache_equal(
            batched_cache,
            singleton_cache,
            num_layers=len(model.layers),
        )


def test_continuous_batching_decode_matches_multiple_singleton_steps(
    local_model_cpu, tokenizer
):
    model = local_model_cpu
    device = "cpu"

    prompts = [
        "Hello!",
        "Hi! How are you?",
    ]
    encoded = tokenizer(prompts)
    prompt_tensors = [torch.tensor([ids], device=device) for ids in encoded.input_ids]

    batched_caches = [RequestKVCache() for _ in prompt_tensors]
    batched_tokens = model.prefill(list(zip(prompt_tensors, batched_caches, strict=True)))

    singleton_states = []
    for prompt in prompt_tensors:
        singleton_token, singleton_cache = run_singleton_prefill(model, prompt)
        singleton_states.append((singleton_token, singleton_cache))

    decode_steps = 3
    for _ in range(decode_steps):
        batched_tokens = model.decode(list(zip(batched_tokens, batched_caches, strict=True)))

        next_singleton_states = []
        for batched_token, batched_cache, (singleton_token, singleton_cache) in zip(
            batched_tokens, batched_caches, singleton_states, strict=True
        ):
            singleton_next_token, singleton_cache = run_singleton_decode(
                model, singleton_token, singleton_cache
            )

            assert torch.equal(batched_token, singleton_next_token)
            assert_request_kv_cache_equal(
                batched_cache,
                singleton_cache,
                num_layers=len(model.layers),
            )
            next_singleton_states.append((singleton_next_token, singleton_cache))

        singleton_states = next_singleton_states
