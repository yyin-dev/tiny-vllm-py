"""
We can define inference correctness based on two results:
- model.forward() output logits
- model.generate() output tokens

There are tradeoffs in checking correctness using logits vs. tokens:

- generate() captures the e2e behavior, including details like maintaining KV
cache, maintaining batch states, constructing attn mask, etc.

- forward() outputs the single-step logits so it's a more direct comparison
of model behavior, without being affected by orchestration logic in generate().

Output tokens = model forward + generation loop w/ state management.

There's value in checking both forward() and generate() results. For example, if
forward() logits are close but output tokens mismatch, it implies that the issue
is in the generation loop.

Checking logits is easy: just run forward() on local and reference model for
one or more steps. On my M1 MackBook Air, MPS seems to introduce meaningful
precision error, so I sometimes run on CPU when comparing logits.

Checking tokens is more complicated. HuggingFace's generate() potentially
contains other quirks that differ from vanilla implementation, making
generate() a less reliable correctness reference. To mitigate this, we could
construct the correctness oracle by manually implementing the generation loop
with HF's forward().
"""

import torch
import util

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

MAX_ABS_DIFF_THRESHOLD = 0.25
MEAN_ABS_DIFF_THRESHOLD = 0.1
TIGHT_MAX_ABS_DIFF_THRESHOLD = 1e-4
TIGHT_MEAN_ABS_DIFF_THRESHOLD = 1e-5


def print_diff(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape == b.shape:
        print(
            f"{name}: max_abs={util.max_abs_diff(a, b):.6f}, mean_abs={util.mean_abs_diff(a, b):.6f}, shape={a.shape}"
        )
    else:
        print(f"{name}: shape mismatch! {a.shape} != {b.shape}")


def assert_allclose(
    a: torch.Tensor,
    b: torch.Tensor,
    max_diff_tolerance,
    mean_diff_tolerance,
):
    assert a.shape == b.shape
    assert util.max_abs_diff(a, b) < max_diff_tolerance
    assert util.mean_abs_diff(a, b) < mean_diff_tolerance


def assert_allclose_loose(a: torch.Tensor, b: torch.Tensor):
    assert_allclose(
        a,
        b,
        max_diff_tolerance=MAX_ABS_DIFF_THRESHOLD,
        mean_diff_tolerance=MEAN_ABS_DIFF_THRESHOLD,
    )


def assert_allclose_tight(a: torch.Tensor, b: torch.Tensor):
    assert_allclose(
        a,
        b,
        max_diff_tolerance=TIGHT_MAX_ABS_DIFF_THRESHOLD,
        mean_diff_tolerance=TIGHT_MEAN_ABS_DIFF_THRESHOLD,
    )


def append_kvs(
    kvs: list[tuple[torch.Tensor, torch.Tensor]],
    new_kvs: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    updated = []
    for (k_prefix, v_prefix), (new_k, new_v) in zip(kvs, new_kvs, strict=True):
        updated.append(
            (
                torch.cat([k_prefix, new_k], dim=-2),
                torch.cat([v_prefix, new_v], dim=-2),
            )
        )
    return updated


def test_milestone1_single_step_logits(local_model, reference_model, tokenizer, device):
    for input in [
        ["How are you?"],
        ["Hi!"],
        ["A really really really really long prompt"],
    ]:
        encoding = tokenizer(input)
        encoded = torch.tensor(encoding.input_ids, device=device)

        last_logits_local = local_model(encoded)[:, -1]
        last_logits_reference = reference_model(encoded).logits[:, -1]

        # check argmax
        max_prob_token_local = torch.argmax(last_logits_local, dim=-1).item()
        max_prob_token_reference = torch.argmax(last_logits_reference, dim=-1).item()
        assert max_prob_token_local == max_prob_token_reference

        # check numerical close
        assert_allclose_loose(last_logits_local, last_logits_reference)


def test_milestone1_multi_steps_logits(local_model, reference_model, tokenizer, device):
    input = ["How are you?"]
    batch_encoding = tokenizer(input)
    encoded = torch.tensor(batch_encoding.input_ids, device=device)

    steps = 8
    for _ in range(steps):
        last_logits_local = local_model(encoded)[:, -1]
        last_logits_reference = reference_model(encoded).logits[:, -1]

        max_prob_token_local = torch.argmax(last_logits_local, dim=-1).item()
        max_prob_token_reference = torch.argmax(last_logits_reference, dim=-1).item()
        assert max_prob_token_local == max_prob_token_reference
        assert_allclose_loose(last_logits_local, last_logits_reference)

        encoded = torch.cat(
            [encoded, torch.tensor([[max_prob_token_local]], device=device)], dim=-1
        )


def test_milestone1_multi_step_generate(
    local_model, reference_model, tokenizer, device
):
    input = ["How are you?"]
    batch_encoding = tokenizer(input)
    encoded = torch.tensor(batch_encoding.input_ids, device=device)  # (1, seq_len)
    prompt_length = encoded.shape[-1]

    max_new_tokens = 8
    local_res = local_model.generate(
        encoded,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        use_kv_cache=False,
    )

    ref_res = reference_model.generate(
        encoded,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        use_cache=False,
    )[:, prompt_length:]

    print("Local result:", local_res)
    print("Ref result:", ref_res)
    assert torch.equal(local_res, ref_res)


def test_milestone2_kv_cache_multi_steps_logits_self_consistency(
    local_model_cpu, tokenizer
):
    model = local_model_cpu
    device = "cpu"

    for input in [
        ["How are you?"],
        ["A really really really really long prompt"],
    ]:
        batch_encoding = tokenizer(input)
        encoded = torch.tensor(batch_encoding.input_ids, device=device)

        # Reference: without kv cache
        ref_prefill_logits, _ = model(encoded)
        ref_first_step_last_logits = ref_prefill_logits[:, -1]
        ref_first_step_token = torch.argmax(ref_first_step_last_logits, dim=-1).item()

        # With cache: Prefill
        prefill_logits, kvs = model(encoded, kvs=None)
        prefill_last_logits = prefill_logits[:, -1]
        prefill_max_prob_token = torch.argmax(prefill_last_logits, dim=-1).item()

        # compare prefill result
        assert ref_first_step_token == prefill_max_prob_token
        assert_allclose_tight(ref_first_step_last_logits, prefill_last_logits)

        # decode
        steps = 8
        curr_prompt_and_decode_res = torch.cat(
            (encoded, torch.tensor([[ref_first_step_token]], device=device)), dim=-1
        )
        next_decode_step_input = torch.tensor([[prefill_max_prob_token]], device=device)
        for _ in range(steps):
            # without cache
            ref_logits, _ = model(curr_prompt_and_decode_res)
            ref_last_logits = ref_logits[:, -1]
            ref_max_prob_token = torch.argmax(ref_last_logits, dim=-1).item()

            # with cache
            decode_step_logits, new_kvs = model(next_decode_step_input, kvs=kvs)
            decode_step_last_logits = decode_step_logits[:, -1]
            kvs = append_kvs(kvs, new_kvs)

            decode_step_max_prob_token = torch.argmax(
                decode_step_last_logits, dim=-1
            ).item()

            assert ref_max_prob_token == decode_step_max_prob_token
            assert_allclose_tight(ref_last_logits, decode_step_last_logits)

            curr_prompt_and_decode_res = torch.cat(
                [
                    curr_prompt_and_decode_res,
                    torch.tensor([[ref_max_prob_token]], device=device),
                ],
                dim=-1,
            )
            next_decode_step_input = torch.tensor(
                [[decode_step_max_prob_token]], device=device
            )


def test_milestone2_kv_cache_multi_steps_logits(
    local_model, reference_model, tokenizer, device
):
    for input in [
        ["How are you?"],
        ["A really really really really long prompt"],
    ]:
        batch_encoding = tokenizer(input)
        encoded = torch.tensor(batch_encoding.input_ids, device=device)

        with torch.no_grad():
            # With cache: prefill.
            prefill_logits, kvs = local_model(encoded, kvs=None)
            prefill_last_logits = prefill_logits[:, -1]

            # Reference prefill: full forward on the same logical prefix.
            ref_prefill_last_logits = reference_model(encoded).logits[:, -1]

            prefill_max_prob_token = torch.argmax(prefill_last_logits, dim=-1).item()
            ref_prefill_max_prob_token = torch.argmax(
                ref_prefill_last_logits, dim=-1
            ).item()

            assert prefill_max_prob_token == ref_prefill_max_prob_token
            assert_allclose_loose(prefill_last_logits, ref_prefill_last_logits)

            # Keep the full logical prefix for the HF oracle and feed only the
            # newest token into the local cached decode path.
            curr_prompt_and_decode_res = torch.cat(
                (encoded, torch.tensor([[ref_prefill_max_prob_token]], device=device)),
                dim=-1,
            )
            next_decode_step_input = torch.tensor(
                [[prefill_max_prob_token]], device=device
            )

            steps = 4
            for _ in range(steps):
                # Reference: recompute logits on the full logical prefix.
                ref_last_logits = reference_model(curr_prompt_and_decode_res).logits[
                    :, -1
                ]
                ref_max_prob_token = torch.argmax(ref_last_logits, dim=-1).item()

                # Local cached decode: consume only the newest token.
                decode_step_logits, new_kvs = local_model(
                    next_decode_step_input, kvs=kvs
                )
                decode_step_last_logits = decode_step_logits[:, -1]
                kvs = append_kvs(kvs, new_kvs)
                decode_step_max_prob_token = torch.argmax(
                    decode_step_last_logits, dim=-1
                ).item()

                assert ref_max_prob_token == decode_step_max_prob_token

                assert_allclose_loose(
                    ref_last_logits,
                    decode_step_last_logits,
                )

                curr_prompt_and_decode_res = torch.cat(
                    [
                        curr_prompt_and_decode_res,
                        torch.tensor([[ref_max_prob_token]], device=device),
                    ],
                    dim=-1,
                )
                next_decode_step_input = torch.tensor(
                    [[decode_step_max_prob_token]], device=device
                )


def test_milestone2_kv_cache_generate(local_model, reference_model, tokenizer, device):
    input = ["How are you?"]
    batch_encoding = tokenizer(input)
    encoded = torch.tensor(batch_encoding.input_ids, device=device)
    attention_mask = torch.ones_like(encoded)
    prompt_length = encoded.shape[-1]

    max_new_tokens = 16

    with torch.no_grad():
        local_res = local_model.generate(
            encoded,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            use_kv_cache=True,
        )

        ref_res = reference_model.generate(
            encoded,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            use_cache=True,
        )[:, prompt_length:]

    print(local_res)
    print(ref_res)
    assert local_res.shape == ref_res.shape
    assert torch.equal(local_res.to(ref_res.device), ref_res)


def test_milestone3_static_batching_singleton_no_cache_logits(
    local_model_cpu, tokenizer
):
    model = local_model_cpu
    device = "cpu"

    input = ["Hi! How are you?"]

    tokenizer.pad_token = tokenizer.eos_token

    # When batch encode using `tokenizer(input)`:
    # If `input` is a single string, the resulting input_ids is a list[int]
    # If `input` is a list of string, the resulting input_ids is a list[list[int]]

    # pad to fixed length
    padded = tokenizer(
        input,
        padding="max_length",
        max_length=8,
        truncation=True,
        padding_side="left",
        device=device,
        return_attention_mask=True,
    )

    padded_input_ids = torch.tensor(padded.input_ids, device=device)
    valid_token_mask = torch.tensor(padded.attention_mask, device=device)

    seq_len_tensor = torch.tensor(
        [[padded_input_ids.shape[-1]]], device=valid_token_mask.device
    )
    num_valid_tokens = torch.sum(valid_token_mask, dim=-1, keepdim=True)
    num_padding_tokens = seq_len_tensor - num_valid_tokens

    attn_mask_padded, position_ids_padded = model.attn_mask_and_position_ids_for_prefix(
        num_padding_tokens,
        curr_seq_len=padded_input_ids.shape[-1],
        device=padded_input_ids.device,
    )

    padded_last_logits = model(
        padded_input_ids,
        attn_mask=attn_mask_padded,
    )[:, -1]

    padded_w_position_ids_last_logits = model(
        padded_input_ids,
        attn_mask=attn_mask_padded,
        position_ids=position_ids_padded,
    )[:, -1]

    unpadded = tokenizer(input, device=device, return_attention_mask=True)
    unpadded_input_ids = torch.tensor(unpadded.input_ids, device=device)
    attn_mask_unpadded, positoin_ids_unpadded = (
        model.attn_mask_and_position_ids_for_prefix(
            num_padding_tokens=torch.zeros((1, 1), device=device),
            curr_seq_len=unpadded_input_ids.shape[-1],
            device=device,
        )
    )

    unpadded_last_logits = model(
        torch.tensor(unpadded.input_ids, device=device),
        attn_mask=attn_mask_unpadded,
        position_ids=positoin_ids_unpadded,
    )[:, -1]

    # When the prompt isn't padded, the physical position is the same as
    # logical positions, so it's ok to not pass in position ids.
    #
    # When the prompt is padded, strictly speaking we need to pass logical
    # position ids to forward() because it differs from physical position ids.
    # O/w we would be using wrong positions for RoPE.
    #
    # However, the logits are very close. The theory is that the relative
    # positions between tokens are the same and RoPE can handle
    # shifts well.
    print_diff(
        "last_logits padded vs. padded w/ position ids",
        padded_last_logits,
        padded_w_position_ids_last_logits,
    )
    print_diff(
        "last_logits padded w/ position ids vs. unpadded",
        padded_w_position_ids_last_logits,
        unpadded_last_logits,
    )
    assert_allclose_tight(padded_w_position_ids_last_logits, unpadded_last_logits)


def test_milestone3_static_batching_no_cache_generate(
    local_model, reference_model, tokenizer, device
):
    input = [
        "Hi!",
        "Hi! How are you?",
        # "A prompt that's really, really long.",
    ]

    # Left-padding is the convention for inference. Autoregressive models
    # generate tokens based on the last position of the input sequence. Left
    # padding ensures that the actual text tokens are at the end of the tensor
    # just before generation starts, preventing the model from generating text
    # based on a dummy pad token.
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    batch_encoding = tokenizer(
        input,
        padding=True,
        padding_side="left",
        device=device,
        return_attention_mask=True,
    )
    input_ids = torch.tensor(batch_encoding.input_ids, device=device)
    attention_mask = torch.tensor(batch_encoding.attention_mask, device=device)

    prompt_length = input_ids.shape[-1]

    max_new_tokens = 16
    with torch.no_grad():
        local_res = local_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            valid_token_mask=attention_mask,
            use_kv_cache=False,
        )

        for i in range(len(input)):
            singleton_encoded = torch.tensor(
                tokenizer.encode(input[i : i + 1]), device=device
            )

            mask = torch.ones_like(singleton_encoded, device=singleton_encoded.device)
            local_res_singleton = local_model.generate(
                singleton_encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                valid_token_mask=mask,
                use_kv_cache=False,
            )

            assert torch.equal(local_res[i : i + 1], local_res_singleton)

        ref_res = reference_model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            use_cache=False,
        )[:, prompt_length:]

    assert local_res.shape == ref_res.shape
    assert torch.equal(local_res, ref_res)


# Compare <batched, kv cache> with <singleton, kv cache>
def test_milestone3_static_batching_w_cache_self_consistency1(
    local_model_cpu, tokenizer
):
    device = "cpu"
    model = local_model_cpu

    inputs = ["Hello!", "Hi! How are you?"]
    decode_steps = 3

    tokenizer.pad_token = tokenizer.eos_token

    # Step 0
    # Batch
    batch_encoding = tokenizer(
        inputs,
        padding=True,
        padding_side="left",
        device=device,
        return_attention_mask=True,
    )
    batch_input_ids = torch.tensor(batch_encoding.input_ids, device=device)
    attention_mask = torch.tensor(batch_encoding.attention_mask, device=device)

    batched_generated_tokens, batched_step_logits = model.generate(
        batch_input_ids,
        max_new_tokens=decode_steps + 1,
        do_sample=False,
        valid_token_mask=attention_mask,
        use_kv_cache=True,
        return_logits=True,
    )

    for i in range(len(inputs)):
        singleton_encoding = tokenizer([inputs[i]], return_attention_mask=True)
        input_ids = torch.tensor(singleton_encoding.input_ids, device=device)
        singleton_attention_mask = torch.tensor(
            singleton_encoding.attention_mask, device=device
        )

        singleton_generated_tokens, singleton_step_logits = model.generate(
            input_ids,
            max_new_tokens=decode_steps + 1,
            do_sample=False,
            valid_token_mask=singleton_attention_mask,
            use_kv_cache=True,
            return_logits=True,
        )

        assert_allclose_tight(
            batched_step_logits[i : i + 1],
            singleton_step_logits,
        )
        assert torch.equal(
            batched_generated_tokens[i : i + 1],
            singleton_generated_tokens,
        )


# Compare <batched, kv cache> with <singleton, no cache>
def test_milestone3_static_batching_w_cache_logits_self_consistency2(
    local_model_cpu, tokenizer
):
    device = "cpu"
    model = local_model_cpu

    inputs = ["Hello!", "Hi! How are you?"]
    decode_steps = 3

    tokenizer.pad_token = tokenizer.eos_token

    batch_encoding = tokenizer(
        inputs,
        padding=True,
        padding_side="left",
        device=device,
        return_attention_mask=True,
    )
    batch_input_ids = torch.tensor(batch_encoding.input_ids, device=device)
    attention_mask = torch.tensor(batch_encoding.attention_mask, device=device)

    # Use the real batched cached generation path under test, and compare its
    # per-step logits against a no-cache singleton generation oracle.
    batched_generated_tokens, batched_step_logits = model.generate(
        batch_input_ids,
        max_new_tokens=decode_steps + 1,
        do_sample=False,
        valid_token_mask=attention_mask,
        use_kv_cache=True,
        return_logits=True,
    )

    for i in range(len(inputs)):
        singleton_encoding = tokenizer([inputs[i]], return_attention_mask=True)
        singleton_input_ids = torch.tensor(singleton_encoding.input_ids, device=device)
        singleton_attention_mask = torch.tensor(
            singleton_encoding.attention_mask, device=device
        )

        oracle_generated_tokens, oracle_step_logits = model.generate(
            singleton_input_ids,
            max_new_tokens=decode_steps + 1,
            do_sample=False,
            valid_token_mask=singleton_attention_mask,
            use_kv_cache=False,
            return_logits=True,
        )

        assert_allclose_tight(batched_step_logits[i : i + 1], oracle_step_logits)
        assert torch.equal(
            batched_generated_tokens[i : i + 1],
            oracle_generated_tokens,
        )
