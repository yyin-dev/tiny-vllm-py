import torch

from forward_test_helpers import append_kvs
from util import assert_allclose_loose, assert_allclose_tight


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

        assert ref_first_step_token == prefill_max_prob_token
        assert_allclose_tight(ref_first_step_last_logits, prefill_last_logits)

        steps = 8
        curr_prompt_and_decode_res = torch.cat(
            (encoded, torch.tensor([[ref_first_step_token]], device=device)), dim=-1
        )
        next_decode_step_input = torch.tensor([[prefill_max_prob_token]], device=device)
        for _ in range(steps):
            ref_logits, _ = model(curr_prompt_and_decode_res)
            ref_last_logits = ref_logits[:, -1]
            ref_max_prob_token = torch.argmax(ref_last_logits, dim=-1).item()

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
            prefill_logits, kvs = local_model(encoded, kvs=None)
            prefill_last_logits = prefill_logits[:, -1]

            ref_prefill_last_logits = reference_model(encoded).logits[:, -1]

            prefill_max_prob_token = torch.argmax(prefill_last_logits, dim=-1).item()
            ref_prefill_max_prob_token = torch.argmax(
                ref_prefill_last_logits, dim=-1
            ).item()

            assert prefill_max_prob_token == ref_prefill_max_prob_token
            assert_allclose_loose(prefill_last_logits, ref_prefill_last_logits)

            curr_prompt_and_decode_res = torch.cat(
                (encoded, torch.tensor([[ref_prefill_max_prob_token]], device=device)),
                dim=-1,
            )
            next_decode_step_input = torch.tensor(
                [[prefill_max_prob_token]], device=device
            )

            steps = 4
            for _ in range(steps):
                ref_last_logits = reference_model(curr_prompt_and_decode_res).logits[
                    :, -1
                ]
                ref_max_prob_token = torch.argmax(ref_last_logits, dim=-1).item()

                decode_step_logits, new_kvs = local_model(
                    next_decode_step_input, kvs=kvs
                )
                decode_step_last_logits = decode_step_logits[:, -1]
                kvs = append_kvs(kvs, new_kvs)
                decode_step_max_prob_token = torch.argmax(
                    decode_step_last_logits, dim=-1
                ).item()

                assert ref_max_prob_token == decode_step_max_prob_token
                assert_allclose_loose(ref_last_logits, decode_step_last_logits)

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
