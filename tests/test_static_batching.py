import torch

from util import assert_allclose_tight, print_diff


def test_milestone3_static_batching_singleton_no_cache_logits(
    local_model_cpu, tokenizer
):
    model = local_model_cpu
    device = "cpu"

    input = ["Hi! How are you?"]

    tokenizer.pad_token = tokenizer.eos_token

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

    attn_mask_padded, position_ids_padded = model.attn_mask_and_position_ids(
        num_padding_tokens=num_padding_tokens,
        logical_length=num_valid_tokens,
        physical_length=padded_input_ids.shape[-1],
        query_physical_idx_start=0,
        query_physical_idx_end=padded_input_ids.shape[-1],
    )

    padded_logits, _ = model(
        padded_input_ids,
        attn_mask=attn_mask_padded,
    )
    padded_last_logits = padded_logits[:, -1]

    padded_w_position_ids_logits, _ = model(
        padded_input_ids,
        attn_mask=attn_mask_padded,
        position_ids=position_ids_padded,
    )
    padded_w_position_ids_last_logits = padded_w_position_ids_logits[:, -1]

    unpadded = tokenizer(input, device=device, return_attention_mask=True)
    unpadded_input_ids = torch.tensor(unpadded.input_ids, device=device)
    unpadded_logical_length = torch.full(
        (1, 1), unpadded_input_ids.shape[-1], device=device
    )
    attn_mask_unpadded, positoin_ids_unpadded = model.attn_mask_and_position_ids(
        num_padding_tokens=torch.zeros((1, 1), device=device),
        logical_length=unpadded_logical_length,
        physical_length=unpadded_input_ids.shape[-1],
        query_physical_idx_start=0,
        query_physical_idx_end=unpadded_input_ids.shape[-1],
    )

    unpadded_logits, _ = model(
        torch.tensor(unpadded.input_ids, device=device),
        attn_mask=attn_mask_unpadded,
        position_ids=positoin_ids_unpadded,
    )
    unpadded_last_logits = unpadded_logits[:, -1]

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
    ]

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

    max_new_tokens = 8
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


def test_milestone3_static_batching_w_cache_self_consistency1(
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
