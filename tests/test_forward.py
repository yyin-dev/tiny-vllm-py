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
from util import assert_allclose_loose


def test_milestone1_single_step_logits(local_model, reference_model, tokenizer, device):
    for input in [
        ["How are you?"],
        ["Hi!"],
        ["A really really really really long prompt"],
    ]:
        encoding = tokenizer(input)
        encoded = torch.tensor(encoding.input_ids, device=device)

        local_logits, _ = local_model(encoded)
        last_logits_local = local_logits[:, -1]
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
        local_logits, _ = local_model(encoded)
        last_logits_local = local_logits[:, -1]
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
