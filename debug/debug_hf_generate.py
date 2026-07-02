"""
For debugging HF model .generate() inconsistency.

When using HF's model as correctness reference, we observed generate()'s output
is inconsistent when in batch or when kv-cache is enabled. This script experiments
with different device and/or data types.

Run the following to see difference:

uv run debug/debug_static_batching.py
uv run debug/debug_static_batching.py --device cpu
uv run debug/debug_static_batching.py --upcast-to-fp32

This shows that generate()'s inconsistency on batch result is from MPS BF16
imprecision.
"""

import argparse
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().mean().item()


def logical_position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    return position_ids.long()


def run_generate(model, input_ids, attention_mask, max_new_tokens, pad_token_id):
    return model.generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False,
        return_dict_in_generate=True,
        output_scores=True,
    )


def summarize_generate_check(tokenizer, batched_res, singleton_res, row_idx):
    batched_tokens = [
        scores[row_idx : row_idx + 1].argmax(dim=-1).item()
        for scores in batched_res.scores
    ]
    singleton_tokens = [scores.argmax(dim=-1).item() for scores in singleton_res.scores]

    first_diff = None
    for step_idx, (a, b) in enumerate(zip(batched_tokens, singleton_tokens)):
        if a != b:
            first_diff = step_idx
            break

    print(f"row {row_idx} generate check")
    print("  batched tokens:  ", batched_tokens)
    print("  singleton tokens:", singleton_tokens)
    if first_diff is None:
        print("  first divergent step: none")
    else:
        print("  first divergent step:", first_diff)
        print(
            "  divergent tokens:",
            tokenizer.convert_ids_to_tokens(
                [batched_tokens[first_diff], singleton_tokens[first_diff]]
            ),
        )
        batched_scores = batched_res.scores[first_diff][row_idx : row_idx + 1]
        singleton_scores = singleton_res.scores[first_diff]
        print(
            "  divergent score diff:",
            f"max_abs={max_abs_diff(batched_scores, singleton_scores):.6f}",
            f"mean_abs={mean_abs_diff(batched_scores, singleton_scores):.6f}",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--upcast-to-fp32", action="store_true")
    args = parser.parse_args()

    requested_device = args.device

    if requested_device is not None:
        device = requested_device
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        "Hi!",
        "Hi! How are you?",
    ]

    batch_encoding = tokenizer(
        prompts,
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    )
    batch_input_ids = torch.tensor(batch_encoding.input_ids, device=device)
    batch_attention_mask = torch.tensor(batch_encoding.attention_mask, device=device)

    print("device:", device)
    print("batch_input_ids:", batch_input_ids.tolist())
    print("batch_attention_mask:", batch_attention_mask.tolist())

    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    if args.upcast_to_fp32:
        model = model.float()

    model.eval()
    print("model dtype:", model.dtype)
    print("upcast_to_fp32:", args.upcast_to_fp32)

    max_new_tokens = 16

    batched_res = run_generate(
        model,
        batch_input_ids,
        batch_attention_mask,
        max_new_tokens,
        tokenizer.pad_token_id,
    )

    for row_idx, prompt in enumerate(prompts):
        singleton_input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        singleton_attention_mask = torch.ones_like(singleton_input_ids)

        singleton_res = run_generate(
            model,
            singleton_input_ids,
            singleton_attention_mask,
            max_new_tokens,
            tokenizer.pad_token_id,
        )

        print()
        print("=" * 60)
        print(f"prompt {row_idx}: {prompt!r}")
        summarize_generate_check(tokenizer, batched_res, singleton_res, row_idx)


if __name__ == "__main__":
    main()
