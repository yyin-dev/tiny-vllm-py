"""
Debug script when working on static batching milestone.
"""

import torch
import logging
from einops import rearrange
from transformers import PreTrainedTokenizer, AutoTokenizer

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from debug_collector import DebugCollector
from load_checkpoint import load_model
from kv_cache import KVCache
from model import LlamaLM

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model: LlamaLM = load_model()
    model = model.to(device).eval()

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
    )

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
    debug_collector = DebugCollector()
    res = model.generate(
        input_ids,
        max_new_tokens=1,
        valid_token_mask=attention_mask,
        eos_token_id=tokenizer.eos_token_id,
        debug_collector=debug_collector,
    )

    print(debug_collector.get("unset", 0, "attn_mask"))
    print(debug_collector.get("unset", 0, "position_ids"))


if __name__ == "__main__":
    main()
