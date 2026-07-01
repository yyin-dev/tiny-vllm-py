"""
Debug script when working on checkpoint loading milestone.
"""

import torch
import logging
from einops import rearrange
from transformers import PreTrainedTokenizer, AutoTokenizer

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from load_checkpoint import load_model
import model
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

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct"
    )

    input = "How are you?"
    encoded = torch.tensor(tokenizer.encode(input), device=device)
    encoded = rearrange(encoded, "(b seq) -> b seq", b=1)
    print(encoded)

    llama = load_model().to(device)
    llama.eval()

    embeds = llama.token_embeddings(encoded)
    print(embeds)

    layer0 = llama.layers[0]

    normed = layer0.ln1(embeds)
    print(normed)

    q = layer0.attn.q_proj(normed)
    print(q)

    k = layer0.attn.k_proj(normed)
    print(k)

    v = layer0.attn.v_proj(normed)
    print(v)

    q = rearrange(q, "... seq (heads d) -> ... heads seq d", heads=32)
    print(q.shape)
    k = rearrange(k, "... seq (heads d) -> ... heads seq d", heads=8)

    cos, sin = layer0.attn.positional_encoder(q)
    rope_q, rope_k = model.apply_rotary_pos_emb(q, k, cos, sin)

    rope_q = rearrange(rope_q, "... heads seq d -> ... seq heads d")
    print("rope_q.shape:", rope_q.shape)
    print("Q after applying RoPE:", rope_q)
    print(rope_q[0, 0, 0, :])

    rope_k = rearrange(rope_k, "... heads seq d -> ... seq heads d")
    print("rope_k.shape:", rope_k.shape)
    print("K after applying RoPE:", rope_k)
    print(rope_k[0, 0, 0, :])

    hidden = embeds
    hidden = layer0(embeds)
    print("1st attn block: ", hidden)


if __name__ == "__main__":
    main()
