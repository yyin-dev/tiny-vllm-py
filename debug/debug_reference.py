"""
Debug script when working on checkpoint loading milestone.

Steps through reference model to inspect activations for debugging purpose.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    PreTrainedTokenizer,
)
import einops

WEIGHTS_PATH = "/Users/yy0125/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/model.safetensors"


# Copied from `transformers` implementation
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from `transformers` implementation
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def main():
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)

    # Batch encoding
    # https://huggingface.co/docs/transformers/main/main_classes/tokenizer#transformers.PythonBackend.__call__
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    input = ["How are you?", "How are you?", "Which model are you?"]

    # Left-padding is the convention for inference. Autoregressive models
    # generate tokens based on the last position of the input sequence. Left
    # padding ensures that the actual text tokens are at the end of the tensor
    # just before generation starts, preventing the model from generating text
    # based on a dummy pad token.
    batch_encoding = tokenizer(
        input, padding=True, return_attention_mask=True, padding_side="left"
    )
    print(batch_encoding.input_ids)
    print(batch_encoding.attention_mask)

    encoded = torch.tensor(batch_encoding.input_ids, device=device)
    attention_mask = torch.tensor(batch_encoding.attention_mask, device=device)
    prompt_len = encoded.shape[-1]
    max_new_tokens = 11

    model.eval()

    logits = model(encoded).logits
    first_next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
    print("reference first greedy token id:", first_next_token)
    print(
        "reference first greedy token text:",
        tokenizer.decode(first_next_token[:]),
    )

    res = model.generate(
        inputs=encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        attention_mask=attention_mask,
    )

    decoded_full = tokenizer.decode(res[0])
    decoded_new = tokenizer.decode(res[0, prompt_len:])
    print("reference decoded full:", decoded_full)
    print("reference decoded new:", decoded_new)

    return

    # ====================== For debugging ==========================
    embeddings: torch.Tensor = model.model.embed_tokens(encoded)
    print("embeddings:", embeddings)

    layer0 = model.model.layers[0]

    normed = layer0.input_layernorm(embeddings)
    print("after first layer norm: ", normed)

    q = layer0.self_attn.q_proj(normed)
    print("q.shape", q.shape)
    print("q:", q)

    k = layer0.self_attn.k_proj(normed)
    print("k.shape", k.shape)
    print("k:", k)

    v = layer0.self_attn.v_proj(normed)
    print("v:", v)

    position_ids = torch.arange(encoded.shape[-1]).unsqueeze(0).to(device)
    cos, sin = model.model.rotary_emb(embeddings, position_ids)

    # number of query heads: 32, number of key value heads: 8.
    q = einops.rearrange(q, "b s (h d) -> b h s d", h=32)
    k = einops.rearrange(k, "b s (h d) -> b h s d", h=8)

    # Expects (batch size, num heads, seq length, head dim)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    q = einops.rearrange(q, "b h s d -> b s h d")
    k = einops.rearrange(k, "b h s d -> b s h d")

    print("Q after applying RoPE: ", q)
    print("Q shape after applying RoPE: ", q.shape)
    print(q[0, 0, 0, :])
    print("K after applying RoPE: ", k)
    print("K shape after applying RoPE: ", k.shape)
    print(k[0, 0, 0, :])

    hidden = layer0(embeddings, position_embeddings=(cos, sin))
    print("1st attn block output:", hidden)


if __name__ == "__main__":
    main()
