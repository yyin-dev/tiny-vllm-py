import torch
import safetensors
import model
import logging
import json
from kv_cache import KVCache
from debug_collector import DebugCollector
from model import LlamaLM
from einops import rearrange
from transformers import PreTrainedTokenizer, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


WEIGHTS_PATH = "/Users/yy0125/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/model.safetensors"
CONFIG_PATH = "/Users/yy0125/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/config.json"


def load_model() -> LlamaLM:
    logger.info("Initializing Llama Model...")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
        vocab_size = config["vocab_size"]
        d_model = config["hidden_size"]
        num_layers = config["num_hidden_layers"]
        num_q_heads = config["num_attention_heads"]
        num_kv_heads = config["num_key_value_heads"]
        d_ff = config["intermediate_size"]
        rope_theta = config["rope_theta"]
        rope_scaling = config.get("rope_scaling")
        context_length = config["max_position_embeddings"]

    llama = model.LlamaLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
    )
    state_dict = llama.state_dict()

    logger.info("Loading from checkpoint...")
    loaded_checkpoint = {}
    with safetensors.safe_open(WEIGHTS_PATH, framework="pt") as f:

        def set(ckpt_name, state_dict_name):
            ckpt_weights = f.get_tensor(ckpt_name)
            state_dict_weights = state_dict[state_dict_name]
            assert ckpt_weights.shape == state_dict_weights.shape
            loaded_checkpoint[state_dict_name] = ckpt_weights

        # Technically it's sufficient to only set "token_embedding.weights" because
        # the two weights are tied. However, set both here s.t. we can use
        # load_state_dict(strict=True)
        set("model.embed_tokens.weight", "token_embeddings.weight")
        set("model.embed_tokens.weight", "lm_head.weight")

        for l in range(num_layers):
            set(f"model.layers.{l}.input_layernorm.weight", f"layers.{l}.ln1.weight")
            set(
                f"model.layers.{l}.self_attn.q_proj.weight",
                f"layers.{l}.attn.q_proj.weight",
            )
            set(
                f"model.layers.{l}.self_attn.k_proj.weight",
                f"layers.{l}.attn.k_proj.weight",
            )
            set(
                f"model.layers.{l}.self_attn.v_proj.weight",
                f"layers.{l}.attn.v_proj.weight",
            )
            set(
                f"model.layers.{l}.self_attn.o_proj.weight",
                f"layers.{l}.attn.output_proj.weight",
            )
            set(
                f"model.layers.{l}.post_attention_layernorm.weight",
                f"layers.{l}.ln2.weight",
            )
            set(f"model.layers.{l}.mlp.up_proj.weight", f"layers.{l}.ffn.w3.weight")
            set(f"model.layers.{l}.mlp.gate_proj.weight", f"layers.{l}.ffn.w1.weight")
            set(f"model.layers.{l}.mlp.down_proj.weight", f"layers.{l}.ffn.w2.weight")

        set("model.norm.weight", "ln_final.weight")

    logger.info("Set model weights based on checkpoint")
    llama.load_state_dict(loaded_checkpoint, strict=True)

    logger.info("Model loaded with weights from checkpoint")
    return llama


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().mean().item()


def print_diff(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape == b.shape:
        print(
            f"{name}: max_abs={max_abs_diff(a, b):.6f}, mean_abs={mean_abs_diff(a, b):.6f}, {a.shape}, {b.shape}"
        )
    else:
        print(f"{name}: shape mismatch! {a.shape} != {b.shape}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Manually replicate forward pass for debugging
def manual_debug(llama: LlamaLM):
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    reference_tokens = torch.tensor([[128000, 4438, 527, 499, 30, 358]], device=device)
    prefill_tokens = torch.tensor([[128000, 4438, 527, 499, 30]], device=device)
    decode_token = torch.tensor([[358]], device=device)

    with torch.no_grad():
        print("================ Reference =========================")
        layer0 = llama.layers[0]
        attn = layer0.attn

        # reference
        embeds = llama.token_embeddings(reference_tokens)
        x = layer0.ln1(embeds)

        Q = attn.q_proj(x)
        Q = rearrange(Q, "... seq (heads d) -> ... heads seq d", heads=attn.num_q_heads)

        Q_ref_last_token_before_rope = Q[:, :, -1:, :]

        K = attn.k_proj(x)
        V = attn.v_proj(x)

        K_ref_last_token_before_rope = K[:, -1:, :]

        K = rearrange(
            K, "... seq (heads d) -> ... heads seq d", heads=attn.num_kv_heads
        )
        V = rearrange(
            V, "... seq (heads d) -> ... heads seq d", heads=attn.num_kv_heads
        )

        print("Q.shape before rope", Q.shape)
        cos, sin = attn.positional_encoder(Q, None)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        Q_ref_last_token_after_rope = Q[:, :, -1:, :]
        print("Q.shape after rope", Q.shape)
        print(K.shape)
        print(V.shape)
        K_ref = K
        V_ref = V

        attn_output_ref = torch.nn.functional.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            is_causal=True,
            enable_gqa=True,
        )
        attn_output_ref = attn_output_ref[:, :, -1:, :]

        # using kv cache
        print("================ KV cache =========================")
        kv_cache = KVCache()

        # prefill
        prefill_embeds = llama.token_embeddings(prefill_tokens)
        _ = layer0(prefill_embeds, kv_cache=kv_cache)

        # decode
        embeds = llama.token_embeddings(decode_token)
        x = layer0.ln1(embeds)
        position_ids = torch.tensor([[len(prefill_tokens[0])]], device=device)

        Q = attn.q_proj(x)  # b 1 (h d)
        Q = rearrange(Q, "... seq (heads d) -> ... heads seq d", heads=attn.num_q_heads)

        current_token_q_before_rope = Q.clone().detach()

        current_token_k = attn.k_proj(x)  # b 1 (h d)
        current_token_v = attn.v_proj(x)  # b 1 (h d)

        current_token_k_before_rope = current_token_k.clone().detach()

        current_token_k = rearrange(
            current_token_k,
            "... seq (heads d) -> ... heads seq d",
            heads=attn.num_kv_heads,
        )
        current_token_v = rearrange(
            current_token_v,
            "... seq (heads d) -> ... heads seq d",
            heads=attn.num_kv_heads,
        )

        cos, sin = attn.positional_encoder(Q, position_ids=position_ids)
        Q, current_token_k = apply_rotary_pos_emb(Q, current_token_k, cos, sin)

        current_token_q_after_rope = Q.clone().detach()

        k_prefix, v_prefix = kv_cache.get_kv_prefix(attn.layer_idx)

        K = torch.cat([k_prefix, current_token_k], dim=-2)
        V = torch.cat([v_prefix, current_token_v], dim=-2)

        kv_cache.append_kvs(attn.layer_idx, current_token_k, current_token_v)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            is_causal=False,
            enable_gqa=True,
        )

        print(K.shape)
        print(V.shape)

        print_diff("K", K_ref, K)
        print_diff("V", V_ref, V)
        print_diff(
            "Q last token before RoPE",
            Q_ref_last_token_before_rope,
            current_token_q_before_rope,
        )
        print_diff(
            "last token k", K_ref_last_token_before_rope, current_token_k_before_rope
        )
        print_diff(
            "Q last token after RoPE",
            Q_ref_last_token_after_rope,
            current_token_q_after_rope,
        )
        print_diff("attn_output", attn_output_ref, attn_output)


def instrumentation_debug(llama: LlamaLM):
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    reference_tokens = torch.tensor([[128000, 4438, 527, 499, 30, 358]], device=device)
    prefill_tokens = torch.tensor([[128000, 4438, 527, 499, 30]], device=device)
    decode_token = torch.tensor([[358]], device=device)

    reference_collector = DebugCollector()
    _ = llama.forward(reference_tokens, debug_collector=reference_collector)

    kv_cache = KVCache()
    kv_collector = DebugCollector()
    kv_collector.set_prefill()
    _ = llama.forward(prefill_tokens, kv_cache=kv_cache, debug_collector=kv_collector)
    kv_collector.set_decode_step(0)
    _ = llama.forward(decode_token, kv_cache=kv_cache, debug_collector=kv_collector)

    layer_idx = 0
    print("================ Instrumentation =========================")
    print_diff(
        "Q last token before RoPE",
        reference_collector.get("unset", layer_idx, "q_pre_rope")[:, :, -1:, :],
        kv_collector.get("decode0", layer_idx, "q_pre_rope"),
    )
    print_diff(
        "Q last token after RoPE",
        reference_collector.get("unset", layer_idx, "q_post_rope")[:, :, -1:, :],
        kv_collector.get("decode0", layer_idx, "q_post_rope"),
    )
    print_diff(
        "K full prefix for attn",
        reference_collector.get("unset", layer_idx, "k_post_rope"),
        kv_collector.get("decode0", layer_idx, "k_for_attn"),
    )
    print_diff(
        "V full prefix for attn",
        reference_collector.get("unset", layer_idx, "v_for_attn"),
        kv_collector.get("decode0", layer_idx, "v_for_attn"),
    )
    print_diff(
        "attn_output",
        reference_collector.get("unset", layer_idx, "attn_output_heads")[:, :, -1:, :],
        kv_collector.get("decode0", layer_idx, "attn_output_heads"),
    )
    print(
        "reference is_causal:",
        reference_collector.get("unset", layer_idx, "is_causal"),
    )
    print("decode is_causal:", kv_collector.get("decode0", layer_idx, "is_causal"))
    print(
        "decode token positions:",
        kv_collector.get("decode0", layer_idx, "token_positions"),
    )


def main():
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct"
    )
    llama = load_model().to(device)
    llama.eval()

    input = "How are you?"
    encoded = torch.tensor(tokenizer.encode(input), device=device)
    encoded = rearrange(encoded, "(b seq) -> b seq", b=1)
    print(encoded)

    manual_debug(llama)
    instrumentation_debug(llama)

    print("================ Full Generation =========================")
    max_new_tokens = 16
    with torch.no_grad():
        kv_cache = KVCache()
        output = llama.generate(
            encoded,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            use_kv_cache=True,
            kv_cache=kv_cache,
        )
        print(output)
        print(output.shape)
        print(f"'{tokenizer.decode(output)}'")

        ref = llama.generate(
            encoded,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            use_kv_cache=False,
        )[0]
        print(ref)
        print(ref.shape)
        print(f"'{tokenizer.decode(ref)}'")


if __name__ == "__main__":
    main()
