from torch import nn, Tensor
import torch
from torch.nn import Linear, Embedding
import torch.nn.functional as F
import math
from einops import einsum, rearrange
import logging
from jaxtyping import Float, Int, Bool
import os
import json
from kv_cache import KVCache
from debug_collector import DebugCollector

logger = logging.getLogger(__name__)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, bias=False)
        self.w2 = Linear(d_ff, d_model, bias=False)
        self.w3 = Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


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


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: float = 10_000.0,
        rope_scaling: dict | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.rope_scaling = rope_scaling
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)

    def _compute_default_inv_freq(self) -> torch.Tensor:
        return 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )

    def _compute_inv_freq(self) -> torch.Tensor:
        inv_freq = self._compute_default_inv_freq()
        if self.rope_scaling is None:
            return inv_freq

        rope_type = self.rope_scaling.get("rope_type", "default")
        if rope_type != "llama3":
            return inv_freq

        factor = self.rope_scaling["factor"]
        low_freq_factor = self.rope_scaling["low_freq_factor"]
        high_freq_factor = self.rope_scaling["high_freq_factor"]
        old_context_len = self.rope_scaling["original_max_position_embeddings"]

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / inv_freq
        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
        )
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (
            1 - smooth_factor
        ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        return inv_freq_llama

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (b, h, seq, d)
        position_id: (b, seq)

        `x` is only used for:
          [device];
          [shape], when [position_ids] isn't provided.
        """
        if position_ids is None:
            position_ids = torch.arange(x.shape[-2], device=x.device).unsqueeze(0)

        inv_freq = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype, device=x.device), sin.to(
            dtype=x.dtype, device=x.device
        )


class CausalMultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention

    This function implements section 3.2.2 of the Transformer paper. In particular,
    given an input tensor of shape `(batch_size, sequence_length, d_model)`, we project
    it to create queries, keys, and values, and then perform causal multi-headed attention with
    those queries, keys, and values.

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.

    Returns:
        Tensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,
        positional_encoder: LlamaRotaryEmbedding | None = None,
        layer_idx: int | None = None,
    ):
        super().__init__()
        if positional_encoder is None:
            print("Warning: No positional encoder provided!")

        assert d_model % num_q_heads == 0
        assert d_model % num_kv_heads == 0
        assert num_q_heads % num_kv_heads == 0

        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

        self.d_k = d_model // num_q_heads
        self.d_v = self.d_k

        self.q_proj = Linear(self.d_model, self.num_q_heads * self.d_k, bias=False)
        self.k_proj = Linear(self.d_model, self.num_kv_heads * self.d_k, bias=False)
        self.v_proj = Linear(self.d_model, self.num_kv_heads * self.d_v, bias=False)

        self.output_proj = Linear(self.num_q_heads * self.d_v, self.d_model, bias=False)

        self.positional_encoder: LlamaRotaryEmbedding | None = (
            positional_encoder  # RoPE
        )

        self.layer_idx = layer_idx if layer_idx else 0

    def forward(
        self,
        x: Float[Tensor, " ... seq d_k"],
        kvs: tuple[torch.Tensor, torch.Tensor] | None = None,
        debug_collector: DebugCollector | None = None,
        attn_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[Float[Tensor, " ... seq d_v"], torch.Tensor, torch.Tensor]:
        """
        Args:
            x: The input to perform multi-headed self-attention on.
            kvs: (..., num_heads, seq_len, head_dim)
            positional_ids: The positional indices along the sequence dimension of the input embeddings.
            attn_mask: If passed in, use it as is for [scaled_dot_product_attention]

        Returns:
            Self-attention outputs, new_k, new_v
        """
        *batch_dims, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        # Need to reshape Q, K, V to dimension expected by [scaled_dot_product_attention]
        Q = self.q_proj(x)
        Q = rearrange(Q, "... seq (heads d) -> ... heads seq d", heads=self.num_q_heads)
        if debug_collector:
            debug_collector.record(self.layer_idx, "q_pre_rope", Q)

        K = self.k_proj(x)
        V = self.v_proj(x)

        K = rearrange(
            K, "... seq (heads d) -> ... heads seq d", heads=self.num_kv_heads
        )
        V = rearrange(
            V, "... seq (heads d) -> ... heads seq d", heads=self.num_kv_heads
        )

        if kvs is None:
            # without cache or prefill
            if debug_collector:
                debug_collector.record(self.layer_idx, "k_pre_rope", K)
                debug_collector.record(self.layer_idx, "v_for_attn", V)

            if self.positional_encoder is not None:  # RoPE is enabled
                cos, sin = self.positional_encoder(Q, position_ids=position_ids)
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
                if debug_collector:
                    debug_collector.record(self.layer_idx, "q_post_rope", Q)
                    debug_collector.record(self.layer_idx, "k_post_rope", K)

            is_causal = True
            new_k, new_v = K, V
        else:
            # decode
            # Q: (b num_heads 1 head_dim)
            assert sequence_length == 1

            current_token_k, current_token_v = K, V

            if debug_collector:
                debug_collector.record(self.layer_idx, "k_pre_rope", current_token_k)
                debug_collector.record(self.layer_idx, "v_current", current_token_v)

            k_prefix, v_prefix = kvs

            # get token position from kv-cache length
            if position_ids is None:
                current_len = k_prefix.shape[-2]
                position_ids = torch.tensor([[current_len]], device=Q.device)

            if debug_collector:
                debug_collector.record(self.layer_idx, "token_positions", position_ids)

            if self.positional_encoder is not None:
                cos, sin = self.positional_encoder(Q, position_ids=position_ids)
                Q, current_token_k = apply_rotary_pos_emb(Q, current_token_k, cos, sin)
                if debug_collector:
                    debug_collector.record(self.layer_idx, "q_post_rope", Q)
                    debug_collector.record(
                        self.layer_idx, "k_post_rope", current_token_k
                    )

            if debug_collector:
                debug_collector.record(self.layer_idx, "k_prefix", k_prefix)
                debug_collector.record(self.layer_idx, "v_prefix", v_prefix)

            K = torch.cat([k_prefix, current_token_k], dim=-2)
            V = torch.cat([v_prefix, current_token_v], dim=-2)
            if debug_collector:
                debug_collector.record(self.layer_idx, "k_for_attn", K)
                debug_collector.record(self.layer_idx, "v_for_attn", V)

            is_causal = False
            new_k, new_v = current_token_k, current_token_v

        if attn_mask is not None:
            # Assumes that attn_mask already incorporates causal-ness.
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query=Q,
                key=K,
                value=V,
                attn_mask=attn_mask,
                enable_gqa=True,
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query=Q,
                key=K,
                value=V,
                is_causal=is_causal,
                enable_gqa=True,
            )

        if debug_collector:
            debug_collector.record(self.layer_idx, "attn_output_heads", attn_output)

        # Concatenate the attention output from all heads.
        # (..., sequence_length, num_heads * d_v).
        attn_output = rearrange(
            attn_output, "... heads seq d_v -> ... seq (heads d_v)"
        ).contiguous()

        # Apply the output projection
        output = self.output_proj(attn_output)
        return output, new_k, new_v


class TransformerBlock(nn.Module):
    """A single Transformer layer.

    This implements a single layer of the Transformer, as described in section 3.1
    of the paper.

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        positional_encoder: RotaryEmbedding
            The RoPE module to use.

    Returns:
        FloatTensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,
        d_ff: int,
        positional_encoder: LlamaRotaryEmbedding,
        layer_idx: int,
    ):
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            positional_encoder=positional_encoder,
            layer_idx=layer_idx,
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.ln1 = nn.RMSNorm(d_model)
        self.ln2 = nn.RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        kvs: tuple[torch.Tensor, torch.Tensor] | None = None,
        debug_collector: DebugCollector | None = None,
        attn_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: FloatTensor of shape `(batch_size, sequence_length, d_model)`.
                The input to process with the Transformer block.

        Returns:
            FloatTensor of shape `(batch_size, sequence_length, d_model)`.
        """
        # NOTE: this is a pre-norm Transformer, and differs from the original
        # description in the paper.

        # Apply the multi-head self-attention sublayer
        x_attn, new_k, new_v = self.attn(
            self.ln1(x),
            kvs=kvs,
            debug_collector=debug_collector,
            attn_mask=attn_mask,
            position_ids=position_ids,
        )
        attn_sublayer_output = x + x_attn

        # Apply the feed-forward sublayer
        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        return ffn_sublayer_output, new_k, new_v


class LlamaLM(nn.Module):
    """A Transformer language model.

    Args:
        vocab_size: int
            The number of unique items in the output vocabulary to be predicted.
        context_length: int,
            The maximum number of tokens to process at once.
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_layers: int
            The number of Transformer layers to use.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).

    Returns:
        FloatTensor of shape (batch size, sequence_length, vocab_size) with the
        predicted unnormalized next-word distribution for each token.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_q_heads: int,
        num_kv_heads: int,
        d_ff: int,
        rope_theta: float = 10_000.0,
        rope_scaling: dict | None = None,
    ):
        # Store the model configuration for serialization / deserialization
        self.config = {
            k: v
            for k, v in locals().items()
            if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model)
        d_head = d_model // num_q_heads
        self.positional_encoder = LlamaRotaryEmbedding(
            dim=d_head,
            max_seq_len=context_length,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    d_ff=d_ff,
                    positional_encoder=self.positional_encoder,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.ln_final = nn.RMSNorm(d_model)

        self.lm_head = Linear(d_model, vocab_size, bias=False)
        # Tie the weights, since the paper mentions that "we share the same weight
        # matrix between the two embedding layers and the pre-softmax linear transformation"
        self.lm_head.weight = self.token_embeddings.weight

        # report number of parameters
        logger.info(
            f"number of non-embedding parameters: {self.get_num_params() / 1e6:.2f}M"
        )

    def get_num_params(self) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(
        self,
        x: Int[Tensor, " ... sequence_length"],
        kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        debug_collector: DebugCollector | None = None,
        attn_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[
        Float[Tensor, " ... sequence_length vocab_size"],
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Args:
            x: Input IDs for language modeling.

        Returns: A FloatTensor of shape
            (batch size, sequence_length, vocab_size) with the predicted unnormalized next-word
            distribution for each token.
        """
        # (batch size, sequence_length, d_model)
        # NOTE: paper mentions "In the embedding layers, we multiply those
        # weights by sqrt(d_model)", but we aren't doing that here.
        embedded_tokens = self.token_embeddings(x)

        # (batch size, sequence_length, d_model)
        x = embedded_tokens
        new_kvs = []

        for idx, layer in enumerate(self.layers):
            layer_kvs = None
            if kvs is not None:
                layer_kvs = kvs[idx]

            # (batch size, sequence_length, d_model)
            x, new_k, new_v = layer.forward(
                x,
                kvs=layer_kvs,
                debug_collector=debug_collector,
                attn_mask=attn_mask,
                position_ids=position_ids,
            )

            new_kvs.append((new_k, new_v))

        # (batch size, sequence_length, d_model)
        x = self.ln_final(x)
        # (batch size, sequence_length, vocab_size)
        logits = self.lm_head(x)
        return (logits, new_kvs)

    def generate_one_token(
        self,
        logits: Float[Tensor, "b 1"],
        temperature: float = 1.0,
        top_k: int | None = None,
        do_sample: bool = False,
    ) -> Tensor:
        # apply temperature scaling
        temperature_scaled_next_token_logits = logits / temperature
        # If top-k is provided, take the tokens with the highest score
        if top_k:
            topk_values, _ = torch.topk(
                temperature_scaled_next_token_logits,
                min(top_k, temperature_scaled_next_token_logits.size(-1)),
            )
            # Get the score of the kth item that we kept---items with lower scores should be masked.
            threshold = topk_values[:, -1]
            topk_mask = temperature_scaled_next_token_logits < threshold
            temperature_scaled_next_token_logits = (
                temperature_scaled_next_token_logits.masked_fill(
                    topk_mask, float("-inf")
                )
            )

        if do_sample:
            next_token_probabilities = F.softmax(
                temperature_scaled_next_token_logits, dim=-1
            )
            next_token_id = torch.multinomial(next_token_probabilities, 1)
        else:
            next_token_id = torch.argmax(
                temperature_scaled_next_token_logits, dim=-1, keepdim=True
            )

        return next_token_id

    def attn_mask_and_position_ids(
        self,
        num_padding_tokens: Int[Tensor, "b 1"],
        logical_length: Int[Tensor, "b 1"],
        physical_length: int,
        query_physical_idx_start: int,
        query_physical_idx_end: int,
    ):
        """
        Returns (`attn_mask`, `position_ids`) for static batching, where:
        - `attn_mask` is (b, 1, q_seq, k_seq), where `q_seq = query_physical_idx_end - query_physical_idx_start`
        and `k_seq = logical_length`. This represents the key positions
        that can be attended to, for each query position.
        - `position_ids` is (b, q_seq). This represents the logical query positions.

        When KV cache is enabled, `q_seq` is either `physical_length` or 1.
        """

        # Attention mask is constructed from two things:
        # 1. A mask that masks out invalid token positions. This includes padding
        # tokens and inactive tokens after EOS.
        # 2. A causal mask.
        #
        # Position ids is basically logical index, clamped between 0 and `logical length - 1`.
        # This padding tokens get position id of 0, and inactive tokens after EOS
        # gets position id of `logical lenght - 1`.

        # Find out physical indices that can be attended to
        key_physical_indices = torch.arange(
            physical_length,
            device=num_padding_tokens.device,
        )  # (k_seq,)
        key_logical_indices = key_physical_indices - num_padding_tokens  # (b, k_seq)
        is_not_padding = key_logical_indices >= 0  # (b, k_seq)
        is_not_finished = key_logical_indices < logical_length  # (b, k_seq)
        key_attn_mask = is_not_padding & is_not_finished  # (b, k_seq)
        key_attn_mask = rearrange(key_attn_mask, "b k_seq -> b 1 k_seq")

        # Causal mask
        query_physical_indices = torch.arange(
            start=query_physical_idx_start,
            end=query_physical_idx_end,
            device=num_padding_tokens.device,
        )  # (q_seq,)

        key_physical_indices = torch.arange(
            physical_length, device=num_padding_tokens.device
        )

        causal_mask = rearrange(
            key_physical_indices, "k_seq -> 1 1 k_seq"
        ) <= rearrange(
            query_physical_indices, "q_seq -> 1 q_seq 1"
        )  # (1, q_seq, k_seq)

        # attn mask (b, q_seq, k_seq)
        attn_mask = causal_mask & key_attn_mask
        attn_mask = rearrange(attn_mask, "b q_seq k_seq -> b 1 q_seq k_seq")

        query_logical_indices = query_physical_indices - num_padding_tokens  # (b, seq)
        position_ids = torch.clamp(
            query_logical_indices,
            torch.zeros_like(logical_length),
            logical_length - 1,
        )

        return attn_mask, position_ids

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        valid_token_mask: Bool[Tensor, ""] | None = None,
        eos_token_id: int | None = None,
        do_sample: bool = False,
        use_kv_cache: bool = False,
        debug_collector: DebugCollector | None = None,
        return_logits: bool = False,
    ):
        """
        Args:
            x: LongTensor of shape `(b, sequence_length,)` or `(sequence_length, )`.
                Input IDs to condition on when generating.
            max_new_tokens: int
                Maximum number of tokens to generate.
            temperature: float
                Temperature to use during generation.
            top_k: int
                If provided, only sample from the `top_k` vocab items (by probability).
            valid_token_mask:
                `(b, sequence_length)`.
                True if a token is a real token. False if it's a padding token.
                If not provided, assume all tokens are valid.
            eos_token_id: int
                If provided, stop generation when we generate this ID.

        Returns:
            If [return_logits] is False, a LongTensor of shape
            `(batch_size, generated_length)` with generated token IDs.

            If [return_logits] is True, returns a tuple
            `(generated_tokens, step_logits)`, where [step_logits] has shape
            `(batch_size, generated_length, vocab_size)` and stores the
            next-token logits used at each generation step.
        """
        if valid_token_mask is not None:
            assert valid_token_mask.shape == x.shape

        # Add batch dim if not present
        if x.dim() == 1:
            x = rearrange(x, "seq_len -> 1 seq_len")

        if valid_token_mask is not None and valid_token_mask.dim() == 1:
            valid_token_mask = rearrange(valid_token_mask, "seq_len -> 1 seq_len")

        batch_size = x.shape[0]
        prompt_len = x.shape[-1]

        print("Prefilling/Step 0 starts")

        # loop state
        is_finished = torch.zeros((batch_size, 1), device=x.device).bool()
        if valid_token_mask is None:
            logical_length = torch.full((batch_size, 1), prompt_len, device=x.device)
        else:
            logical_length = torch.sum(valid_token_mask, dim=-1, keepdim=True)  # (b, 1)
        num_padding_tokens = prompt_len - logical_length
        print("logical length before prefill")
        print(logical_length)

        # prefill
        attn_mask, position_ids = self.attn_mask_and_position_ids(
            num_padding_tokens=num_padding_tokens,
            logical_length=logical_length,
            physical_length=prompt_len,
            query_physical_idx_start=0,
            query_physical_idx_end=prompt_len,
        )

        prefill_logits, new_kvs = self.forward(
            x,
            kvs=None,
            debug_collector=debug_collector,
            attn_mask=attn_mask,
            position_ids=position_ids,
        )

        if use_kv_cache:
            kv_cache = KVCache()
            for i in range(len(self.layers)):
                new_k, new_v = new_kvs[i]
                kv_cache.append_kvs(i, new_k, new_v)
        else:
            kv_cache = None

        next_token_logits = prefill_logits[:, -1]
        step_logits = [next_token_logits]
        next_token_id = self.generate_one_token(
            next_token_logits, temperature, top_k, do_sample
        )

        if eos_token_id:
            is_finished = next_token_id == eos_token_id
            logical_length[~is_finished] += 1
        else:
            logical_length += 1

        assert next_token_id.dim() == 2
        generated_tokens = next_token_id  # (b, 1)

        # decode
        for step in range(max_new_tokens - 1):
            print(f"Decode step {step}")
            if kv_cache:
                prev_token = generated_tokens[:, -1:]

                seq_len = x.shape[-1] + generated_tokens.shape[-1]

                # attn mask: (b, 1, seq)
                attn_mask, position_ids = self.attn_mask_and_position_ids(
                    num_padding_tokens,
                    logical_length=logical_length,
                    physical_length=seq_len,
                    query_physical_idx_start=seq_len - 1,
                    query_physical_idx_end=seq_len,
                )

                logits, new_kvs = self.forward(
                    prev_token,
                    kvs=[kv_cache.get_kv_prefix(i) for i in range(len(self.layers))],
                    attn_mask=attn_mask,
                    position_ids=position_ids,
                    debug_collector=debug_collector,
                )

                for i in range(len(self.layers)):
                    new_k, new_v = new_kvs[i]
                    kv_cache.append_kvs(i, new_k, new_v)
            else:
                full_prefix = torch.cat([x, generated_tokens], dim=-1)

                curr_seq_len = full_prefix.shape[-1]
                attn_mask, position_ids = self.attn_mask_and_position_ids(
                    num_padding_tokens=num_padding_tokens,
                    logical_length=logical_length,
                    physical_length=curr_seq_len,
                    query_physical_idx_start=0,
                    query_physical_idx_end=curr_seq_len,
                )

                # beyond the model's context length
                # need to truncate all: query dim, key dim, and position_ids
                if full_prefix.shape[-1] > self.context_length:
                    full_prefix = full_prefix[:, -self.context_length :]
                    attn_mask = attn_mask[
                        :, :, -self.context_length :, -self.context_length :
                    ]
                    position_ids = position_ids[:, -self.context_length :]

                logits, _new_kvs = self.forward(
                    full_prefix,
                    attn_mask=attn_mask,
                    position_ids=position_ids,
                    debug_collector=debug_collector,
                )  # (b, s, vocab_size)

            next_token_logits = logits[:, -1]
            step_logits.append(next_token_logits)
            next_token_id = self.generate_one_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                do_sample=do_sample,
            )

            generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)

            if eos_token_id:
                finished_at_this_step = next_token_id == eos_token_id
            else:
                finished_at_this_step = torch.zeros_like(next_token_id).bool()

            logical_length[~finished_at_this_step] += 1

            is_finished = is_finished | finished_at_this_step
            if is_finished.all():
                break

        if return_logits:
            return generated_tokens, torch.stack(step_logits, dim=1)

        return generated_tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path) as f:
            config = json.load(f)

        model = cls(**config)
        weights_path = os.path.join(pretrained_model_path, "model.pt")
        state_dict = torch.load(weights_path)

        # Remove _orig_mod. prefix that comes from serializing a compiled model
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model
