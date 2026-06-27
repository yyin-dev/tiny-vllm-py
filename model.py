from torch import nn, Tensor
import torch
import math
from einops import einsum, rearrange
import logging
from jaxtyping import Float, Int, Bool
import os
import json

logger = logging.getLogger(__name__)


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        """A linear layer initialized with truncated normal fan-in fan-out.

        Args:
            d_in: int
                The number of input features.
            d_out: int
                The number of output features.
        """

        super().__init__()
        std = math.sqrt(2 / (d_in + d_out))
        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_out, d_in), std=std, a=-3 * std, b=3 * std
            ),
            requires_grad=True,
        )

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

    def extra_repr(self):
        return f"d_out={self.weight.shape[0]}, d_in={self.weight.shape[1]}"


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        std = 1.0
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(vocab_size, d_model), std=std, a=-3 * std, b=3 * std
            ),
            requires_grad=True,
        )

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids, :]

    def extra_repr(self):
        return f"vocab_size={self.weight.shape[0]}, d={self.weight.shape[1]}"


def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    """
    This module implements root mean square layer normalization, as
    described in Eq. 4 of https://arxiv.org/abs/1910.07467

    Args:
        hidden_size: int
            Dimensionality of the input to normalize.
        eps: float, default is 1e-5
            A value added to the denominator for numerical stability.

    Returns:
        FloatTensor of same shape as input.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: FloatTensor of shape `(batch_size, *)`.
                The input to apply root mean square layer normalization on.

        Returns:
            FloatTensor of same shape as input
        """
        # NOTE: in practice, many implementations will
        # manually upcast the input to fp32 here to prevent overflow when you
        # square the input.
        # https://github.com/pytorch/pytorch/issues/66707
        in_dtype = x.dtype

        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms

        return (self.weight * x).to(in_dtype)

    def extra_repr(self):
        return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"


def softmax(x, dim=-1):
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(
        exponentiated_rescaled_input, dim=dim, keepdim=True
    )


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

        self.q_proj = Linear(self.d_model, self.num_q_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_kv_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_kv_heads * self.d_v)

        self.output_proj = Linear(self.num_q_heads * self.d_v, self.d_model)

        self.positional_encoder: LlamaRotaryEmbedding | None = (
            positional_encoder  # RoPE
        )

    def forward(
        self,
        x: Float[Tensor, " ... seq d_k"],
        token_positions: Int[Tensor, " ... seq"] | None = None,
    ) -> Float[Tensor, " ... seq d_v"]:
        """
        Args:
            x: The input to perform multi-headed self-attention on.
            positional_ids: The positional indices along the sequence dimension of the input embeddings.

        Returns:
            Self-attention outputs.
        """
        *batch_dims, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Take apart each head from the embedding dimension of Q, K, V to shape (..., num_heads, seq_len, d_k).
        Q = rearrange(Q, "... seq (heads d) -> ... heads seq d", heads=self.num_q_heads)
        K = rearrange(
            K, "... seq (heads d) -> ... heads seq d", heads=self.num_kv_heads
        )
        V = rearrange(
            V, "... seq (heads d) -> ... heads seq d", heads=self.num_kv_heads
        )

        if self.positional_encoder is not None:  # RoPE is enabled
            cos, sin = self.positional_encoder(Q, token_positions)
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # Shape: (..., num_heads, sequence_length, d_k)
        # Use [torch.nn.functional.scaled_dot_product_attention] to support GQA
        # When is_causal=True, casaul mask is constructed internally
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            is_causal=True,
            enable_gqa=True,
        )

        # Concatenate the attention output from all heads.
        # (..., sequence_length, num_heads * d_v).
        attn_output = rearrange(
            attn_output, "... heads seq d_v -> ... seq (heads d_v)"
        ).contiguous()

        # Apply the output projection
        output = self.output_proj(attn_output)
        return output


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
    ):
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            positional_encoder=positional_encoder,
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor):
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
        x_attn = self.attn(self.ln1(x))
        attn_sublayer_output = x + x_attn

        # Apply the feed-forward sublayer
        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        return ffn_sublayer_output


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
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)

        self.lm_head = Linear(d_model, vocab_size)
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
        self, x: Int[Tensor, " ... sequence_length"]
    ) -> Float[Tensor, " ... sequence_length vocab_size"]:
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

        for layer in self.layers:
            # (batch size, sequence_length, d_model)
            x = layer(x)
        # (batch size, sequence_length, d_model)
        x = self.ln_final(x)
        # (batch size, sequence_length, vocab_size)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | list[int] | None = None,
        do_sample: bool = False,
    ):
        """
        Args:
            x: LongTensor of shape `(1, sequence_length,)` or `(sequence_length, )`.
                Input IDs to condition on when generating.
            max_new_tokens: int
                Maximum number of tokens to generate.
            temperature: float
                Temperature to use during generation.
            top_k: int
                If provided, only sample from the `top_k` vocab items (by probability).
            eos_token_id: int
                If provided, stop generation when we generate this ID.

        Returns: A LongTensor of shape (max_new_tokens,) with the generated model output.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        original_sequence_length = x.size(-1)
        for _ in range(max_new_tokens):
            # Take the last `context_length` tokens if the input is
            # beyond the model's context length
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x
            # Get the logits from the model
            logits = self.forward(x)
            # Take the logits for the next token
            next_token_logits = logits[:, -1]
            # apply temperature scaling
            temperature_scaled_next_token_logits = next_token_logits / temperature
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
                next_token_probabilities = softmax(
                    temperature_scaled_next_token_logits, dim=-1
                )
                next_token_id = torch.multinomial(next_token_probabilities, 1)
            else:
                next_token_id = torch.argmax(
                    temperature_scaled_next_token_logits, dim=-1, keepdim=True
                )

            # End generation if we see the EOS token ID
            if eos_token_id is not None:
                eos_token_ids = (
                    eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
                )
                if next_token_id.item() in eos_token_ids:
                    break
            x = torch.cat((x, next_token_id), dim=-1)
        new_token_ids = x[:, original_sequence_length:]
        return new_token_ids

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
