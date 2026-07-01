# tiny-vllm-py

This project is a learning-oriented LLM inference engine built from scratch. It prioritizes clarity and hands-on understanding over
raw performance: the implementation is in Python, avoids custom CUDA kernels, and uses manually implemented model components on top
of PyTorch tensor ops. The initial scope is intentionally narrow, focusing on a single fixed model family and the minimum set of
modules needed to run it. The main objective is to understand and implement serving-side inference techniques such as prefill/
decode separation, KV cache management, and continuous batching.

## Milestones

- [x] Checkpoint loading & Forward decode
- [x] KV cache
- [ ] Static batching
- [ ] Continuous batching
- [ ] Paged KV cache

## Checkpoint loading & Forward generation

I will use "local model" to refer to our implementation of the model.

The major parts of this milestone are:
1. Ensure local model architecture matches the checkpoint architecture
2. Load checkpoint weights into the model
3. Ensure forward and generation matches reference

We started with Llama-3.2-1B-Instruct because it's small and has relatively 
simple architecture. 

To load checkpoints, translate names in `.safetensor` to match names in the 
model's `state_dict`, then call `load_state_dict` to load it back into the model.

Initially, model weights load fine and forward runs without runtime error, but
the model output was garbage. I spent some time debugging by printing out 
activations (e.g. embeddings, Q/K/V projections, Q/K after applying RoPE) and 
found that the activations are very close until applying RoPE.

In the end, we discovered that Llama-3.2-1B-Instruct uses a Llama-specific variant
that uses frequency-dependent scaling and thus there's an 
architecture mismatch. We fixed this by copying HuggingFace's implementation
in the `transformers` library. See more detailed workthrough in `notes/debug_llama3.md`.

After that, I considered switching to `Qwen-2-1.5B-Instruct` and `Qwen-2.5-1.5B-Instruct`
because they seem to follow original RoPE more closely. However, after digging,
it's also not trivial:

- Qwen uses bias in attention projection
- Qwen uses different RMS-norm epsilon
- Qwen's RoPE implementation is still different from original RoPE: it uses a rotate-half variant (see `notes/rope.md` for details).

In the end, I decided to continue using Llama-3.2-1B-Instruct because:
- It's already matching reference implementation
- Later milestones is mostly about using it correctly without needing to think about its internals. The biggest thing is to ensure that `position_ids` are constructed correctly.
- I can just treat RoPE as a black box going forwards.

## KV Cache

### Mental Model

We don't need to create new API for inference with KV-cache: just add a `use_kv_cache` argument to the current `generate` method. Internally, `generate` creates a `KVCache` object.

The KV-cache only matters at the attention-layer. For other modules, just need to thread through the kv-cache argument.

The KV cache stores RoPE-rotated K and plain Vs. The KV cache interface:
```python
KVCache:
    def append_kvs(self, layer_idx, ks, vs)
    	# ks: (batch_size, num_kv_heads, seq_len, k_head_dim)
    	# vs: (batch_size, num_kv_heads, seq_len, v_head_dim)
    	# This supports both prefill and decode
    	# During prefill, seq_len > 1. During decode, seq_len = 1

    def get_kv_prefix(self, layer_idx) -> (ks, vs):
      # Always returns the full prefix

    def current_length(self, layer_idx) -> int:
```

At a high-level, what KV-cache really changes is just how you compute K/V in the attn block. In prefill, need to store K/V to the cache. In decode, compute Q/K/V only for the new token, and retrieve K/V prefix for previous tokens from the cache, construct the full K/V for the full sequence, then run the attention computation.

For attention layer: 

* At prefill, input is full prompt, computes the full K/Vs, apply RoPE to Ks, and store those kv-cache. Returns attention block output and updated KV cache. Prefill will use the exact same attention path as a full-sequence forward. 
* At decode, input is new token and kv cache, computes q/k/v vector for the new token, apply RoPE to k, and store the k/v for the new token in KV-cache. For Q/K/V used in attention computation, Q comes from the current token input, but most K/V come from the cache. Returns attention block output and updated KV cache. Decode cannot reuse the same attention as a full-sequence forward because it needs to pull KV entries from the cache instead of computing it. In fact, it couldn't compute KVs for previous tokens because the input is just a single token. 

At decode, the self-attention layer needs to use the correct token-position index of the entire sequence s.t. positional embedding works correctly. 

At time=t, the model is allowed to attend to all previous tokens, including t. 

### Design Choices

**How would the self-attention layer know whether it's in prefill or decode phase?** One option is to pass in an extra argument (e.g. `is_prefill: bool`). However, a better option is to just use the kv-cache state. When the kv-cache is empty, it's in the prefill phase. When it's not empty, it's in the decode phase.

**In the decode phase, how would the self-attention layer know the position of the current token?** One option is to pass in extra argument like `token_position` from `generate`. However, we can similarly get this information from the cache. The token position is the kv-cache length *before* appending k/v of the current token into the cache.

**Debugging workflow.** I manually replicated the forward flow in my debug script `debug_kv_cache.py`. It works but has several downsides: (1) it's cumbersome to write (2) the forward path and the debug path are two separate paths and it's expensive and error-prone to keep the two in sync. There are cases where I got really confused when I updated the model forward path without updating the debug path.

A better approach is to run the actual forward path during debugging, but add instrumentation logic for debugging purposes. In this case, we can add in a debug collector that collects intermediate results. I asked Codex to implement a basic version.

### Mistakes I made in Initial Implementation

Forward vs. Debug path mismatch.

**Causalness in prefill vs. decode.** At one point, I found that Q, K, V matches but the attention results mismatch. This implies the problem is in the attention calculation itself. In prefill, we need causal self-attention (`is_causal=True`). In decode, the query is allowed to attend to all previous tokens, so `is_causal=False`!

## Static Batching

Static batching means sequences in a batch advance in lock-step. The two immediate consequences are:

* During prefill, we need to pad sequences to the same length
* During decode, some sequences already ended while others are still decoding

The goal is to make batch inference produce result just like without batching. Similar to implementing KV-cache, the focus is still the self-attention: in decoder-only transformer, self-attention is the only operation that mixes information across sequence positions and thus requires handling things like padding tokens carefully. 

For correctness, we need to distinguish "physical sequence" and "logical sequence". 

```
physical index:  0   1   2   3   4   5   6   7
logical index :          0   1   2   3
                 PAD PAD a   b   c   EOS X   X
```

In the example sequence above, we use PAD for both prefill padding and X for inactive slots after the sequence finishes. 0..1 is padding, 6..7 is inactive slots, 0..7 is the physical sequence and 2..5 is the logical sequence. The convention is to treat EOS as a real generated token and as part of the logical sequence.

What should we store in the KV cache? We have two options:

1. Only store kvs for logical sequence. 
   * Pro: No wasted memory. 
   * Con: KVCache needs to manage the kv entries for each sequence as separate tensors instead of a single tensor. This also introduces extra complexity around indexing. 
2. Store KVs for the full physical sequence
   * Pro: simplicity.
   * Con: wasted memory and computation. The cache doesn't track logical sequences and requires extra bookkeeping. 

For now, we will implement Option 2 for simplicity. 

Self-attention needs two pieces of information to handle batching:

* An attention mask that tells it the positions it can attend to. For example, attention shouldn't attend to PAD tokens. 
* A `position_ids` that tells it the position of the current position in the **logical** sequence. 

This means we need to track at least the following metadata:

* `pad_offset[i]`: physical position of the last padding token
* `logical_seq_len[i]`: length of the current logical sequence
* `is_finished[i]`: whether EOS has been generated for a sequence

For sequence `i`, the physical positions in range `[pad_offset[i], pad_offset[i] + logical_seq_len[i])` are valid. Note that the range is left-closed, right-open. 


## Reference
https://github.com/jmaczan/tiny-vllm