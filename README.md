# tiny-vllm-py

This project is a learning-oriented LLM inference engine built from scratch. It prioritizes clarity and hands-on understanding over
raw performance: the implementation is in Python, avoids custom CUDA kernels, and uses manually implemented model components on top
of PyTorch tensor ops. The initial scope is intentionally narrow, focusing on a single fixed model family and the minimum set of
modules needed to run it. The main objective is to understand and implement serving-side inference techniques such as prefill/
decode separation, KV cache management, and continuous batching.

## Milestones

- [x] Checkpoint loading & Forward decode
- [x] KV cache
- [x] Static batching
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

**At a high-level, what KV-cache really changes is just how you compute K/V in the attn block**. In prefill, need to store K/V to the cache. In decode, compute Q/K/V only for the new token, and retrieve K/V prefix for previous tokens from the cache, construct the full K/V for the full sequence, then run the attention computation.

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

The goal is to make batch inference produce result just like without batching. Static batching means sequences in a batch advance in lock-step. The consequences are:

* During prefill, we need to pad sequences to the same length
* During decode, the position ids need to account for padding tokens. Also, some sequences already ended while others are still decoding

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

* An `attn_mask` that tells it the positions it can attend to. For example, attention shouldn't attend to PAD tokens. 
* A `position_ids` that tells it the position of the current token in the **logical** sequence. 

This means we need to track at least the following metadata:

* `num_padding_tokens[i]`: number of padding tokens in each sequence
* `is_finished[i]`: whether EOS has been generated for a sequence

The core of implementing static batching is just:

* Maintain the `attn_mask` and `position_ids` correctly in `generate()`
* Update self-attention module to use the two arguments appropriately

In static batching, correctness should be defined at the level of the **logical sequence**, not the full physical batch tensor. *Active slot*s correspond to real logical sequences and should behave exactly the same as singleton generation. When a sequence produces EOS, it may still remain in the physical batch for shape alignment, but subsequent slots for that row are *inactive slot*s rather than part of the logical sequence. 

For these inactive slots, we do not make a semantic guarantee about the tokens or logits produced. The only important requirement is that active sequences continue to behave correctly. Under this weaker semantic, a simple implementation that keeps advancing physical positions after EOS can still be acceptable. A stricter implementation may additionally freeze position ids and prevent inactive slots from being attended to, but that is an implementation choice rather than a required part of the semantic contract.

## Continuous Batching

### Problem

A clear problem with static batching: during decode, sequences finish at different times. Finished sequences leave inactive slots while other sequences are still generating. We waste GPU compute and memory on inactive slots of already-finished sequences. 

### Scheduling Techniques

A mental model for scheduling techniques based on the unit of scheduling (batch vs. sequence vs. one step):

* Batch-level scheduling: Form a batch and run the whole batch until all requests finish. The batch membership is fixed. 
* Sequence-level scheduling: Once a sequence is admitted, it usually stays scheduled until it finishes.
* Iteration-level scheduling: At each decode iteration, decide which sequences participate in the next forward pass.

Static batching is batch-level scheduling because once a batch is formed, its membership remains fixed until the batch completes.

### Continuous batching

**Continuous batching** addresses the problem in static batching by dynamically changing the batch membership in the decode phase. When a sequence generates an \<eos>, it's immediately removed from the batch and replaced by another sequence. This eliminates inactive slots and improves GPU throughput. 

Example:

```
Step t:     A B C D
            B finishes
Step t+1:   A E C D
            D finishes
Step t+2:   A E C F
```

Continuous batching removes wasted batch slots. It does not require waiting for the longest sequence in the original batch.

Continuous batching is iteration-level scheduling because the scheduler can update the active decode batch after every decode iteration. In the common case, unfinished sequences continue decoding; finished sequences are removed; newly ready sequences are admitted. More advanced schedulers may also pause or preempt unfinished sequences for memory, fairness, or priority reasons.

### Request Lifecycle

For each request, it can be in one of three durable states:

* pending_prefill: request has arrived but hasn't been prefilled yet. 
* pending_decode: Prefill complete and KV cache exists. Eligible to be decoded.
* finished: Request is complete. KV cache can be freed. 

`active_decode` is **not** a separate durable lifecycle state. It is a transient scheduler view: the subset of requests in `pending_decode` selected for the current decode step.

A natural question is: why do we need a separate `pending_decode` state? Why cannot a request move from `pending_prefill` directly into `active_decode`? 

Instead of moving from `pending_prefill` to `active_decode` directly, a separate `pending_decode` state is useful because finishing prefill only makes a request eligible for decode. It does not mean the request should immediately enter the active decode set. The scheduler may delay admission for reasons like fairness/priority policy, preference to optimize inter-token latency for already-active requests, etc.

### Inference Server Flow

If we go one level higher above and consider what an inference server is doing, it's conceptually running in a loop:

```python
pending_prefill = { ... }
pending_decode = { ... }
finished = { ... }

while True:
    receive_new_requests(pending_prefill)
    
	if should_prefill(pending_prefill, pending_decode, ...):
        # prefill
        prefill_batch = get_prefill_batch(pending_prefill)
        next_tokens = model.prefill(prefill_batch)
        update_sequences(next_tokens)
        move_to_pending_decode(prefill_batch, ready_to_decode)
    else:
        # decode
        active_decode = select_decode_batch(pending_decode)
        next_tokens = model.decode(active_decode)
        update_sequences(next_tokens)
        move_finished_requests(active_decode, finished)
        return_unfinished_to_pending_decode(active_decode, pending_decode)
```

Here, `active_decode` only exists for a single decode execution. After that step, unfinished requests go back to `pending_decode` and finished requests move to `finished`.

At each scheduling step, the server decides whether to run a prefill step, a decode step, or a mix. Mixing prefill and decode requires more sophisticated scheduling and kernels, and ragged representations can help execute such variable-shaped work efficiently (see "Ragged Batching" section below).

Prefill has better arithmetic intensity because it processes sequence positions in parallel, resulting in higher GPU utilization. It directly affects time-to-first-token (TTFT).  Decode has lower arithmetic and is often more memory intensive due to KV-cache access. It directly affects inter-token latency. 

Whether to prioritize prefill or decode is a tradeoff: TTFT vs. inter-token latency, GPU utilization, etc. 

### Model API Boundary

One of the most important implementation decisions is how to split responsibilities between the shared model execution path and the batching-mode-specific wrappers.

Our design:

* `model.forward()` is batching-agnostic. It operates on normalized dense execution inputs, takes in raw prior KV state for the current execution, and returns the newly produced KV state for the current input. It does **not** own persistent KV-cache mutation and only computes logits and new KV slices.
* `model.generate()` is a wrapper around `model.forward()` for offline inference and static batching.
* `model.prefill()` and `model.decode()` are wrappers around `model.forward()` for continuous batching.

It is important that `model.forward()` does not own KV-cache mutation because static batching and continuous batching need different persistent cache interfaces:

* In static batching, the natural cache interface is batch-oriented: one rectangular KV cache for the whole batch.
* In continuous batching, the natural cache interface is sequence-oriented: each request needs its own resumable KV state.

If `model.forward()` mutated a particular cache object directly, it would become tied to one specific persistent KV representation. By keeping `forward()` limited to raw dense KV inputs/outputs, the wrappers are free to implement the cache orchestration differently for static batching and continuous batching.

This means the wrappers are responsible for orchestration:

* gathering persistent state into the dense execution view needed by `model.forward()`
* constructing batching-mode-specific `attn_mask` and `position_ids`
* appending the returned KV slices back into the persistent cache representation

Under this split, `model.forward()` stays reusable across static batching and continuous batching, while `generate`, `prefill`, and `decode` each handle the bookkeeping specific to their execution mode.

### KV Cache

For static batching, we can manage the KV cache for all sequences in a batch together. For continuous batching, we need to be able to manage the KV cache for each sequence independently. 

### Dense vs. Ragged batching

Continuous batching is a scheduling strategy for determining the decode batch. 

Ragged batch is an execution technique for processing variable-length sequences without padding everything to the length of the longest sequence. 

Let's consider how continuous batching can be implemented. One way is to represent the batch similarly to static batching, `(batch, seq_len, ...)`. For prefill, clearly we need to pad the prompts to the same length. For decode, each active sequence contributes one new query token so the input is already the same shape `(batch, 1, hidden_dim)`. However, padding is still needed for the attention operation. 

For example:

```
A current length = 128
B current length = 2048
C current length = 512
```

The real work is:

```
A attends over 128 KV positions
B attends over 2048 KV positions
C attends over 512 KV positions
```

A dense implementation may instead treat every sequence as if it had length `2048`, masking out padded positions for the shorter sequences. This wastes attention work and KV-cache memory bandwidth.

Ragged batching avoids representing the batch as a fully rectangular tensor of shape `(batch, max_seq_len, ...)`. Instead, it packs real tokens or KV blocks together and tracks metadata such as sequence lengths, offsets, block tables, attention masks, and position IDs.. Read detailed explanation in Hugging Face's article: [Continuous Batching](https://huggingface.co/blog/continuous_batching).  

### Summary

Continuous batching should be understood as a scheduling technique. It updates the active decode set after each decode iteration: finished requests are removed immediately, and decode-ready requests are admitted into freed capacity.

At the server-flow level, requests move through the durable lifecycle:

```
pending_prefill -> pending_decode -> finished
```

During a decode step, the scheduler temporarily selects an `active_decode` subset from `pending_decode` for that one execution.

The scheduler balances prefill and decode work. Prefill affects time-to-first-token and tends to have higher arithmetic intensity. Decode affects inter-token latency and is often constrained by KV-cache memory bandwidth.

At the implementation level, continuous batching requires flexible per-sequence state management. KV cache can no longer be tied permanently to fixed batch slots. Systems often use sequence-oriented or paged KV cache, plus ragged/paged attention, to execute variable-length decode batches efficiently.

A layered view:

- Top level: request lifecycle and inference server loop.
- Middle level: scheduling strategy, such as static batching vs. continuous batching.
- Bottom level: execution and memory mechanisms, such as KV cache management, ragged batching, and paged KV cache and paged attention (later milestones).



## Directory Structure
- `tests/`: testcases. Runnable throughout the project.
- `debug/`: debug scripts when working on milestones. Expected to be runnable at the commit it's created or updated, but later commits might break it.

## Reference
https://github.com/jmaczan/tiny-vllm
