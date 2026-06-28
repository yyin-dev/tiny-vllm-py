# tiny-vllm-py

This project is a learning-oriented LLM inference engine built from scratch. It prioritizes clarity and hands-on understanding over
raw performance: the implementation is in Python, avoids custom CUDA kernels, and uses manually implemented model components on top
of PyTorch tensor ops. The initial scope is intentionally narrow, focusing on a single fixed model family and the minimum set of
modules needed to run it. The main objective is to understand and implement serving-side inference techniques such as prefill/
decode separation, KV cache management, and continuous batching.

## Milestones

- [x] Checkpoint loading & Forward decode
- [ ] Prefill/decode split
- [ ] KV cache
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


## Reference
https://github.com/jmaczan/tiny-vllm