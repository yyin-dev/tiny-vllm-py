# tiny-vllm-py

This project is a learning-oriented LLM inference engine built from scratch. It prioritizes clarity and hands-on understanding over
raw performance: the implementation is in Python, avoids custom CUDA kernels, and uses manually implemented model components on top
of PyTorch tensor ops. The initial scope is intentionally narrow, focusing on a single fixed model family and the minimum set of
modules needed to run it. The main objective is to understand and implement serving-side inference techniques such as prefill/
decode separation, KV cache management, and continuous batching.

## Milestones

- [ ] Reference-correct full forward
- [ ] Generation loop
- [ ] Prefill/decode split
- [ ] KV cache
- [ ] Static batching
- [ ] Continuous batching
- [ ] Paged KV cache

## Reference
https://github.com/jmaczan/tiny-vllm