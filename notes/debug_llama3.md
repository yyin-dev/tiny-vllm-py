# Debug Investigation Summary

## Goal

Get a reference-correct short-sequence forward/generation path for `Llama-3.2-1B-Instruct` after loading weights from the Hugging Face `.safetensors` checkpoint into the local `LlamaLM` implementation.

## Initial Symptoms

- Weight loading ran without shape errors.
- Generated text was nonsensical.
- Internal tensors before attention looked close to the Hugging Face reference, but later values diverged.

## Investigation Timeline


### Compare intermediate tensors with a reference model

We used `debug_reference.py` and `debug_generate.py` to compare:

- embeddings
- RMSNorm output
- `q_proj`, `k_proj`, `v_proj`
- RoPE-applied `Q/K`
- layer 0 output

Findings:

- Embeddings matched exactly.
- Pre-RoPE tensors were very close.
- Divergence became more noticeable after RoPE.
- Position 0 looked deceptively close because RoPE is effectively identity there.
- Later-token differences were much larger.

This shifted suspicion away from checkpoint loading and toward positional encoding.

### Root cause: two RoPE mismatches

The main bugs were:

1. `torchtune.modules.RotaryPositionalEmbeddings` was not Llama-compatible for this use case.

- It rotates adjacent pairs in the last dimension.
- Hugging Face Llama uses `rotate_half`, which rotates the first half of the head against the second half.

2. The checkpoint uses Llama-3 scaled RoPE, not plain RoPE.

From `config.json`:

- `rope_theta = 500000.0`
- `rope_scaling.rope_type = "llama3"`
- `factor = 32.0`
- `high_freq_factor = 4.0`
- `low_freq_factor = 1.0`
- `original_max_position_embeddings = 8192`

So even with the correct `rope_theta`, a plain RoPE implementation still diverges from the reference model.

### Fix RoPE path

We replaced the torchtune RoPE path with a minimal Llama-compatible implementation in `model.py`:

- `rotate_half`
- `apply_rotary_pos_emb`
- `LlamaRotaryEmbedding`

This new path matches the Hugging Face Llama convention and supports `rope_scaling` with `rope_type="llama3"`.

### Quantify remaining numerical drift

We added `debug_compare.py` to measure differences at key points.

Representative output after the RoPE fix:

- `embeds: max_abs=0.000000, mean_abs=0.000000`
- `layer0.ln1: max_abs=0.008036, mean_abs=0.000239`
- `layer0.q_proj: max_abs=0.038582, mean_abs=0.001742`
- `layer0.k_proj: max_abs=0.034471, mean_abs=0.002195`
- `q_post_rope_student_vs_reference: max_abs=0.058962, mean_abs=0.002436`
- `k_post_rope_student_vs_reference: max_abs=0.038739, mean_abs=0.002908`
- `attention_output_pre_o_proj: max_abs=0.003036, mean_abs=0.000099`
- `layer0_output: max_abs=0.068668, mean_abs=0.000171`

Conclusion:

- The remaining drift was small.
- It was consistent with backend / dtype differences (`fp32` local path vs mostly `bf16` reference, plus MPS behavior).
- It was no longer large enough to explain the earlier nonsensical generation.

### Investigate generation loop

Once the forward path looked correct, generation still differed from Hugging Face.

We found several decoding issues:

1. Sampling was always used instead of greedy decoding.

- Local `generate()` always called `torch.multinomial(...)`.
- Hugging Face `generate()` is greedy by default unless sampling is enabled.

2. EOS handling was incorrect.

- A token string from the tokenizer special-token map was being passed instead of integer token ids.

3. `top_k` masking had a bug.

- `masked_fill(...)` was called without assigning the result back.

4. Length semantics differed.

- Reference used total-length semantics initially.
- Local path used `max_new_tokens`.

### Fix generation path

We updated `generate()` to:

- support `do_sample=False`
- perform greedy `argmax` decoding when sampling is disabled
- accept `eos_token_id` as `int | list[int] | None`
- fix the `top_k` masking assignment bug

We also updated `debug_generate.py` to:

- pass real EOS ids
- use greedy decoding
- compare the same number of newly generated tokens as the reference

### Final verification

After the decode fixes, the results matched exactly on the test prompt:

- first greedy token id matched
- first greedy token text matched
- full decoded string matched
- newly generated text matched

## Final Root Causes

The original bad generations were caused by a combination of:

1. Wrong RoPE implementation path
   - torchtune pairing convention mismatch
   - missing Llama-3 `rope_scaling`

2. Generation-loop issues
   - sampling instead of greedy decoding
   - incorrect EOS handling
   - `top_k` masking bug
   - mismatched generation-length semantics during comparison

## What Did *Not* Turn Out To Be The Main Problem

- Checkpoint key translation, after the MLP mapping fix
- bf16 checkpoint loading into an fp32 local model
- GQA support, once the shape/plumbing bugs were corrected

## Current Status

For the current milestone:

- weight loading works
- short-sequence forward matches the reference closely
- greedy generation matches the reference on the tested prompt

This is sufficient to move on to later project goals.

NOTE: There is still a TODO in `main.py` about passing an explicit `attention_mask`.

Why it matters:

- it is not necessary for the current single unpadded prompt
- it will matter later for padded batching and cache-related inference paths

