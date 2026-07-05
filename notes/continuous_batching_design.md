# Continuous Batching Design

## Goal

Implement continuous batching with a higher-level `Engine` above `model.generate()`. Keep `generate()` as the convenience API for offline single-request / fixed-batch inference.

## Abstraction Split

- `generate()`
  - Offline convenience path.
  - Not the primary abstraction for continuous batching.

- `Engine`
  - Owns request lifecycle, scheduling, and KV ownership.
  - Exposes `step()`, which performs exactly one model execution.

- `Model`
  - Exposes separate `prefill(...)` and `decode(...)` APIs.
  - Does not know about engine request wrapper types.

## Model Execution Boundary

- `LlamaLM.forward` and internal modules in `model.py` operate on a normalized dense execution view.
- They are batch-agnostic with respect to static batching vs. continuous batching.
- `generate`, `prefill`, and `decode` are wrappers around `forward` and handle batching-mode-specific cache orchestration.

## Request Representations

- User-facing request
  - Immutable submission payload.
  - Contains original prompt and caller-facing config.

- Engine-internal prefill work item
  - Request waiting for first execution.
  - No KV state.
  - Cannot be decoded.

- Engine-internal decode-ready state
  - Resumable generation state after prefill.
  - Contains generated output so far, next decode input token, stop-condition metadata, and engine-owned KV state/handle.

## Durable Engine State

- `pending_prefill`
  - Pool of prefill work items.
  - Not modeled as a queue; policy may vary.

- `pending_decode`
  - Pool of decode-ready unfinished requests.

- `finished`
  - Terminal runtime objects for completed requests.

## Transient / Scheduler State

- `active_decode`
  - Transient subset chosen for one decode execution.
  - Not a durable lifecycle bucket.

- Scheduler sticky metadata
  - Remembers which decode-ready requests should stay in service across steps.

## Lifecycle

- Submitted request -> prefill work item
- Prefill step consumes prefill work item
- After prefill:
  - if EOS emitted: move to `finished`
  - else: create new decode-ready state and place in `pending_decode`
- Decode step selects transient subset from `pending_decode`
- After decode:
  - if EOS emitted: move to `finished`
  - else: return to `pending_decode`

## `Engine.step()` Contract

- Performs exactly one model execution.
- For now, execution is either prefill-only or decode-only.
- Later it may support mixed prefill+decode while still remaining one execution.
- Returns outputs only for requests that advanced in that execution.
- Current invariant: each participating request emits exactly one token per step.

## Scheduling Policy: First Version

- Prefer decode by default.
- Run prefill only when:
  - `num_pending_prefill` exceeds a threshold.
- If no decode-ready requests exist and prefill work exists, run prefill.
- Time-based aging can be added later if needed.

## Batch Selection Policy: First Version

- Prefill selection: FIFO.
- Decode selection: sticky membership.
  - Once a request is admitted into decode service, keep decoding it until it finishes.
  - Newly decode-ready requests may wait in `pending_decode`.

## Model-Facing API

- `generate`, `prefill`, and `decode` prepare dense execution inputs and call `forward()`.
- `forward()` takes dense prior K/V state for the current execution and returns newly produced K/V slices.
- `generate`, `prefill`, and `decode` are responsible for persistent cache writes, e.g. `append_kvs(...)`.
- This keeps the shared model path agnostic to whether persistent KV state is stored batch-wise or per sequence.

## KV Design

- For this milestone, use dense layout and defer ragged batching.
- Persistent KV state is managed per sequence/request outside the shared `forward()` path.
- The shared model execution path should not depend on per-request cache objects.
- Raw dense K/V tensors are preferred at the `forward()` boundary because they preserve a single batch-agnostic execution path.
- Tradeoff: wrappers (`generate`, `prefill`, `decode`) do more explicit cache orchestration and writeback.

## Ownership Boundary

- Engine owns request lifecycle and KV state association.
- Shared model execution only consumes normalized dense inputs needed for execution.
- Model should not depend on engine-specific types like prefill work item or decode-ready state.

## Decode-Ready State Essentials

Before decode:
- next input token
- resumable KV state
- metadata needed to evaluate stop condition
- request identity / routing info

After decode:
- append emitted token to accumulated output
- either mark finished or set emitted token as next input token
- preserve updated KV state

## Implementation Direction

Start by implementing:
1. Separate model APIs: `prefill` and `decode`
2. Engine state containers: `pending_prefill`, `pending_decode`, `finished`
3. `Engine.step()` with prefill-vs-decode scheduling
4. Simple FIFO prefill and sticky decode membership
5. Request transitions: prefill work item -> decode-ready state -> finished
