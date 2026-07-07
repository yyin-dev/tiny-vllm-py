# Benchmarks

This directory contains three complementary benchmark types for this project:

* **Inference modes benchmark**: compares model-side execution paths such as singleton generation, static batching, KV cache, prefill, and decode.
* **Scheduler simulation**: isolates scheduling policy in abstraction, without running the real model.
* **Serving policy benchmark**: measures end-to-end wall-clock behavior for static vs. continuous batching on the local model.

These benchmarks are designed for **relative comparisons** on the current machine. On non-CUDA setups, including macOS/MPS, the results should be read as implementation-level trends rather than production serving numbers.

Benchmark Hardware

* Machine: `macOS 15.5 arm64`
* Device: `MPS`

## Inference Modes Benchmark

Script: `benchmark_inference_modes.py`

### Scenarios

For the current saved run, the batched scenarios use `batch_size=4`.

* `singleton_no_kv`: one request at a time, recomputing the full prefix each decode step
* `singleton_kv`: one request at a time, reusing KV cache during decode
* `static_batch_no_kv`: one padded batch with `batch_size=4`, without KV cache
* `static_batch_kv`: one padded batch with `batch_size=4`, with KV cache
* `continuous_batch_kv`: the current `Engine` path with all 4 requests submitted up front

This shows how the local implementation behaves under different inference modes and API paths.

Latest saved run:

* Workload: 4 prompts, `batch_size=4`, `max_new_tokens=4`, `warmup=1`, `repeats=2`
* Raw data: inference_modes_latest.json

### Full-Generation Comparisons

These rows all generate the same total amount of new-token work, so direct speedup comparisons are meaningful here.

Speedup baseline: `singleton_no_kv`

| Scenario | Wall time (s) | New tok/s | Speedup |
| --- | ---: | ---: | ---: |
| `singleton_no_kv` | 2.762 | 5.79 | `1.00x` |
| `singleton_kv` | 1.876 | 8.53 | `1.47x` |
| `static_batch_no_kv` | 1.130 | 14.16 | `2.45x` |
| `static_batch_kv` | 0.581 | 27.54 | `4.75x` |
| `continuous_batch_kv` | 0.671 | 23.84 | `4.11x` |

Main takeaways:

* KV cache improved singleton generation from `2.762s` to `1.876s`, about `1.47x` faster in this workload.
* `static_batch_kv` was about `4.75x` faster than `singleton_no_kv`.
* `continuous_batch_kv` remained in the same ballpark as `static_batch_kv`, but somewhat slower in this run.

### Stage Microbenchmarks

These rows isolate one stage of the pipeline and do not perform the same total work as full generation, so they should not be read as direct speedup peers of the table above.

Setup: static batch with batch_size=4, with KV cache.

The `Timed input tok/s` column counts only the tokens that actually enter the model during the measured stage:

* for `prefill_only`, that is all prompt tokens across the batch
* for `decode_only`, that is one current-token input per request, since decode consumes one token per active sequence at a time

More concretely:

* `prefill_only` means: take 4 prompts, run one batched `model.prefill(...)` call, build KV caches for all requests, and emit the first next-token prediction for each request. It is closest to the prompt-processing stage of batched cached generation.
* `decode_only` means: first run a prefill to populate KV caches, then measure only one batched `model.decode(...)` step on top of those caches. The setup prefill is required to create a valid cached state, but it is not included in the reported wall time.

| Scenario | Wall time (s) | New tok/s | Timed input tok/s | What it isolates |
| --- | ---: | ---: | ---: | --- |
| `prefill_only` | 0.346 | 11.58 | 182.32 | One batched prefill call that emits the first token for each request |
| `decode_only` | 0.108 | 37.03 | 37.03 | One batched decode iteration, timed after cache setup |

In this run, prefill processed timed input tokens at about `182 tok/s`, while decode processed about `37 tok/s`, which makes the “many prompt tokens at once” vs. “one token per request per step” difference much more visible.

## Scheduler Simulation

Script: `simulate_scheduler.py`

This is intentionally different from the model benchmark above. It does **not** measure model execution time. Instead, it isolates the scheduling effect of:

* **static batching**: batch membership is fixed until the whole batch completes
* **continuous batching**: finished requests leave immediately and freed slots can be refilled

This is useful for constructing traces where continuous batching clearly wins due to better slot utilization, even on a machine where the local model benchmark is dominated by Python/PyTorch overhead.

Latest saved run:

* Scenario: capacity `4`, decode lengths `[16, 4, 4, 4, 16, 4, 4, 4]`
* Raw data: scheduler_simulation_latest.json

Speedup baseline: `static_batch_scheduler` by decode-step makespan

| Policy | Decode steps | Useful token steps | Scheduled slot steps | Utilization | Mean finish step | Max finish step | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_batch_scheduler` | 32 | 56 | 128 | 43.75% | 15.0 | 32 | `1.00x` |
| `continuous_batch_scheduler` | 20 | 56 | 56 | 100.00% | 9.5 | 20 | `1.60x` |

Main takeaways:

* Both policies do the same `56` units of useful decode work.
* Static batching wastes iterations on dead slots; continuous batching refills them immediately.
* In this trace, continuous batching reduces makespan by `37.5%` and lowers mean completion step from `15.0` to `9.5`.

## Serving Policy Benchmark

Script `benchmark_serving_policies.py`.

This benchmark uses the local model and compares two end-to-end policies on the same request trace:

* **static policy**: process fixed cohorts of size `capacity` and keep decoding the cohort until its longest request finishes
* **continuous policy**: run the real `Engine` scheduling loop on the same request trace

Unlike the scheduler simulation, this benchmark measures actual model execution time on the local machine.

Latest saved run:

* Capacity: `4`
* Requests: `8`
* Workloads: `uniform`, `moderately_uneven`, `highly_uneven`, `extreme_uneven`
* Raw data: serving_policies_latest.json

Speedup baseline: `static_policy` by wall-clock makespan within each workload

| Workload | Target new tokens | Static wall (s) | Continuous wall (s) | Speedup | Takeaway |
| --- | --- | ---: | ---: | ---: | --- |
| `uniform` | `[8, 8, 8, 8, 8, 8, 8, 8]` | 2.003 | 2.189 | `0.91x` | Little to no benefit; engine overhead slightly hurts |
| `moderately_uneven` | `[12, 8, 8, 8, 12, 8, 8, 8]` | 2.879 | 2.645 | `1.09x` | Mild skew gives a modest makespan win |
| `highly_uneven` | `[16, 4, 4, 4, 16, 4, 4, 4]` | 3.641 | 2.495 | `1.46x` | Strong skew makes refillable batching clearly better |
| `extreme_uneven` | `[32, 4, 4, 4, 32, 4, 4, 4]` | 7.565 | 4.382 | `1.73x` |  |

Main takeaways:

* The continuous-policy row now corresponds to the real `Engine` path rather than a separate handwritten approximation.
* On uniform workloads, continuous batching does not help much and can be slightly worse because there are no dead slots to refill.
* As request lengths become more uneven, continuous batching improves makespan more noticeably: about `1.09x` on the moderate trace, `1.46x` on the highly uneven trace, and `1.73x` on the stress-test trace.


## Run

To run the inference-mode benchmark:

```bash
uv run python benchmark/benchmark_inference_modes.py --device auto --max-new-tokens 4 --repeats 2 --warmup 1
```

Results are written to `benchmark/results/inference_modes_latest.json` by default.

To run the scheduler simulation:

```bash
python3 benchmark/simulate_scheduler.py --capacity 4 --lengths 16 4 4 4 16 4 4 4
```

Results are written to `benchmark/results/scheduler_simulation_latest.json` by default.

To run the serving-policy benchmark:

```bash
uv run python benchmark/benchmark_serving_policies.py --device auto --capacity 4 --repeats 2 --warmup 1
```

Results are written to `benchmark/results/serving_policies_latest.json` by default.
