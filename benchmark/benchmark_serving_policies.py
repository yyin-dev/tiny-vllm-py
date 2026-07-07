import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from engine import Engine, Request
from kv_cache import RequestKVCache
from load_checkpoint import load_model
from model import LlamaLM


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_CAPACITY = 4
DEFAULT_PROMPTS = [
    "Explain how KV cache changes the cost of autoregressive decoding in two sentences.",
    "Define KV cache.",
    "What is prefill?",
    "What is decode?",
    "Describe the difference between static batching and continuous batching for an inference server.",
    "What is TTFT?",
    "What is inter-token latency?",
    "Why do attention masks matter with padding?",
]
WORKLOADS = [
    (
        "uniform",
        [8, 8, 8, 8, 8, 8, 8, 8],
        "All requests have the same target length, so static and continuous batching should behave similarly.",
    ),
    (
        "moderately_uneven",
        [12, 8, 8, 8, 12, 8, 8, 8],
        "A mildly skewed trace where some requests run longer but the gap is limited.",
    ),
    (
        "highly_uneven",
        [16, 4, 4, 4, 16, 4, 4, 4],
        "A skewed trace where short requests leave dead slots behind under static batching.",
    ),
    (
        "extreme_uneven",
        [32, 4, 4, 4, 32, 4, 4, 4],
        "A stress-test trace where one long request shares each cohort with three very short requests.",
    ),
]


@dataclass
class RequestSpec:
    request_id: int
    prompt: str
    target_new_tokens: int


@dataclass
class DecodeState:
    spec: RequestSpec
    kv_cache: RequestKVCache
    prev_token: torch.Tensor
    generated_tokens: int
    completion_time_s: float | None = None


@dataclass
class PolicyRunResult:
    name: str
    wall_time_s: float
    requests: int
    capacity: int
    prompt_tokens: int
    total_target_new_tokens: int
    mean_completion_time_s: float
    max_completion_time_s: float
    completion_times_s: list[float]
    note: str


@dataclass
class WorkloadResult:
    name: str
    target_new_tokens: list[int]
    note: str
    results: list[dict]


@dataclass
class EngineRequestState:
    spec: RequestSpec
    completion_time_s: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/serving_policies_latest.json"),
    )
    return parser.parse_args()


def pick_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS, but MPS is unavailable")
        return "mps"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def load_runtime(device: str) -> tuple[LlamaLM, PreTrainedTokenizer]:
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model: LlamaLM = load_model()
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.device = torch.device(device)
    return model, tokenizer


def build_request_specs(target_lengths: list[int]) -> list[RequestSpec]:
    return [
        RequestSpec(
            request_id=i,
            prompt=DEFAULT_PROMPTS[i],
            target_new_tokens=target_lengths[i],
        )
        for i in range(len(DEFAULT_PROMPTS))
    ]


def prompt_token_count(
    tokenizer: PreTrainedTokenizer,
    requests: list[RequestSpec],
) -> int:
    batch = tokenizer([req.prompt for req in requests])
    return sum(len(ids) for ids in batch.input_ids)


def prefill_batch(
    model: LlamaLM,
    tokenizer: PreTrainedTokenizer,
    requests: list[RequestSpec],
    device: str,
    start_time: float,
) -> tuple[list[DecodeState], list[DecodeState]]:
    prompts_and_kvs: list[tuple[torch.Tensor, RequestKVCache]] = []
    caches: list[RequestKVCache] = []
    for spec in requests:
        batch = tokenizer([spec.prompt])
        input_ids = torch.tensor(batch.input_ids, device=device)
        cache = RequestKVCache()
        prompts_and_kvs.append((input_ids, cache))
        caches.append(cache)

    prefill_tokens = model.prefill(prompts_and_kvs)
    now = time.perf_counter()

    finished: list[DecodeState] = []
    decode_ready: list[DecodeState] = []
    for spec, cache, prefill_token in zip(requests, caches, prefill_tokens, strict=True):
        state = DecodeState(
            spec=spec,
            kv_cache=cache,
            prev_token=prefill_token,
            generated_tokens=1,
        )
        if spec.target_new_tokens <= 1:
            state.completion_time_s = now - start_time
            finished.append(state)
        else:
            decode_ready.append(state)

    return decode_ready, finished


def finalize_policy_result(
    name: str,
    wall_time_s: float,
    requests: list[RequestSpec],
    capacity: int,
    tokenizer: PreTrainedTokenizer,
    finished_states: list[DecodeState],
    note: str,
) -> PolicyRunResult:
    completion_times = [state.completion_time_s for state in finished_states]
    assert all(t is not None for t in completion_times)
    completion_times = [float(t) for t in completion_times]
    completion_times.sort()

    return PolicyRunResult(
        name=name,
        wall_time_s=wall_time_s,
        requests=len(requests),
        capacity=capacity,
        prompt_tokens=prompt_token_count(tokenizer, requests),
        total_target_new_tokens=sum(req.target_new_tokens for req in requests),
        mean_completion_time_s=statistics.mean(completion_times),
        max_completion_time_s=max(completion_times),
        completion_times_s=completion_times,
        note=note,
    )


def remove_completed_engine_requests(engine: Engine, completed_ids: set[int]) -> None:
    engine.pending_decode = [
        req for req in engine.pending_decode if req.id not in completed_ids
    ]


def run_static_policy(
    model: LlamaLM,
    tokenizer: PreTrainedTokenizer,
    requests: list[RequestSpec],
    device: str,
    capacity: int,
) -> PolicyRunResult:
    start_time = time.perf_counter()
    finished_states: list[DecodeState] = []

    for batch_start in range(0, len(requests), capacity):
        batch_specs = requests[batch_start : batch_start + capacity]
        decode_states, prefill_finished = prefill_batch(
            model=model,
            tokenizer=tokenizer,
            requests=batch_specs,
            device=device,
            start_time=start_time,
        )
        finished_states.extend(prefill_finished)

        if not decode_states:
            continue

        max_target = max(state.spec.target_new_tokens for state in decode_states)

        while any(state.generated_tokens < max_target for state in decode_states):
            decode_inputs = [
                (state.prev_token, state.kv_cache) for state in decode_states
            ]
            outputs = model.decode(decode_inputs)
            now = time.perf_counter()

            for state, output in zip(decode_states, outputs, strict=True):
                state.prev_token = output
                if state.generated_tokens < max_target:
                    state.generated_tokens += 1

                if (
                    state.completion_time_s is None
                    and state.generated_tokens >= state.spec.target_new_tokens
                ):
                    state.completion_time_s = now - start_time

        finished_states.extend(decode_states)

    wall_time_s = time.perf_counter() - start_time
    return finalize_policy_result(
        name="static_policy",
        wall_time_s=wall_time_s,
        requests=requests,
        capacity=capacity,
        tokenizer=tokenizer,
        finished_states=finished_states,
        note="Processes fixed cohorts of size capacity and keeps decoding each cohort until its longest request completes.",
    )


def run_continuous_policy(
    model: LlamaLM,
    tokenizer: PreTrainedTokenizer,
    requests: list[RequestSpec],
    device: str,
    capacity: int,
) -> PolicyRunResult:
    engine = Engine(model=model, tokenizer=tokenizer, prefill_threshold=capacity)
    engine.prefill_batch_size = capacity
    engine.decode_batch_size = capacity

    request_states = {
        req.request_id: EngineRequestState(spec=req) for req in requests
    }
    generated_counts = {req.request_id: 0 for req in requests}
    benchmark_completed_ids: set[int] = set()

    for req in requests:
        engine.add_request(Request(req.request_id, req.prompt))

    start_time = time.perf_counter()
    while len(benchmark_completed_ids) < len(requests):
        outputs = engine.step()
        now = time.perf_counter()

        for request_id in outputs:
            generated_counts[request_id] += 1
            state = request_states[request_id]
            if (
                state.completion_time_s is None
                and generated_counts[request_id] >= state.spec.target_new_tokens
            ):
                state.completion_time_s = now - start_time
                benchmark_completed_ids.add(request_id)

        eos_finished_ids = set(engine.collect_finished_requests())
        for request_id in eos_finished_ids:
            if request_id in benchmark_completed_ids:
                continue
            state = request_states[request_id]
            state.completion_time_s = now - start_time
            benchmark_completed_ids.add(request_id)

        if benchmark_completed_ids:
            remove_completed_engine_requests(engine, benchmark_completed_ids)

    wall_time_s = time.perf_counter() - start_time
    finished_states = [
        DecodeState(
            spec=state.spec,
            kv_cache=RequestKVCache(),
            prev_token=torch.empty((1, 1), device=device, dtype=torch.long),
            generated_tokens=generated_counts[state.spec.request_id],
            completion_time_s=state.completion_time_s,
        )
        for state in request_states.values()
    ]
    return finalize_policy_result(
        name="continuous_policy",
        wall_time_s=wall_time_s,
        requests=requests,
        capacity=capacity,
        tokenizer=tokenizer,
        finished_states=finished_states,
        note="Uses the real Engine scheduling loop and treats each request as benchmark-complete once it reaches its target token count.",
    )


def time_policy(device: str, fn, *args) -> PolicyRunResult:
    synchronize(device)
    result = fn(*args)
    synchronize(device)
    return result


def benchmark_policy(device: str, repeats: int, warmup: int, fn, *args) -> dict:
    for _ in range(warmup):
        _ = time_policy(device, fn, *args)

    runs = [time_policy(device, fn, *args) for _ in range(repeats)]
    representative = min(runs, key=lambda run: run.wall_time_s)
    wall_times = [run.wall_time_s for run in runs]

    return {
        **asdict(representative),
        "all_wall_times_s": wall_times,
        "median_wall_time_s": statistics.median(wall_times),
        "min_wall_time_s": min(wall_times),
    }


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    model, tokenizer = load_runtime(device)
    workloads: list[dict] = []
    for workload_name, target_lengths, workload_note in WORKLOADS:
        requests = build_request_specs(target_lengths)
        results = [
            benchmark_policy(
                device,
                args.repeats,
                args.warmup,
                run_static_policy,
                model,
                tokenizer,
                requests,
                device,
                args.capacity,
            ),
            benchmark_policy(
                device,
                args.repeats,
                args.warmup,
                run_continuous_policy,
                model,
                tokenizer,
                requests,
                device,
                args.capacity,
            ),
        ]
        workloads.append(
            asdict(
                WorkloadResult(
                    name=workload_name,
                    target_new_tokens=target_lengths,
                    note=workload_note,
                    results=results,
                )
            )
        )

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "capacity": args.capacity,
        "prompts": DEFAULT_PROMPTS,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "workloads": workloads,
        "notes": [
            "This benchmark measures real wall-clock time on the local model for two serving policies.",
            "Within each workload, both policies use the same prompts and the same per-request target output lengths.",
            "The purpose is to show that refillable decode membership helps more as request lengths become more uneven.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
