import argparse
import contextlib
import io
import json
import platform
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
from load_checkpoint import load_model
from model import LlamaLM


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_PROMPTS = [
    "Summarize how KV cache helps autoregressive decoding in one sentence.",
    "Write two short bullet points about the difference between prefill and decode.",
    "Explain why static batching needs attention masks and position ids when prompts have different lengths.",
    "Give a short explanation of why continuous batching is a scheduling technique.",
]




@dataclass
class ScenarioResult:
    name: str
    scenario_group: str
    wall_time_s: float
    new_tokens: int
    prompt_tokens: int
    requests: int
    timed_input_tokens: int | None = None
    steps: int | None = None
    note: str | None = None

    @property
    def new_tokens_per_s(self) -> float:
        return self.new_tokens / self.wall_time_s

    @property
    def total_tokens_per_s(self) -> float:
        return (self.prompt_tokens + self.new_tokens) / self.wall_time_s

    @property
    def timed_input_tokens_per_s(self) -> float | None:
        if self.timed_input_tokens is None:
            return None
        return self.timed_input_tokens / self.wall_time_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/inference_modes_latest.json"),
    )
    return parser.parse_args()


def pick_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS, but torch.backends.mps.is_available() is False")
        return "mps"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


@contextlib.contextmanager
def suppress_stdout():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


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


def encode_single_prompt(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    device: str,
) -> torch.Tensor:
    batch = tokenizer([prompt])
    return torch.tensor(batch.input_ids, device=device)


def encode_prompt_batch(
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = tokenizer(
        prompts,
        padding=True,
        return_attention_mask=True,
    )
    input_ids = torch.tensor(batch.input_ids, device=device)
    attention_mask = torch.tensor(batch.attention_mask, device=device).bool()
    return input_ids, attention_mask


def prompt_token_count(
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
) -> int:
    batch = tokenizer(prompts)
    return sum(len(ids) for ids in batch.input_ids)


def run_singleton_generate(
    model: LlamaLM,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    device: str,
    max_new_tokens: int,
    use_kv_cache: bool,
) -> ScenarioResult:
    total_new_tokens = 0
    total_prompt_tokens = 0

    with suppress_stdout():
        for prompt in prompts:
            input_ids = encode_single_prompt(tokenizer, prompt, device)
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_kv_cache=use_kv_cache,
            )
            total_new_tokens += int(output.shape[-1])
            total_prompt_tokens += int(input_ids.shape[-1])

    return ScenarioResult(
        name=f"singleton_{'kv' if use_kv_cache else 'no_kv'}",
        scenario_group="full_generation",
        wall_time_s=0.0,
        new_tokens=total_new_tokens,
        prompt_tokens=total_prompt_tokens,
        requests=len(prompts),
        timed_input_tokens=None,
    )


def run_static_batch_generate(
    model: LlamaLM,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    device: str,
    max_new_tokens: int,
    use_kv_cache: bool,
) -> ScenarioResult:
    input_ids, attention_mask = encode_prompt_batch(tokenizer, prompts, device)

    with suppress_stdout():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            valid_token_mask=attention_mask,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            use_kv_cache=use_kv_cache,
        )

    return ScenarioResult(
        name=f"static_batch_{'kv' if use_kv_cache else 'no_kv'}",
        scenario_group="full_generation",
        wall_time_s=0.0,
        new_tokens=int(output.numel()),
        prompt_tokens=int(attention_mask.sum().item()),
        requests=len(prompts),
        timed_input_tokens=None,
    )


def run_prefill_only(
    model: LlamaLM,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    device: str,
) -> ScenarioResult:
    prompts_and_kvs = []
    for prompt in prompts:
        input_ids = encode_single_prompt(tokenizer, prompt, device)
        from kv_cache import RequestKVCache

        prompts_and_kvs.append((input_ids, RequestKVCache()))

    tokens = model.prefill(prompts_and_kvs)
    return ScenarioResult(
        name="prefill_only",
        scenario_group="microbenchmark",
        wall_time_s=0.0,
        new_tokens=len(tokens),
        prompt_tokens=prompt_token_count(tokenizer, prompts),
        requests=len(prompts),
        timed_input_tokens=prompt_token_count(tokenizer, prompts),
        note="Measures one batched prefill call that emits the first token for each request.",
    )


def run_decode_only(
    model: LlamaLM,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    device: str,
) -> ScenarioResult:
    from kv_cache import RequestKVCache

    prompts_and_kvs = []
    for prompt in prompts:
        input_ids = encode_single_prompt(tokenizer, prompt, device)
        prompts_and_kvs.append((input_ids, RequestKVCache()))

    prev_tokens = model.prefill(prompts_and_kvs)
    decode_inputs = list(zip(prev_tokens, [cache for _, cache in prompts_and_kvs], strict=True))
    synchronize(device)
    start = time.perf_counter()
    tokens = model.decode(decode_inputs)
    synchronize(device)
    end = time.perf_counter()
    return ScenarioResult(
        name="decode_only",
        scenario_group="microbenchmark",
        wall_time_s=end - start,
        new_tokens=len(tokens),
        prompt_tokens=prompt_token_count(tokenizer, prompts),
        requests=len(prompts),
        timed_input_tokens=len(tokens),
        note="Measures one batched decode iteration after caches have already been populated by prefill.",
    )


def run_continuous_batch_kv(
    model: LlamaLM,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> ScenarioResult:
    engine = Engine(model=model, tokenizer=tokenizer, prefill_threshold=1)
    engine.prefill_batch_size = len(prompts)
    engine.decode_batch_size = len(prompts)

    for request_id, prompt in enumerate(prompts):
        engine.add_request(Request(request_id, prompt))

    generated_counts = {request_id: 0 for request_id in range(len(prompts))}
    finished_ids: set[int] = set()
    steps = 0

    while True:
        outputs = engine.step()
        steps += 1
        for request_id in outputs:
            generated_counts[request_id] += 1

        finished_ids.update(engine.collect_finished_requests())

        if all(
            generated_counts[request_id] >= max_new_tokens or request_id in finished_ids
            for request_id in generated_counts
        ):
            break

    return ScenarioResult(
        name="continuous_batch_kv",
        scenario_group="full_generation",
        wall_time_s=0.0,
        new_tokens=sum(generated_counts.values()),
        prompt_tokens=prompt_token_count(tokenizer, prompts),
        requests=len(prompts),
        timed_input_tokens=None,
        steps=steps,
        note="Uses the current Engine implementation with all requests arriving at time 0.",
    )


def time_scenario(device: str, fn, *args, **kwargs) -> ScenarioResult:
    synchronize(device)
    start = time.perf_counter()
    result: ScenarioResult = fn(*args, **kwargs)
    synchronize(device)
    end = time.perf_counter()
    if result.wall_time_s == 0.0:
        result.wall_time_s = end - start
    return result


def benchmark_scenario(device: str, repeats: int, warmup: int, fn, *args, **kwargs) -> dict:
    for _ in range(warmup):
        _ = time_scenario(device, fn, *args, **kwargs)

    runs = [time_scenario(device, fn, *args, **kwargs) for _ in range(repeats)]
    wall_times = [run.wall_time_s for run in runs]
    representative = min(runs, key=lambda run: run.wall_time_s)

    return {
        **asdict(representative),
        "new_tokens_per_s": representative.new_tokens_per_s,
        "total_tokens_per_s": representative.total_tokens_per_s,
        "timed_input_tokens_per_s": representative.timed_input_tokens_per_s,
        "all_wall_times_s": wall_times,
        "median_wall_time_s": statistics.median(wall_times),
        "min_wall_time_s": min(wall_times),
    }


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    model, tokenizer = load_runtime(device)
    prompts = DEFAULT_PROMPTS

    scenarios = [
        benchmark_scenario(
            device,
            args.repeats,
            args.warmup,
            run_singleton_generate,
            model,
            tokenizer,
            prompts,
            device,
            args.max_new_tokens,
            False,
        ),
        benchmark_scenario(
            device,
            args.repeats,
            args.warmup,
            run_singleton_generate,
            model,
            tokenizer,
            prompts,
            device,
            args.max_new_tokens,
            True,
        ),
        benchmark_scenario(
            device,
            args.repeats,
            args.warmup,
            run_static_batch_generate,
            model,
            tokenizer,
            prompts,
            device,
            args.max_new_tokens,
            False,
        ),
        benchmark_scenario(
            device,
            args.repeats,
            args.warmup,
            run_static_batch_generate,
            model,
            tokenizer,
            prompts,
            device,
            args.max_new_tokens,
            True,
        ),
        benchmark_scenario(
            device,
            args.repeats,
            args.warmup,
            run_prefill_only,
            model,
            tokenizer,
            prompts,
            device,
        ),
        benchmark_scenario(
            device,
            args.repeats,
            args.warmup,
            run_decode_only,
            model,
            tokenizer,
            prompts,
            device,
        ),
        benchmark_scenario(
            device,
            args.repeats,
            args.warmup,
            run_continuous_batch_kv,
            model,
            tokenizer,
            prompts,
            args.max_new_tokens,
        ),
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "mps_available": torch.backends.mps.is_available(),
        "prompts": prompts,
        "max_new_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "scenarios": scenarios,
        "notes": [
            "These measurements are relative comparisons on the local machine, not production GPU benchmarks.",
            "The local implementation is Python/PyTorch and does not use custom CUDA kernels.",
            "continuous_batch_kv measures the current Engine path with all requests submitted up front.",
            "Only the full_generation scenarios should be interpreted as direct speedup peers.",
            "prefill_only and decode_only are stage-level microbenchmarks and do not perform the same total work as full generation.",
        ],
    }

    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
