import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_LENGTHS = [16, 4, 4, 4, 16, 4, 4, 4]


@dataclass
class RequestState:
    request_id: int
    arrival_step: int
    decode_length: int
    remaining: int
    finish_step: int | None = None


@dataclass
class SchedulerResult:
    name: str
    total_decode_steps: int
    useful_token_steps: int
    scheduled_slot_steps: int
    utilization: float
    mean_finish_step: float
    max_finish_step: int
    completion_steps: list[int]
    scheduled_slots_per_step: list[int]
    useful_tokens_per_step: list[int]
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/scheduler_simulation_latest.json"),
    )
    return parser.parse_args()


def build_requests(lengths: list[int]) -> list[RequestState]:
    return [
        RequestState(
            request_id=i,
            arrival_step=0,
            decode_length=length,
            remaining=length,
        )
        for i, length in enumerate(lengths)
    ]


def finalize_result(
    name: str,
    requests: list[RequestState],
    scheduled_slots_per_step: list[int],
    useful_tokens_per_step: list[int],
    note: str,
) -> SchedulerResult:
    completion_steps = [req.finish_step for req in requests]
    assert all(step is not None for step in completion_steps)
    completion_steps = [int(step) for step in completion_steps]

    useful = sum(useful_tokens_per_step)
    scheduled = sum(scheduled_slots_per_step)
    return SchedulerResult(
        name=name,
        total_decode_steps=len(scheduled_slots_per_step),
        useful_token_steps=useful,
        scheduled_slot_steps=scheduled,
        utilization=useful / scheduled,
        mean_finish_step=statistics.mean(completion_steps),
        max_finish_step=max(completion_steps),
        completion_steps=completion_steps,
        scheduled_slots_per_step=scheduled_slots_per_step,
        useful_tokens_per_step=useful_tokens_per_step,
        note=note,
    )


def simulate_static_batching(
    lengths: list[int],
    capacity: int,
) -> SchedulerResult:
    requests = build_requests(lengths)
    pending = requests[:]
    step = 0
    scheduled_slots_per_step: list[int] = []
    useful_tokens_per_step: list[int] = []

    while pending:
        batch = pending[:capacity]
        pending = pending[capacity:]
        batch_slots = len(batch)

        while any(req.remaining > 0 for req in batch):
            useful_this_step = 0
            for req in batch:
                if req.remaining > 0:
                    req.remaining -= 1
                    useful_this_step += 1
                    if req.remaining == 0:
                        req.finish_step = step + 1

            scheduled_slots_per_step.append(batch_slots)
            useful_tokens_per_step.append(useful_this_step)
            step += 1

    return finalize_result(
        name="static_batch_scheduler",
        requests=requests,
        scheduled_slots_per_step=scheduled_slots_per_step,
        useful_tokens_per_step=useful_tokens_per_step,
        note="Batch membership is fixed until the whole batch completes. Finished requests keep occupying scheduled slots.",
    )


def simulate_continuous_batching(
    lengths: list[int],
    capacity: int,
) -> SchedulerResult:
    requests = build_requests(lengths)
    pending = requests[:]
    active: list[RequestState] = []
    step = 0
    scheduled_slots_per_step: list[int] = []
    useful_tokens_per_step: list[int] = []

    while pending or active:
        while pending and len(active) < capacity:
            active.append(pending.pop(0))

        useful_this_step = 0
        for req in active:
            req.remaining -= 1
            useful_this_step += 1
            if req.remaining == 0:
                req.finish_step = step + 1

        scheduled_slots_per_step.append(len(active))
        useful_tokens_per_step.append(useful_this_step)
        active = [req for req in active if req.remaining > 0]
        step += 1

    return finalize_result(
        name="continuous_batch_scheduler",
        requests=requests,
        scheduled_slots_per_step=scheduled_slots_per_step,
        useful_tokens_per_step=useful_tokens_per_step,
        note="Finished requests are removed immediately and freed slots are refilled on the next decode iteration.",
    )


def main() -> None:
    args = parse_args()
    started = time.strftime("%Y-%m-%d %H:%M:%S")

    static_result = simulate_static_batching(args.lengths, args.capacity)
    continuous_result = simulate_continuous_batching(args.lengths, args.capacity)

    payload = {
        "timestamp": started,
        "capacity": args.capacity,
        "lengths": args.lengths,
        "results": [
            asdict(static_result),
            asdict(continuous_result),
        ],
        "notes": [
            "This benchmark is a scheduler simulation, not a model-throughput benchmark.",
            "Each request contributes one token of useful decode work per iteration until its decode_length is exhausted.",
            "The point is to isolate the effect of fixed batch membership vs. refillable batch membership.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
