import os
import sys
from collections import deque

import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from engine import Engine, Op, Request, DecodeReadyRequest
from engine_model import EngineModel
from kv_cache import RequestKVCache


class FakeBatchEncoding:
    def __init__(self, input_ids: list[list[int]]):
        self.input_ids = input_ids


class FakeTokenizer:
    """
    Minimum interface to mimic `transformers.PreTrainedTokenizer`.
    """

    def __init__(self):
        self.eos_token_id = 0

    def __call__(self, prompts: list[str]) -> FakeBatchEncoding:
        """
        Each prompt encodes to `len(prompt)`.
        """
        input_ids = [[len(prompt)] for prompt in prompts]
        return FakeBatchEncoding(input_ids=input_ids)

    def decode(self, token_id: int) -> str:
        return f"tok_{token_id}"


class FakeEngineModel(EngineModel):
    """
    A fake model for engine tests.

    `prefill_outputs` and `decode_outputs` are queues of model-return batches.
    Each inner list corresponds to one call, and each integer is the token id
    returned for the matching request in that batch.

    Example:
    - `prefill_outputs=[[5]]` means the next `prefill(...)` call should return
      token `5` for its single request.
    - `decode_outputs=[[8, 9]]` means the next `decode(...)` call should return
      token `8` for the first request in the batch and token `9` for the
      second request.

    Internally records calls to verify in tests.
    """

    def __init__(
        self,
        prefill_outputs: list[list[int]] | None = None,
        decode_outputs: list[list[int]] | None = None,
    ):
        self.device = torch.device("cpu")
        self.prefill_outputs = deque(prefill_outputs or [])
        self.decode_outputs = deque(decode_outputs or [])

        # Record raw call arguments so tests can assert what the engine selected and
        # passed to the model.
        self.prefill_calls: list[list[tuple[torch.Tensor, RequestKVCache]]] = []
        self.decode_calls: list[list[tuple[torch.Tensor, RequestKVCache]]] = []

    def prefill(
        self, prompts_and_kvs: list[tuple[torch.Tensor, RequestKVCache]]
    ) -> list[torch.Tensor]:
        self.prefill_calls.append(prompts_and_kvs)
        outputs = self.prefill_outputs.popleft()
        assert len(outputs) == len(prompts_and_kvs)
        # EngineModel requires each tensor shaped as (1, 1).
        return [torch.tensor([[token]]) for token in outputs]

    def decode(
        self, prev_tokens_and_kvs: list[tuple[torch.Tensor, RequestKVCache]]
    ) -> list[torch.Tensor]:
        self.decode_calls.append(prev_tokens_and_kvs)
        outputs = self.decode_outputs.popleft()
        assert len(outputs) == len(prev_tokens_and_kvs)
        # EngineModel requires each tensor shaped as (1, 1).
        return [torch.tensor([[token]]) for token in outputs]


def make_engine(
    model: EngineModel,
    prefill_threshold: int = 3,
    prefill_batch_size: int = 2,
    decode_batch_size: int = 4,
) -> Engine:
    engine = Engine(
        model=model,
        tokenizer=FakeTokenizer(),
        prefill_threshold=prefill_threshold,
    )
    engine.prefill_batch_size = prefill_batch_size
    engine.decode_batch_size = decode_batch_size
    return engine


def test_next_step_op_no_op_when_no_requests():
    engine = make_engine(FakeEngineModel())

    assert engine.next_step_op() == Op.No_op


def test_next_step_op_prefers_decode_below_threshold():
    engine = make_engine(FakeEngineModel(), prefill_threshold=3)
    engine.pending_prefill = [Request(1, "a"), Request(2, "bb")]
    engine.pending_decode = [
        DecodeReadyRequest(Request(3, "ccc"), RequestKVCache(), torch.tensor([[7]]))
    ]

    assert engine.next_step_op() == Op.Decode


def test_next_step_op_prefill_when_threshold_reached():
    engine = make_engine(FakeEngineModel(), prefill_threshold=2)
    engine.pending_prefill = [Request(1, "a"), Request(2, "bb")]
    engine.pending_decode = [
        DecodeReadyRequest(Request(3, "ccc"), RequestKVCache(), torch.tensor([[7]]))
    ]

    assert engine.next_step_op() == Op.Prefill


def test_step_prefill_moves_non_eos_request_to_pending_decode():
    model = FakeEngineModel(prefill_outputs=[[6]])
    engine = make_engine(model, prefill_threshold=1, prefill_batch_size=1)
    engine.pending_prefill = [Request(1, "hello")]

    result = engine.step()

    assert result == {1: "tok_6"}
    assert len(engine.pending_prefill) == 0
    assert len(engine.pending_decode) == 1
    assert len(engine.finished) == 0

    decode_ready = engine.pending_decode[0]
    assert decode_ready.id == 1
    assert torch.equal(decode_ready.generated_tokens, torch.tensor([[6]]))

    assert len(model.prefill_calls) == 1
    encoded_prompt, _kv_cache = model.prefill_calls[0][0]
    assert torch.equal(encoded_prompt, torch.tensor([[5]]))


def test_step_prefill_moves_eos_request_to_finished():
    model = FakeEngineModel(prefill_outputs=[[0]])
    engine = make_engine(model, prefill_threshold=1, prefill_batch_size=1)
    engine.pending_prefill = [Request(1, "hello")]

    result = engine.step()

    assert result == {1: "tok_0"}
    assert len(engine.pending_decode) == 0
    assert len(engine.finished) == 1
    assert engine.finished[0].id == 1


def test_step_decode_requeues_unfinished_requests_at_front():
    model = FakeEngineModel(decode_outputs=[[8, 9]])
    engine = make_engine(model, decode_batch_size=2)

    first = DecodeReadyRequest(Request(1, "a"), RequestKVCache(), torch.tensor([[5]]))
    second = DecodeReadyRequest(Request(2, "b"), RequestKVCache(), torch.tensor([[6]]))
    waiting = DecodeReadyRequest(Request(3, "c"), RequestKVCache(), torch.tensor([[7]]))
    engine.pending_decode = [first, second, waiting]

    result = engine.step()

    assert result == {1: "tok_8", 2: "tok_9"}
    assert len(engine.finished) == 0
    assert [request.id for request in engine.pending_decode] == [1, 2, 3]
    assert torch.equal(
        engine.pending_decode[0].generated_tokens, torch.tensor([[5, 8]])
    )
    assert torch.equal(
        engine.pending_decode[1].generated_tokens, torch.tensor([[6, 9]])
    )

    assert len(model.decode_calls) == 1
    prev_token_1, _kv_cache_1 = model.decode_calls[0][0]
    prev_token_2, _kv_cache_2 = model.decode_calls[0][1]
    assert torch.equal(prev_token_1, torch.tensor([[5]]))
    assert torch.equal(prev_token_2, torch.tensor([[6]]))


def test_step_decode_moves_eos_request_to_finished():
    model = FakeEngineModel(decode_outputs=[[0, 9]])
    engine = make_engine(model, decode_batch_size=2)

    first = DecodeReadyRequest(Request(1, "a"), RequestKVCache(), torch.tensor([[5]]))
    second = DecodeReadyRequest(Request(2, "b"), RequestKVCache(), torch.tensor([[6]]))
    waiting = DecodeReadyRequest(Request(3, "c"), RequestKVCache(), torch.tensor([[7]]))
    engine.pending_decode = [first, second, waiting]

    result = engine.step()

    assert result == {1: "tok_0", 2: "tok_9"}
    assert [request.id for request in engine.finished] == [1]
    assert [request.id for request in engine.pending_decode] == [2, 3]
    assert torch.equal(
        engine.pending_decode[0].generated_tokens, torch.tensor([[6, 9]])
    )
