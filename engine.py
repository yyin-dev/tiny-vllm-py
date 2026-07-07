from engine_model import EngineModel
from kv_cache import RequestKVCache
from typing import List, Dict
from enum import Enum
import torch
from transformers import PreTrainedTokenizer


class Request:
    def __init__(self, id: int, prompt: str):
        self.id = id
        self.prompt = prompt


class DecodeReadyRequest:
    def __init__(
        self,
        request: Request,
        kv_cache: RequestKVCache,
        prefill_token: torch.Tensor,  # (1, 1)
    ):
        self.id = request.id
        self.kv_cache = kv_cache
        self.generated_tokens = prefill_token

    def get_prev_token(self) -> torch.Tensor:
        return self.generated_tokens[:, -1:]

    def add_decoded_token(self, token: torch.Tensor):
        self.generated_tokens = torch.cat([self.generated_tokens, token], dim=-1)


class FinishedRequest:
    def __init__(self, decode_ready_request: DecodeReadyRequest):
        self.id = decode_ready_request.id


class Op(Enum):
    Prefill = 1
    Decode = 2
    No_op = 3


class Engine:
    """
    Continuous-batching inference engine.
    """

    def __init__(
        self,
        model: EngineModel,
        tokenizer: PreTrainedTokenizer,
        prefill_threshold: int = 3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.pending_prefill: List[Request] = []
        self.pending_decode: List[DecodeReadyRequest] = []
        self.finished: List[FinishedRequest] = []

        self.prefill_batch_size = 2
        self.decode_batch_size = 4
        self.prefill_threshold = prefill_threshold

    def add_request(self, request: Request):
        self.pending_prefill.append(request)

    def collect_finished_requests(self):
        finished_ids = [req.id for req in self.finished]
        self.finished.clear()
        return finished_ids

    def next_step_op(self) -> Op:
        if len(self.pending_prefill) >= self.prefill_threshold:
            return Op.Prefill

        if len(self.pending_decode) > 0:
            return Op.Decode

        if len(self.pending_prefill) > 0:
            return Op.Prefill

        return Op.No_op

    def select_prefill_requests(self) -> List[Request]:
        """
        Remove requests from `self.pending_prefill`.
        """

        res = self.pending_prefill[: self.prefill_batch_size]
        self.pending_prefill = self.pending_prefill[self.prefill_batch_size :]
        return res

    def select_decode_requests(self) -> List[DecodeReadyRequest]:
        """
        Removes requests from `self.pending_decode`
        """

        res = self.pending_decode[: self.decode_batch_size]
        self.pending_decode = self.pending_decode[self.decode_batch_size :]
        return res

    def step(self) -> Dict[int, str]:
        res = {}

        match self.next_step_op():
            case Op.No_op:
                pass
            case Op.Prefill:
                prefill_requests = self.select_prefill_requests()
                prompts_and_kvs = [
                    (
                        torch.tensor(
                            self.tokenizer([req.prompt]).input_ids,
                            device=self.model.device,
                        ),
                        RequestKVCache(),
                    )
                    for req in prefill_requests
                ]

                prefill_result = self.model.prefill(prompts_and_kvs=prompts_and_kvs)
                assert len(prompts_and_kvs) == len(prefill_result)

                for request, (_encoded_prompt, kv_cache), prefill_token in zip(
                    prefill_requests, prompts_and_kvs, prefill_result
                ):
                    decode_ready_request = DecodeReadyRequest(
                        request, kv_cache, prefill_token
                    )

                    prefill_token_int = int(prefill_token.item())
                    if prefill_token_int == self.tokenizer.eos_token_id:
                        self.finished.append(FinishedRequest(decode_ready_request))
                    else:
                        self.pending_decode.append(decode_ready_request)

                    res[request.id] = self.tokenizer.decode(prefill_token_int)
            case Op.Decode:
                requests = self.select_decode_requests()
                prev_tokens_and_kvs = [
                    (req.get_prev_token(), req.kv_cache) for req in requests
                ]

                decode_result = self.model.decode(
                    prev_tokens_and_kvs=prev_tokens_and_kvs
                )
                assert len(prev_tokens_and_kvs) == len(decode_result)

                unfinished_requests = []
                for request, decode_token in zip(requests, decode_result):
                    request.add_decoded_token(decode_token)

                    decode_token_int = int(decode_token.item())

                    if decode_token_int == self.tokenizer.eos_token_id:
                        self.finished.append(FinishedRequest(request))
                    else:
                        unfinished_requests.append(request)

                    res[request.id] = self.tokenizer.decode(decode_token_int)

                # NOTE: `select_decode_requests` pick requests at the front. Add
                # at the front to preserve the sticky membership.
                self.pending_decode = unfinished_requests + self.pending_decode

        return res
