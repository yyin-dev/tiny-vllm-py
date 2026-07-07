import torch
import pytest
from transformers import (
    PreTrainedTokenizer,
    AutoTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
import logging


import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from load_checkpoint import load_model
from model import LlamaLM

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@pytest.fixture(scope="session")
def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


@pytest.fixture(scope="session")
def local_model(device: str) -> LlamaLM:
    model: LlamaLM = load_model()
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


@pytest.fixture(scope="session")
def local_model_cpu() -> LlamaLM:
    model: LlamaLM = load_model()
    model = model.to("cpu").eval()
    model.requires_grad_(False)
    return model


@pytest.fixture(scope="session")
def reference_model(device: str) -> LlamaForCausalLM:
    logger.info("Start loading reference model")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
    )
    model = model.to(device).eval()
    logger.info("Finished loading reference model")

    # When running on MPS, upcast to fp32 to avoid
    # bf16 precision issue. See debug/debug_hf_generate.py for details.
    if device == "mps":
        model = model.float()

    model.requires_grad_(False)
    return model


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizer:
    logger.info("Start loading tokenizer")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
    )
    logger.info("Finished loading tokenizer")
    return tokenizer
