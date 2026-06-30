import torch
import logging
from model import LlamaLM
from einops import rearrange
from transformers import PreTrainedTokenizer, AutoTokenizer
from load_checkpoint import load_model

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct"
    )

    input = "How are you?"
    encoded = torch.tensor(tokenizer.encode(input), device=device)
    encoded = rearrange(encoded, "(b seq) -> b seq", b=1)
    print(encoded)

    llama = load_model().to(device)
    llama.eval()

    logger.info("Start decoding...")

    # TODO: pass an explicit attention_mask once we start padding/batching inputs.
    # For this single unpadded prompt the outputs match the reference, but later
    # batching and KV-cache work should not rely on Transformers inferring it.
    res = llama.generate(
        encoded,
        max_new_tokens=16,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        use_kv_cache=False,
    )

    print(tokenizer.decode(res))


if __name__ == "__main__":
    main()
