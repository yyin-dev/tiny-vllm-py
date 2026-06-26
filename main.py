import safetensors
import model
import logging
import json
from model import LlamaLM

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


WEIGHTS_PATH = "/Users/yy0125/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/model.safetensors"
CONFIG_PATH = "/Users/yy0125/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/config.json"


def load_model() -> LlamaLM:
    """
    Llama 3.2 1B Instruct architecture

    ```
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    print(model)
    ```
    Output:
    LlamaForCausalLM(
    (model): LlamaModel(
        (embed_tokens): Embedding(128256, 2048)
        (layers): ModuleList(
        (0-15): 16 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (k_proj): Linear(in_features=2048, out_features=512, bias=False)
            (v_proj): Linear(in_features=2048, out_features=512, bias=False)
            (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
            )
            (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
            (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
            (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
            (act_fn): SiLUActivation()
            )
            (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
            (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
        )
        )
        (norm): LlamaRMSNorm((2048,), eps=1e-05)
        (rotary_emb): LlamaRotaryEmbedding()
    )
    (lm_head): Linear(in_features=2048, out_features=128256, bias=False)
    )

    Diagram: https://www.linkedin.com/posts/sebastianraschka_the-llama-32-1b-and-3b-models-are-my-favorite-share-7248317827831455744-EmVt/
    """

    """
    (['token_embeddings.weight', 
     'layers.0.attn.q_proj.weight', 
     'layers.0.attn.k_proj.weight', 
     'layers.0.attn.v_proj.weight', 
     'layers.0.attn.output_proj.weight', 
     'layers.0.ffn.w1.weight', 
     'layers.0.ffn.w2.weight', 
     'layers.0.ffn.w3.weight', 
     'layers.0.ln1.weight',
     'layers.0.ln2.weight',
     ...
     'ln_final.weight', 'lm_head.weight'])
    """
    logger.info("Initializing Llama Model...")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
        vocab_size = config["vocab_size"]
        d_model = config["hidden_size"]
        num_layers = config["num_hidden_layers"]
        num_q_heads = config["num_attention_heads"]
        num_kv_heads = config["num_kv_heads"]
        d_ff = config["intermediate_size"]
        rope_theta = config["rope_theta"]
        context_length = config["max_position_embeddings"]

    # NOTE: to fully match the model, need to handle RoPE more correctly. Things
    # like: factor, high_freq_factor, low_freq_factor, rope_type etc. But it
    # seems that, as long as `rope_theta` is set correctly, the behavior is
    # correct for short sequences.
    llama = model.LlamaLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    state_dict = llama.state_dict()

    """
    ['model.embed_tokens.weight', 
     'model.layers.0.input_layernorm.weight', 
     'model.layers.0.mlp.down_proj.weight', 
     'model.layers.0.mlp.gate_proj.weight', 
     'model.layers.0.mlp.up_proj.weight', 
     'model.layers.0.post_attention_layernorm.weight', 
     'model.layers.0.self_attn.k_proj.weight', 
     'model.layers.0.self_attn.o_proj.weight', 
     'model.layers.0.self_attn.q_proj.weight', 
     'model.layers.0.self_attn.v_proj.weight', 
     ... [up until model.layers.15.xxx]
     'model.norm.weight'
    ]
    """
    logger.info("Loading from checkpoint...")
    loaded_checkpoint = {}
    with safetensors.safe_open(WEIGHTS_PATH, framework="pt") as f:

        def set(ckpt_name, state_dict_name):
            ckpt_weights = f.get_tensor(ckpt_name)
            state_dict_weights = state_dict[state_dict_name]
            assert ckpt_weights.shape == state_dict_weights.shape
            loaded_checkpoint[state_dict_name] = ckpt_weights

        # Technically it's sufficient to only set "token_embedding.weights" because
        # the two weights are tied. However, set both here s.t. we can use
        # load_state_dict(strict=True)
        set("model.embed_tokens.weight", "token_embeddings.weight")
        set("model.embed_tokens.weight", "lm_head.weight")

        for l in range(num_layers):
            set(f"model.layers.{l}.input_layernorm.weight", f"layers.{l}.ln1.weight")
            set(
                f"model.layers.{l}.self_attn.q_proj.weight",
                f"layers.{l}.attn.q_proj.weight",
            )
            set(
                f"model.layers.{l}.self_attn.k_proj.weight",
                f"layers.{l}.attn.k_proj.weight",
            )
            set(
                f"model.layers.{l}.self_attn.v_proj.weight",
                f"layers.{l}.attn.v_proj.weight",
            )
            set(
                f"model.layers.{l}.self_attn.o_proj.weight",
                f"layers.{l}.attn.output_proj.weight",
            )
            set(
                f"model.layers.{l}.post_attention_layernorm.weight",
                f"layers.{l}.ln2.weight",
            )
            set(f"model.layers.{l}.mlp.up_proj.weight", f"layers.{l}.ffn.w3.weight")
            set(f"model.layers.{l}.mlp.gate_proj.weight", f"layers.{l}.ffn.w1.weight")
            set(f"model.layers.{l}.mlp.down_proj.weight", f"layers.{l}.ffn.w2.weight")

        set("model.norm.weight", "ln_final.weight")

    logger.info("Set model weights based on checkpoint")
    llama.load_state_dict(loaded_checkpoint, strict=True)

    logger.info("Model loaded with weights from checkpoint")
    return llama


def main():
    llama = load_model()


if __name__ == "__main__":
    main()
