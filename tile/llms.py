"""The LLMs of TILE: download them, cut them at their layer, and load them."""

from dataclasses import dataclass
from glob import glob


@dataclass(frozen=True)
class LLM:
    repo_id: str
    # The number of transformer blocks kept: the output of the last one is
    # one of the layers whose cell embeddings predicted best in the layer
    # probing of the paper
    layer: int


LLMS = {
    "llama-3.1-8b": LLM("meta-llama/Llama-3.1-8B", 14),
    "llama-3.1-70b": LLM("meta-llama/Llama-3.1-70B", 32),
    "llama-3.2-1b": LLM("meta-llama/Llama-3.2-1B", 10),
    "llama-3.2-3b": LLM("meta-llama/Llama-3.2-3B", 11),
    "qwen3-0.6b-base": LLM("Qwen/Qwen3-0.6B-Base", 16),
    "qwen3-4b-base": LLM("Qwen/Qwen3-4B-Base", 20),
    "qwen3-8b-base": LLM("Qwen/Qwen3-8B-Base", 19),
}
DEFAULT_LLM = "llama-3.1-8b"


def download(repo_id, cache_dir=None):
    """Download an LLM from the Hugging Face hub, and return its local path.

    Only the safetensors weights, the configs and the tokenizer: not the
    original checkpoints that the Llama repositories also hold.
    """
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"],
        ignore_patterns=["original/*"],
    )
    # Without its tokenizer files, a snapshot still loads a tokenizer: an
    # empty one, which turns every text into the same few tokens, silently
    if not glob(f"{path}/tokenizer*"):
        raise FileNotFoundError(f"No tokenizer files in {path}")
    return path


def truncate(repo_id, layer, save_dir, cache_dir=None):
    """Save the first `layer` blocks of an LLM, with its tokenizer, in bfloat16."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    path = download(repo_id, cache_dir)
    config = AutoConfig.from_pretrained(path)
    config.num_hidden_layers = layer
    if getattr(config, "layer_types", None) is not None:
        config.layer_types = config.layer_types[: config.num_hidden_layers]
    # The weights of the blocks past the cut are never read
    model = AutoModel.from_pretrained(path, config=config, dtype=torch.bfloat16)
    model.save_pretrained(save_dir)
    AutoTokenizer.from_pretrained(path).save_pretrained(save_dir)


def load(path):
    """A truncated LLM and its tokenizer, on the GPUs if there are any.

    A full LLM loads too, for the layer probing: the output of its last block
    then goes without the final norm, like that of every other block.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")
    # The probing read the raw output of the last block kept: the final norm,
    # which only the output of the last block of the full LLM goes through,
    # is left out
    model.norm = torch.nn.Identity()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model.eval(), tokenizer
