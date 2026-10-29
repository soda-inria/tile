"""Download LLMs from the Hugging Face hub and cut them at their layer.

    python scripts/download_llms.py                     # TILE's LLM, Llama-3.1-8B
    python scripts/download_llms.py --llms llama-3.2-1b qwen3-0.6b-base
    python scripts/download_llms.py --llms all          # the 7 LLMs of the paper

The truncated models are saved to `<models_dir>/<llm>`. The Llama models are
gated: accept their license on the hub, and log in (`hf auth login`) first.
"""

import argparse
from pathlib import Path

from tile.llms import DEFAULT_LLM, LLMS, truncate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--llms",
        nargs="+",
        default=[DEFAULT_LLM],
        choices=list(LLMS) + ["all"],
        help=f"LLMs to download, or `all` (default: {DEFAULT_LLM}).",
    )
    parser.add_argument("--models_dir", default="models")
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Hugging Face cache of the full models (default: the hub's own).",
    )
    args = parser.parse_args()

    for name in list(LLMS) if "all" in args.llms else args.llms:
        save_dir = Path(args.models_dir) / name
        if (save_dir / "config.json").exists():
            print(f"{save_dir} already exists, skipping")
            continue
        print(f"Truncating {name} to its first {LLMS[name].layer} layers")
        truncate(LLMS[name].repo_id, LLMS[name].layer, save_dir, args.cache_dir)
