"""Add an LLM to TILE: probe its layers, and cut it at the best one.

    python scripts/add_llm.py --name mistral-7b --repo_id mistralai/Mistral-7B-v0.3

Every layer of the LLM is probed on the 27 probing tables of the TILE
benchmark (see `tile.probing`), which `--data_dir` holds. The score of each
table is saved to `<results_dir>/probing/<name>/<table>.csv`, and a table
already probed is skipped. The LLM is then cut at the layer of the best mean
score, and saved to `<models_dir>/<name>`, with the scores of its layers in
`probing.json`: `--llm <name>` then runs TILE with it.

The full LLM is loaded, spread over the visible GPUs. Every layer of the rows
of a batch is held on the GPU at once: lower `--max_tokens` if it runs out of
memory. The embeddings of every layer of a table are held in memory, or on
disk in `--tmp_dir`: up to (layers x string columns x 10,240 rows x hidden
size) float16 values, 21 GB for Llama-3.1-8B on the largest table.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tile.data import load_task
from tile.llms import download, load, truncate
from tile.probing import PROBING_TABLES, best_layer, probe_layers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="Name of the LLM in TILE.")
    parser.add_argument(
        "--repo_id", required=True, help="The LLM on the Hugging Face hub."
    )
    parser.add_argument(
        "--data_dir", default="data/tile", help="The TILE benchmark (default: data/tile)."
    )
    parser.add_argument(
        "--step", type=int, default=1, help="Probe every `step`-th layer (default: 1)."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Token budget of a batch of rows, padding included (default: 8192).",
    )
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument(
        "--tmp_dir",
        default=None,
        help="Where to hold the embeddings of every layer (default: in memory).",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Hugging Face cache of the full model (default: the hub's own).",
    )
    args = parser.parse_args()

    save_dir = Path(args.models_dir) / args.name
    if (save_dir / "config.json").exists():
        raise SystemExit(f"{save_dir} already exists")

    results_dir = Path(args.results_dir) / "probing" / args.name
    results_dir.mkdir(parents=True, exist_ok=True)
    model = None
    results = []
    for name in PROBING_TABLES:
        path = results_dir / f"{name}.csv"
        if not path.exists():
            if model is None:
                import torch

                # cuDNN's attention kernel fails on padded batches in some
                # environments
                torch.backends.cuda.enable_cudnn_sdp(False)
                model, tokenizer = load(download(args.repo_id, args.cache_dir))
                layers = list(
                    range(args.step, model.config.num_hidden_layers + 1, args.step)
                )
            print(f"Probing {len(layers)} layers of {args.name} on {name}")
            table, target, task, _ = load_task(Path(args.data_dir) / name)
            y = table[target].to_numpy()
            if task == "classification":
                _, y = np.unique(y, return_inverse=True)
            result = probe_layers(
                table.drop(columns=target),
                y,
                task,
                model,
                tokenizer,
                layers,
                args.max_tokens,
                args.tmp_dir,
            )
            result.insert(0, "table", name)
            result.to_csv(path, index=False)
        results.append(pd.read_csv(path))
    del model

    layer, mean = best_layer(pd.concat(results))
    print("Mean score (R² clipped at 0, or ROC AUC) of each layer:")
    print(mean.round(4).to_string())
    print(f"Truncating {args.name} to its first {layer} layers")
    truncate(args.repo_id, int(layer), save_dir, args.cache_dir)
    with open(save_dir / "probing.json", "w") as f:
        json.dump(
            {
                "repo_id": args.repo_id,
                "layer": int(layer),
                "scores": {int(k): round(v, 6) for k, v in mean.items()},
            },
            f,
            indent=2,
        )
    print(
        "To list it among the LLMs of TILE, add to `LLMS` in tile/llms.py:\n"
        f'    "{args.name}": LLM("{args.repo_id}", {layer}),'
    )
