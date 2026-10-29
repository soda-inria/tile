"""Evaluate TILE with a tabular learner on a table.

    python scripts/evaluate.py --table data/my_table.parquet --target price \\
        --task regression --learner tabpfn-2.5
    python scripts/evaluate.py --table data/tile/carte_wina_pl --learner tabpfn-2.5
    python scripts/evaluate.py ... --protocol holdout --num_train 10000
    python scripts/evaluate.py ... --protocol strable

A table is a .parquet or .csv file, or the folder of a table of the TILE or
STRABLE benchmark, whose config.json gives the target and the task. Its
embeddings come first (scripts/extract_embeddings.py). The score of each split
is saved to `<results_dir>/<protocol>/<name>/<llm>_<learner>_<split>.csv`, and
a split already done is skipped. Under the holdout protocol, a table of no
more rows than `--num_train` is skipped, as in the paper.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tile.data import load_task
from tile.embeddings import Embeddings
from tile.evaluation import PROTOCOLS, evaluate_split, splits
from tile.learners import LEARNERS
from tile.llms import DEFAULT_LLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--table",
        required=True,
        help="A .parquet or .csv file, or the folder of a benchmark table.",
    )
    parser.add_argument(
        "--target", default=None, help="The column to predict (default: that of config.json)."
    )
    parser.add_argument(
        "--task",
        default=None,
        choices=["classification", "regression"],
        help="(default: that of config.json)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Name of the table (default: its file or folder name).",
    )
    parser.add_argument("--learner", default="tabpfn-2.5", choices=list(LEARNERS))
    parser.add_argument(
        "--llm",
        default=DEFAULT_LLM,
        help=f"The LLM of the embeddings (default: {DEFAULT_LLM}).",
    )
    parser.add_argument("--protocol", default="holdout", choices=PROTOCOLS)
    parser.add_argument(
        "--num_train",
        type=int,
        default=1024,
        help="Training rows of the holdout protocol (default: 1024).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embeddings_dir", default="embeddings")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    table, target, task, name = load_task(args.table, args.target, args.task)
    name = args.name or name
    if task is None:
        parser.error(f"{args.table} has no config.json: give its --task")
    if args.protocol == "holdout" and args.num_train >= len(table):
        print(f"{name}: {len(table)} rows, not more than {args.num_train}, skipping")
        raise SystemExit

    if args.device.startswith("cuda"):
        import torch

        # As in the runs of the paper: cuDNN's attention kernel fails on padded
        # batches in some environments
        torch.backends.cuda.enable_cudnn_sdp(False)

    X = table.drop(columns=target)
    y = table[target].to_numpy()
    if task == "classification":
        _, y = np.unique(y, return_inverse=True)
    embeddings = Embeddings.load(Path(args.embeddings_dir) / args.llm / f"{name}.npy")
    if len(embeddings) != len(table) or embeddings.columns != list(map(str, X.columns)):
        raise ValueError(f"The embeddings of {name} are not those of this table")

    results_dir = Path(args.results_dir) / args.protocol / name
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, train, test in splits(y, task, args.protocol, args.num_train):
        path = results_dir / f"{args.llm}_{args.learner}_{split}.csv"
        if not path.exists():
            print(f"{name}, {args.learner}, {split}")
            result = evaluate_split(
                X, y, train, test, embeddings, args.learner, task, args.device
            )
            row = {
                "table": name,
                "llm": args.llm,
                "learner": args.learner,
                "protocol": args.protocol,
                "split": split,
                **result,
            }
            pd.DataFrame([row]).to_csv(path, index=False)
        rows.append(pd.read_csv(path))

    results = pd.concat(rows)
    score = "r2" if task == "regression" else "roc_auc"
    print(results[["split", score, "fit_time", "predict_time"]].to_string(index=False))
    print(f"Mean {score}: {results[score].mean():.4f}")
