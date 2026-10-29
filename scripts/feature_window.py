"""The feature window of a tabular learner, from a budget sweep.

    python scripts/feature_window.py --learner my-tfm

TILE runs on the 27 probing tables of the TILE benchmark, which `--data_dir`
holds, with the whole input of the learner pinned to each budget in turn (see
`tile.probing`). The window is read off the scores, and printed with the line
that gives it to the learner in `LEARNERS`.

The learner must be in `LEARNERS` (tile/learners.py) and `make_learner`: add
it there first, without a window. The embeddings of the tables missing from
`<embeddings_dir>/<llm>` are extracted first. The score of each split is saved
to `<results_dir>/feature_window/<llm>/<learner>/<table>_<budget>_<split>.csv`,
and a split already done is skipped.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tile.data import load_task
from tile.embeddings import Embeddings, embed_table, save
from tile.evaluation import evaluate_split, splits
from tile.learners import LEARNERS
from tile.llms import DEFAULT_LLM, load
from tile.probing import BUDGETS, PROBING_TABLES, feature_window, score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--learner", required=True, choices=list(LEARNERS))
    parser.add_argument(
        "--llm",
        default=DEFAULT_LLM,
        help=f"A folder of <models_dir> (default: {DEFAULT_LLM}).",
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=list(BUDGETS)
    )
    parser.add_argument(
        "--data_dir", default="data/tile", help="The TILE benchmark (default: data/tile)."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--embeddings_dir", default="embeddings")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        import torch

        # cuDNN's attention kernel fails on padded batches in some environments
        torch.backends.cuda.enable_cudnn_sdp(False)

    tables = {name: load_task(Path(args.data_dir) / name) for name in PROBING_TABLES}

    # The embeddings missing, with the LLM loaded once
    llm = None
    for name, (table, target, _, _) in tables.items():
        path = Path(args.embeddings_dir) / args.llm / f"{name}.npy"
        if not path.exists():
            if llm is None:
                llm = load(Path(args.models_dir) / args.llm)
            print(f"Extracting the embeddings of {name}")
            X = table.drop(columns=target)
            embeddings, stats = embed_table(X, *llm)
            save(path, embeddings, X.columns, llm=args.llm, **stats)
    del llm

    results_dir = Path(args.results_dir) / "feature_window" / args.llm / args.learner
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (table, target, task, _) in tables.items():
        X = table.drop(columns=target)
        y = table[target].to_numpy()
        if task == "classification":
            _, y = np.unique(y, return_inverse=True)
        embeddings = Embeddings.load(Path(args.embeddings_dir) / args.llm / f"{name}.npy")
        for budget in sorted(args.budgets):
            for split, train, test in splits(y, task, "holdout", 1024):
                path = results_dir / f"{name}_{budget}_{split}.csv"
                if not path.exists():
                    print(f"{name}, {args.learner}, budget {budget}, {split}")
                    result = evaluate_split(
                        X,
                        y,
                        train,
                        test,
                        embeddings,
                        args.learner,
                        task,
                        args.device,
                        window=(budget, budget),
                    )
                    row = {"table": name, "budget": budget, "split": split, **result}
                    pd.DataFrame([row]).to_csv(path, index=False)
                rows.append(pd.read_csv(path))

    results = pd.concat(rows)
    results["score"] = [score(row) for row in results.to_dict("records")]
    scores = results.groupby(["table", "budget"])["score"].mean().unstack()
    best, window, mean, se = feature_window(scores)

    print(f"{args.learner}, {len(scores)} tables: best budget {best}")
    print("  budget             " + "".join(f"{b:>7d}" for b in mean.index))
    print("  error vs best (%)  " + "".join(f"{100 * np.expm1(m):>+7.1f}" for m in mean))
    print("  s.e. (log, x100)   " + "".join(f"{100 * s:>7.1f}" for s in se))
    # A bound on the edge of the budgets run may lie beyond it
    for bound, edge in [(window[0], mean.index[0]), (window[1], mean.index[-1])]:
        if bound == edge:
            print(
                f"  The window reaches the edge of the budgets ({edge}): its bound "
                "may lie beyond, run more budgets with --budgets"
            )
    native = LEARNERS[args.learner].native_categories
    print(
        f"The feature window of {args.learner}: {window[0]} to {window[1]}. "
        "In `LEARNERS` (tile/learners.py):\n"
        f'    "{args.learner}": Learner({window}, {native}),'
    )
