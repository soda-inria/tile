"""Extract the cell embeddings of a table with a truncated LLM.

    python scripts/extract_embeddings.py --table data/my_table.parquet --target price
    python scripts/extract_embeddings.py --table data/tile/carte_wina_pl

Every column but the target is embedded. A table is a .parquet or .csv file,
or the folder of a table of the TILE or STRABLE benchmark, whose config.json
gives the target. The embeddings are saved to `<embeddings_dir>/<llm>/<name>.npy`,
a float16 array of shape (n_columns, n_rows, hidden size), next to a .json
with the columns and the extraction stats (time, peak GPU memory).
"""

import argparse
from pathlib import Path

from tile.data import load_task
from tile.embeddings import embed_table, save
from tile.llms import DEFAULT_LLM, load

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
        "--name",
        default=None,
        help="Name of the table (default: its file or folder name).",
    )
    parser.add_argument(
        "--llm",
        default=DEFAULT_LLM,
        help=f"A folder of <models_dir> (default: {DEFAULT_LLM}).",
    )
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--embeddings_dir", default="embeddings")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=65536,
        help="Token budget of a batch of rows, padding included.",
    )
    args = parser.parse_args()

    table, target, _, name = load_task(args.table, args.target)
    name = args.name or name
    path = Path(args.embeddings_dir) / args.llm / f"{name}.npy"
    if path.exists():
        print(f"{path} already exists, skipping")
        raise SystemExit

    table = table.drop(columns=target)
    model, tokenizer = load(Path(args.models_dir) / args.llm)
    embeddings, stats = embed_table(table, model, tokenizer, args.max_tokens)
    print(stats)
    save(path, embeddings, table.columns, llm=args.llm, **stats)
