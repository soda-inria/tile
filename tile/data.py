"""Loading a table."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_table(path):
    """A table from a parquet or csv file, with the cleaning TILE expects.

    The text columns are of `object` dtype (pandas 3 reads them as `str`),
    their missing values NaN, and the line breaks and tabs within their cells
    spaces.
    """
    path = Path(path)
    table = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    for col in table.columns:
        if isinstance(table[col].dtype, pd.StringDtype):
            table[col] = table[col].astype(object)
    table = table.fillna(value=np.nan)
    return table.replace(["\n", "\t", "\r"], [" ", " ", ""], regex=True)


def load_task(path, target=None, task=None):
    """A table, its target, its task and its name.

    `path` is a .parquet or .csv file, named after its stem, or the folder of a
    table of the TILE or STRABLE benchmark, named after the folder: its
    `data.parquet`, with the target and the task read from its `config.json`.
    `target` and `task` ("classification" or "regression") override those of
    the config; a file has none, its task is then None.
    """
    path = Path(path)
    if not path.is_dir():
        if target is None:
            raise ValueError(f"{path} has no config.json: give its target")
        return load_table(path), target, task, path.stem
    with open(path / "config.json") as f:
        config = json.load(f)
    target = target or config["target_name"]
    # "regression", "b-classification" or "m-classification"
    task = task or (
        "regression" if config["task"] == "regression" else "classification"
    )
    return load_table(path / "data.parquet"), target, task, path.name
