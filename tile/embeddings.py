"""The cell embeddings of a table: each row serialized as one text, read by a
truncated LLM, and the hidden states of each cell mean-pooled."""

import json
import os
import socket
import time
from pathlib import Path

import numpy as np


def serialize(table):
    """Each row as one text, and the characters of each of its cells.

    A row reads "The <column> is <value>. The <column> is <value>. ...", one
    sentence per cell in the order of the table, with the underscores of the
    column names written as spaces, and the missing values and blank strings
    as "NaN".

    Returns:
        texts: The serialized rows.
        spans: (n_rows, n_columns, 2) array, the [start, end) characters of
            each cell sentence in its row.
    """
    names = [str(col).replace("\n", "").replace("_", " ") for col in table.columns]
    values = table.astype(str)
    missing = table.isna() | values.apply(lambda col: col.str.strip() == "")
    values = values.mask(missing, "NaN").to_numpy()

    texts = []
    spans = np.empty((len(values), len(names), 2), dtype=np.int64)
    for i, row in enumerate(values):
        cells, start = [], 0
        for j, (name, value) in enumerate(zip(names, row)):
            cell = f"The {name} is {value}."
            # A value ending in a space ends in "<value>."
            if cell.endswith(" ."):
                cell = cell[:-2] + "."
            spans[i, j] = (start, start + len(cell))
            cells.append(cell)
            start += len(cell) + 1
        texts.append(" ".join(cells))
    return texts, spans


def pool_cells(model, tokenizer, texts, spans, max_tokens=65536, layers=None, out=None):
    """Mean-pools the last hidden state of the model over the tokens of each cell.

    The spans are read off the tokenization of the whole row: a token belongs
    to every cell whose characters it overlaps, so a token that straddles two
    cells is pooled into both. Special and padding tokens cover no character
    and belong to no cell. The rows run from the longest to the shortest, in
    batches of at most `max_tokens` tokens, padding included (a row longer than
    that goes alone).

    With `layers`, the hidden states of these layers are pooled instead, layer
    k the output of the k-th block (0 that of the embeddings), for the layer
    probing: into `out` if given, e.g. a memory map.

    Returns:
        A (n_columns, n_rows, hidden_size) float16 array, or with `layers` a
        (n_layers, n_columns, n_rows, hidden_size) one.
    """
    import torch
    from tqdm import tqdm

    n_rows, n_cols, _ = spans.shape
    # Right padding: under causal attention, no real token then sees the
    # padding, and the positions of the real tokens start at 0 in every row
    tokenizer.padding_side = "right"
    no_bos = len(tokenizer("")["input_ids"]) == 0

    # Token count of each row, in chunks to bound the memory of the encodings
    lengths = np.concatenate(
        [
            [len(ids) for ids in tokenizer(texts[i : i + 10_000])["input_ids"]]
            for i in range(0, n_rows, 10_000)
        ]
    )
    order = np.argsort(-lengths, kind="stable")

    shape = (n_cols, n_rows, model.config.hidden_size)
    if layers is None:
        embeddings = np.empty(shape, np.float16)
        targets = [embeddings]
    else:
        embeddings = np.empty((len(layers), *shape), np.float16) if out is None else out
        targets = list(embeddings)
    start = 0
    with tqdm(total=n_rows, desc="Pooling cells", mininterval=60) as progress:
        while start < n_rows:
            # The first row of the batch is its longest
            batch_size = max(1, max_tokens // lengths[order[start]])
            idx = order[start : start + batch_size]

            inputs = tokenizer(
                [texts[k] for k in idx],
                return_tensors="pt",
                padding=True,
                return_offsets_mapping=True,
            )
            offsets = inputs.pop("offset_mapping")

            with torch.inference_mode():
                outputs = model(
                    **inputs.to(model.device), output_hidden_states=layers is not None
                )
                states = (
                    [outputs.last_hidden_state]
                    if layers is None
                    else [outputs.hidden_states[layer] for layer in layers]
                )

                # (batch, n_cols, seq_len): the token overlaps the cell
                device = states[0].device
                offsets = offsets.to(device)
                cells = torch.as_tensor(spans[idx], device=device)
                tok_start = offsets[:, None, :, 0]
                tok_end = offsets[:, None, :, 1]
                in_cell = (
                    (tok_start < cells[:, :, 1:])
                    & (tok_end > cells[:, :, :1])
                    & (tok_end > tok_start)
                ).float()
                if no_bos:
                    # The sink, the first token of the row
                    in_cell[:, :, 0] = 0
                for target, hidden in zip(targets, states):
                    # The layers of a model spread over GPUs are on different ones
                    weights = in_cell.to(hidden.device)
                    pooled = torch.bmm(weights, hidden.float()) / weights.sum(
                        -1, keepdim=True
                    )
                    target[:, idx] = pooled.transpose(0, 1).half().cpu().numpy()
            start += len(idx)
            progress.update(len(idx))

    return embeddings


def embed_table(table, model, tokenizer, max_tokens=65536):
    """The cell embeddings of every column of `table`, and how they were made.

    `table` is the table without its target. The extraction time covers the
    serialization, the tokenization and the forward passes; the peak GPU
    memory is the one PyTorch reserved.
    """
    import torch

    on_gpu = torch.cuda.is_available()
    if on_gpu:
        # cuDNN's attention kernel fails on padded batches in some
        # environments (it compiles its kernels with NVRTC): left out
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    texts, spans = serialize(table)
    embeddings = pool_cells(model, tokenizer, texts, spans, max_tokens)
    stats = {
        "time_s": round(time.perf_counter() - start, 4),
        "max_tokens": max_tokens,
        "node": socket.gethostname(),
    }
    if on_gpu:
        stats["gpu"] = torch.cuda.get_device_name(0)
        stats["peak_gpu_memory_mb"] = round(
            torch.cuda.max_memory_reserved() / 1024**2, 1
        )
    return embeddings, stats


def save(path, embeddings, columns, **info):
    """Saves the embeddings to `path` (.npy), and their columns and `info` next
    to it (.json)."""
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump({"columns": [str(col) for col in columns], **info}, f, indent=2)
    # Written last, and renamed into place: the .npy marks the table as done
    with open(f"{path}.tmp", "wb") as f:
        np.save(f, embeddings)
    os.replace(f"{path}.tmp", path)


class Embeddings:
    """The cell embeddings of a table, one (n_rows, hidden_size) block per column.

    Read from disk one column at a time, when the column is asked for:
    `embeddings[column]`. `take(rows)` restricts them to some rows of the
    table, as positions.
    """

    def __init__(self, array, columns, info=None, rows=None):
        self.array = array
        self.columns = [str(col) for col in columns]
        self.info = info or {}
        self.rows = rows
        self._index = {col: i for i, col in enumerate(self.columns)}

    @classmethod
    def load(cls, path):
        path = Path(path)
        with open(path.with_suffix(".json")) as f:
            info = json.load(f)
        columns = info.pop("columns")
        return cls(np.load(path, mmap_mode="r"), columns, info)

    def __len__(self):
        return self.array.shape[1] if self.rows is None else len(self.rows)

    def __getitem__(self, column):
        block = self.array[self._index[str(column)]]
        return np.asarray(block if self.rows is None else block[self.rows])

    def take(self, rows):
        rows = np.asarray(rows)
        return Embeddings(
            self.array,
            self.columns,
            self.info,
            rows if self.rows is None else self.rows[rows],
        )
