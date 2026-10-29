# TILE: Table-Interfaced LLM Embeddings

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/inria-soda/TILE-benchmark)
[![arXiv](https://img.shields.io/badge/arXiv-2610.17065-blue.svg)](https://arxiv.org/pdf/2610.17065)

Paper: [TILE: Bridging LLMs and Table Foundation Models for Tabular Learning
with Strings](https://arxiv.org/pdf/2610.17065)

TILE brings the string columns of a table to a tabular learner, such as a
table foundation model like TabPFN, through the embeddings of a truncated
LLM. This repository runs TILE on a table: it downloads and truncates the
LLMs, extracts the cell embeddings, and evaluates TILE with a tabular learner.

## How it works

1. **Serialization.** Each row becomes one text, one sentence per cell:
   `The <column> is <value>. The <column> is <value>. ...`. Every column except
   the target is included, numbers and dates too.
2. **Cell embeddings.** The text goes through the first layers of an LLM: by
   default, the first 14 layers of Llama-3.1-8B. The hidden states of the
   tokens of each cell are mean-pooled. This gives one embedding per cell,
   which has seen the cells before it in the row.
3. **Reduction.** The embeddings of each string column are standardized and
   reduced by their own PCA, to 30 components. The number of components is
   then rescaled so that the whole input of the learner falls inside its
   *feature window*: the range of input widths the learner handles best.
4. **Tabular learner.** The learner receives the reduced embeddings, along
   with:
   - the numerical and date columns, through skrub's `TableVectorizer`;
   - a target encoding of the string columns, computed out of fold;
   - for the learners that take categories, the string columns with fewer than
     50 distinct values.

## Installation

```bash
pip install -e ".[llm,tabpfn,tabicl]"
```

The `llm` extra installs what the extraction of the embeddings needs. Each
learner has its own extra, pinned to the version used in the paper. Some
learners conflict with each other, so they need a separate environment.
The foundation models download their checkpoints on first use. The TabPFN
checkpoints require you to accept their license, for example through the
`TABPFN_TOKEN` environment variable (see the TabPFN documentation).

## 1. Download and truncate the LLMs

The Llama models are gated: accept their license on the Hugging Face hub, then
log in with `hf auth login`.

```bash
python scripts/download_llms.py                   # Llama-3.1-8B, the LLM of TILE
python scripts/download_llms.py --llms all        # the 7 LLMs of the paper
```

Each LLM is cut at one of the layers whose cell embeddings predicted best in
the layer probing of the paper, and saved to `models/<llm>`. The scores of the
top layers are close: for some LLMs, the layer kept is not the very best of
the final probing runs, but one of its top three.

| LLM | Name | Layers kept |
|---|---|---|
| Llama-3.1-8B (default) | `llama-3.1-8b` | 14 of 32 |
| Llama-3.1-70B | `llama-3.1-70b` | 32 of 80 |
| Llama-3.2-1B | `llama-3.2-1b` | 10 of 16 |
| Llama-3.2-3B | `llama-3.2-3b` | 11 of 28 |
| Qwen3-0.6B-Base | `qwen3-0.6b-base` | 16 of 28 |
| Qwen3-4B-Base | `qwen3-4b-base` | 20 of 36 |
| Qwen3-8B-Base | `qwen3-8b-base` | 19 of 36 |

Only the weights of the kept layers are loaded. The loading report lists the
other layers as `UNEXPECTED`; this is expected. To run TILE with one of these
LLMs, see [Using another LLM](#using-another-llm); to add an LLM of your own,
see [Adding an LLM](#adding-an-llm).

## 2. Extract the embeddings

A table is a `.parquet` or `.csv` file:

```bash
python scripts/extract_embeddings.py --table data/my_table.parquet --target price
```

A table can also be the folder of a table of the TILE or STRABLE benchmark:
its `data.parquet`, with a `config.json` that gives the target and the task.
Then `--target` (and `--task` below) can be left out, and the table is named
after its folder.

The embeddings are saved to `embeddings/<llm>/my_table.npy`. This is a float16
array of shape (columns, rows, hidden size). Next to it, a `.json` file holds
the columns and the extraction statistics: time, peak GPU memory and GPU.

Rows are batched by a budget of 65,536 tokens (`--max_tokens`). Lower it if
the GPU runs out of memory. Truncated to 32 layers, Llama-3.1-70B needs about
56 GB of GPU memory for its weights. It is spread over the visible GPUs.

## 3. Evaluate TILE

```bash
python scripts/evaluate.py --table data/my_table.parquet --target price \
    --task regression --learner tabpfn-2.5
```

There are two protocols (`--protocol`):

- `holdout` (default), the protocol of the paper's benchmark. The learner is
  trained on `--num_train` rows drawn at random (default 1024) and tested on
  up to 1024 other rows, for 5 seeds. The draw is stratified for a
  classification. A table with no more rows than `--num_train` is skipped.
- `strable`, the protocol of the STRABLE benchmark: a 3-fold cross-validation
  over every row, stratified for a classification.

The scores are R² and RMSE for a regression, and ROC AUC and accuracy for a
classification. For more than two classes, the ROC AUC is one-vs-rest,
macro-averaged.

Each split is saved to
`results/<protocol>/my_table/<llm>_<learner>_<split>.csv`, and a split already
done is skipped. Besides the scores, a result holds:

- the width of the learner's input;
- the fit and prediction times;
- the share of the extraction time that falls on the rows used;
- the peak GPU memory of the learner.

### Using another LLM

TILE runs with Llama-3.1-8B by default. To run it with another LLM, download
it, then pass its name (see the table of step 1) with `--llm`, to both the
extraction and the evaluation:

```bash
python scripts/download_llms.py --llms qwen3-8b-base
python scripts/extract_embeddings.py --table data/my_table.parquet --target price \
    --llm qwen3-8b-base
python scripts/evaluate.py --table data/my_table.parquet --target price \
    --task regression --learner tabpfn-2.5 --llm qwen3-8b-base
```

The embeddings of each LLM are kept apart, in `embeddings/<llm>/`, and the
results are named after it, so the LLMs can be compared on the same tables. In
the Python API, load the embeddings of that LLM:
`Embeddings.load("embeddings/qwen3-8b-base/my_table.npy")`. An LLM added with
`scripts/add_llm.py` is used the same way, under the name it was given.

### Learners and their feature windows

The window of a learner was selected on the 27 tables of the paper's probing
(see [Adding a tabular learner](#adding-a-tabular-learner)). The learners
without a window keep 30 components per string column, whatever the width of
the table.

| Learner | Name | Feature window | Categories |
|---|---|---|---|
| TabPFN-2.5 | `tabpfn-2.5` | 200 to 400 | ✓ |
| TabPFN-3 | `tabpfn-3` | 150 to 400 | ✓ |
| TabPFN-3.5 | `tabpfn-3.5` | 200 to 800 | ✓ |
| TabICLv2 | `tabicl` | 200 to 800 | ✓ |
| EXAONE-Tabular | `exaone` | 150 to 400 | ✓ |
| Causilo | `causilo` | 200 to 800 | ✓ |
| KumoTabular (small, medium, large) | `kumo-small`, `kumo-medium`, `kumo-large` | 200 to 600 | ✓ |
| RealMLP | `realmlp` | none | |
| XGBoost | `xgboost` | none | |
| CatBoost | `catboost` | none | |
| Ridge | `ridge` | none | |

XGBoost is tuned: the best of its defaults and 99 random configurations, by a
5-fold cross-validation repeated 5 times. RealMLP runs its own search. The
other learners keep their defaults.

## Reproducing the TILE benchmark results

The paper's benchmark gathers 100 tables with strings, from CARTE,
TextTabBench and WikiDBs, released as the
[TILE benchmark](https://huggingface.co/datasets/inria-soda/TILE-benchmark).

**1. Download the tables** (206 MB). Each table is a folder holding
`data.parquet` and a `config.json` with its target and task:

```bash
hf download inria-soda/TILE-benchmark --repo-type dataset --local-dir data/tile
```

**2. Extract the embeddings and evaluate TILE** on every table, with the
holdout protocol, on 1,024, 10,000 and 50,000 training rows as in the paper.
The tables too small for a training size are skipped:

```bash
LLM=llama-3.1-8b
LEARNER=tabpfn-2.5
for dir in data/tile/*/; do
    python scripts/extract_embeddings.py --table "$dir" --llm "$LLM"
    for n in 1024 10000 50000; do
        python scripts/evaluate.py --table "$dir" --llm "$LLM" --learner "$LEARNER" \
            --num_train $n
    done
done
```

The embeddings are extracted once per table and LLM, and reused by every
learner. A split already evaluated is skipped, so the loop can be run again
after an interruption. Each learner runs from its own environment (see
Installation). The results of a table are saved to `results/holdout/<table>/`,
one file per seed and training size. To gather them:

```python
import pandas as pd
from glob import glob

results = pd.concat(map(pd.read_csv, glob("results/holdout/*/*.csv")))
print(results.groupby(["llm", "learner", "n_train"])[["r2", "roc_auc"]].mean())
```

## Reproducing the STRABLE results

The paper evaluates TILE on the 108 tables of the
[STRABLE benchmark](https://huggingface.co/datasets/inria-soda/STRABLE-benchmark),
with the STRABLE protocol.

**1. Download the tables** (210 MB), in the same layout as the TILE benchmark:

```bash
hf download inria-soda/STRABLE-benchmark --repo-type dataset \
    --include "*/data.parquet" "*/config.json" --local-dir data/strable
```

**2. Extract the embeddings and evaluate TILE** on every table:

```bash
LLM=llama-3.1-8b
LEARNER=tabpfn-2.5
for dir in data/strable/*/; do
    python scripts/extract_embeddings.py --table "$dir" --llm "$LLM"
    python scripts/evaluate.py --table "$dir" --llm "$LLM" --learner "$LEARNER" \
        --protocol strable
done
```

The results of a table are saved to `results/strable/<table>/`, one file per
fold.

**Results.** Raw scores of the paper (×100), by tabular learner:

- **Classification:** the ROC AUC, averaged over the 32 classification tables.
- **Regression:** the R² clipped at 0, averaged over the 76 regression tables.

Each table is scored by its average over the 3 folds.

- **TF-IDF:** the TF-IDF baseline of STRABLE with the same learner.
- **Best other encoder:** the best of the other string encoders of the
  [STRABLE leaderboard](https://huggingface.co/datasets/inria-soda/STRABLE-benchmark)
  with that learner, chosen separately for each task.
- **"no PCA":** the embeddings reach the learner unreduced.

| Learner | TILE (Llama-3.1-8B) | TILE (Llama-3.1-70B) | TF-IDF | Best other encoder |
|---|---|---|---|---|
| **Classification ROC AUC** | | | | |
| Causilo | 87.19 | **87.63** | 86.36 | 86.34 (Qwen3-Embedding-8B, no PCA) |
| TabPFN-3.5 | 87.27 | **87.66** | 86.75 | 86.56 (Qwen3-Embedding-8B, no PCA) |
| TabPFN-2.5 | **86.59** | — | 85.93 | 85.83 (Llama-3.1-8B, PCA to 30) |
| TabICLv2 | 86.62 | **87.05** | 85.38 | 85.06 (Qwen3-Embedding-8B, no PCA) |
| TabPFN-3 | 86.18 | **86.75** | 85.19 | 84.75 (Qwen3-Embedding-8B, no PCA) |
| **Regression R²** | | | | |
| Causilo | 73.41 | **74.23** | 70.66 | 71.52 (Qwen3-Embedding-8B, no PCA) |
| TabPFN-3.5 | 72.67 | **73.63** | 70.99 | 71.41 (Qwen3-Embedding-8B, no PCA) |
| TabPFN-2.5 | **72.71** | — | 69.65 | 70.03 (Qwen3-Embedding-8B, no PCA) |
| TabICLv2 | 71.41 | **72.40** | 68.33 | 68.59 (Qwen3-Embedding-8B, no PCA) |
| TabPFN-3 | 71.80 | **72.67** | 68.66 | 68.91 (Qwen3-Embedding-8B, no PCA) |

— : TILE (Llama-3.1-70B) with TabPFN-2.5 has not run on every table.

The standard errors due to the cross-validation splits alone are between 0.09
and 0.17. They hold the tables fixed, so they leave out the spread of scores
across tables.

The paper's runs used one A100 or A40 GPU and 150 GB of RAM per table. The
tables have from 550 to 75,000 rows, and each fold trains on two thirds of
them. A learner that runs out of GPU memory through
fragmentation can be run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
the paper's EXAONE runs always used it.

## Adding an LLM

`scripts/add_llm.py` makes any decoder LLM of the Hugging Face hub an LLM of
TILE, with the selection of the paper. Every layer is probed on the 27
probing tables of the TILE benchmark, downloaded to `data/tile` (see above):

- only the string columns are serialized, and the cell embeddings of a row,
  side by side and without any reduction, feed a Ridge;
- the Ridge is trained on 1,024 rows and tested on up to 1,024 others, for 5
  seeds;
- the score of a table is its R², clipped at 0, or its ROC AUC, averaged over
  the seeds.

The LLM is then cut at the layer of the best mean score over the tables:

```bash
python scripts/add_llm.py --name mistral-7b --repo_id mistralai/Mistral-7B-v0.3
python scripts/extract_embeddings.py --table data/tile/carte_wina_pl --llm mistral-7b
```

The truncated LLM is saved to `models/mistral-7b`, with the mean score of every
layer in `probing.json`. `--llm mistral-7b` then runs TILE with it. The
script prints the line that adds it to `LLMS` in `tile/llms.py`, for
`download_llms.py`.

The score of each table is saved to `results/probing/<name>/`, and a table
already probed is skipped. The full LLM is loaded, spread over the visible
GPUs, and the hidden states of every layer are held on the GPU for a batch of
rows: lower `--max_tokens` (default 8,192) if it runs out of memory. The cell
embeddings of every layer of a table are held at once: about 21 GB for
Llama-3.1-8B on the largest table, which `--tmp_dir` puts on disk instead of
in memory. `--step 2` probes every other layer, to halve that.

## Adding a table foundation model

A new table foundation model, takes two steps.

**1. Add it to `tile/learners.py`**, without a window: an entry in `LEARNERS`,
such as `"my-tfm": Learner(native_categories=True)`, and a branch in
`make_learner` that builds it. Its input is an array whose first `n_base`
columns are numbers; with `native_categories`, the columns after them are the
string columns of fewer than 50 distinct values, as strings, missing values as
`"nan"`.

**2. Find its feature window:**

```bash
python scripts/feature_window.py --learner my-tfm
```

TILE runs on the 27 probing tables with the whole input of the learner pinned
to each of 50, 100, 150, 200, 300, 400, 600, 800 and 1,200 features, for 5
seeds of 1,024 training rows. The embeddings of Llama-3.1-8B (`--llm`) that are
missing are extracted first.

For each table, the error at a budget (1 - score) is divided by the error at
the best budget. The window is the run of budgets around the best one whose
mean ratio over the tables, minus one standard error, is within 1% of the best.

The script prints the window and the line to put in `LEARNERS`, such as
`"my-tfm": Learner((200, 600), True)`. If the window reaches the smallest or
the largest budget, its bound may lie beyond: run more with `--budgets`. The
score of each split is saved to `results/feature_window/<llm>/<learner>/`, and
a split already done is skipped.

## Python API

```python
from tile import TILE, Embeddings
from tile.data import load_table

table = load_table("data/my_table.parquet")
embeddings = Embeddings.load("embeddings/llama-3.1-8b/my_table.npy")
X, y = table.drop(columns="price"), table["price"]
train, test = range(0, 1000), range(1000, 1500)

model = TILE(learner="tabpfn-2.5", task="regression")
model.fit(X.iloc[train], y.iloc[train], embeddings.take(train))
prediction = model.predict(X.iloc[test], embeddings.take(test))
```

`Embeddings.take` restricts the embeddings to some rows of the table, given
by position. The embeddings of each column are only read from disk when they
are needed.

## Reproducibility

The LLMs run in bfloat16, and several learners run on GPU: repeated runs, and
runs on other hardware, can differ slightly.

## Citation

If you use TILE, please cite the paper:

```bibtex
@article{lefebvre2026tile,
  title={TILE: Bridging LLMs and Table Foundation Models for Tabular Learning with Strings},
  author={Lefebvre, F{\'e}lix and Holzm{\"u}ller, David and Blayer, Gioia and Kim, Myung Jun and Morvan, Marine Le and Varoquaux, Ga{\"e}l},
  journal={arXiv preprint arXiv:2610.17065},
  year={2026}
}
```
