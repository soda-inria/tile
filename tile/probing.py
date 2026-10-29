"""The two selections of the paper, made on its 27 probing tables: the layer an
LLM is cut at, and the feature window of a tabular learner.

Layer probing. Every layer of the LLM embeds the string columns of a table,
serialized alone. The cell embeddings of a row, side by side and without any
reduction, feed a Ridge, trained on 1,024 rows for the 5 seeds of the holdout
protocol. The LLM is cut at the layer of the best mean score over the tables.

Feature window. TILE runs with the whole input of the learner pinned to each
of `BUDGETS` features in turn (a window of a single width), on the embeddings
of an LLM, on 1,024 training rows for the 5 seeds of the holdout protocol. The
error of a table, 1 - score, is divided by its error at the best budget; the
ratios are averaged over the tables geometrically. The window is the run of
budgets around the best one whose error could be within `TOLERANCE` of it: the
mean ratio, minus one standard error across the tables, on the log scale, is
no more than `TOLERANCE` above the best.

The score of a table is R² clipped at 0, or ROC AUC, averaged over the seeds.
"""

import numpy as np
import pandas as pd

from tile.embeddings import pool_cells, serialize
from tile.evaluation import scores, splits
from tile.features import string_columns
from tile.learners import make_learner, predict, predict_proba

PROBING_TABLES = [
    "carte_beer_ratings",
    "carte_chocolate_bar_ratings",
    "carte_coffee_ratings",
    "carte_company_employees",
    "carte_journal_jcr",
    "carte_michelin",
    "carte_movies",
    "carte_nba_draft",
    "carte_prescription_drugs",
    "carte_ramen_ratings",
    "carte_roger_ebert",
    "carte_used_cars_24",
    "carte_used_cars_dot_com",
    "carte_videogame_sales",
    "carte_wine_vivino_price",
    "carte_yelp",
    "carte_zomato",
    "ttb_beer",
    "ttb_mercari",
    "wdbs_14976_DrawingsCatalog",
    "wdbs_28324_ukrainian_village_instances",
    "wdbs_36100_MagicNarrativeMotifs",
    "wdbs_53353_research_articles",
    "wdbs_56474_Sculpture_Instances",
    "wdbs_66610_geopolitical_regions",
    "wdbs_82939_Territorial_Entities",
    "wdbs_89039_Business_Entity_Locations",
]

# The widths of the input of the learner of the budget sweep
BUDGETS = (50, 100, 150, 200, 300, 400, 600, 800, 1200)
# At most 1% more error than at the best budget, on average over the tables
TOLERANCE = 0.01
# Floor of the error, so that a near-perfect score does not blow up its ratio
MIN_ERROR = 1e-3


def score(result):
    """R² clipped at 0, or ROC AUC, of the scores of a split."""
    r2 = result.get("r2", np.nan)
    return result["roc_auc"] if pd.isna(r2) else max(r2, 0)


def probe_layers(X, y, task, model, tokenizer, layers, max_tokens=8192, out_dir=None):
    """The score of a Ridge on the cell embeddings of each layer, on each
    split of the holdout protocol (1,024 training rows, 5 seeds).

    `X` is the table without its target, `y` the target (class indices for a
    classification). Only the string columns are serialized, and only the rows
    of the splits embedded. Every layer is held at once, on disk in `out_dir`
    if given (a memory map), else in memory.

    Returns:
        A DataFrame of the layer, the split and its score.
    """
    import tempfile

    X = X[string_columns(X)]
    folds = splits(y, task, "holdout", 1024)
    rows = np.unique(np.concatenate([np.concatenate(fold[1:]) for fold in folds]))
    # The position of a row of the table among the rows embedded
    position = np.empty(len(y), dtype=int)
    position[rows] = np.arange(len(rows))

    texts, spans = serialize(X.iloc[rows])
    shape = (len(layers), X.shape[1], len(rows), model.config.hidden_size)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        out = None
        if out_dir is not None:
            out = np.lib.format.open_memmap(
                f"{tmp}/layers.npy", mode="w+", dtype=np.float16, shape=shape
            )
        embeddings = pool_cells(
            model, tokenizer, texts, spans, max_tokens, layers=layers, out=out
        )

        results = []
        for layer, cells in zip(layers, embeddings):
            # The cell embeddings of a row, side by side
            Z = np.hstack(list(cells)).astype(np.float32)
            for split, train, test in folds:
                ridge = make_learner("ridge", task).fit(Z[position[train]], y[train])
                Z_test = Z[position[test]]
                prediction = (
                    predict(ridge, Z_test)
                    if task == "regression"
                    else predict_proba(ridge, Z_test)
                )
                result = scores(y[test], prediction, task)
                results.append({"layer": layer, "split": split, "score": score(result)})
    return pd.DataFrame(results)


def best_layer(results):
    """The layer of the best mean score over the tables, and the mean score of
    every layer.

    `results` holds the table, the layer, the split and the score; the layers
    are compared on the tables they all ran on.
    """
    per_table = results.groupby(["table", "layer"])["score"].mean().unstack()
    mean = per_table.dropna().mean()
    return mean.idxmax(), mean


def feature_window(scores):
    """The feature window of a learner, from its budget sweep.

    `scores` is a (table, budget) DataFrame of seed-averaged scores, with the
    budgets as sorted columns, run on every table. Returns the best budget,
    the window, and the mean and standard error, across tables, of the log of
    the error ratio to the best budget.
    """
    log_error = np.log((1 - scores).clip(lower=MIN_ERROR))
    best = log_error.mean().idxmin()
    log_ratio = log_error.sub(log_error[best], axis=0)
    mean, se = log_ratio.mean(), log_ratio.std() / np.sqrt(len(log_ratio))

    # Grow the window from the best budget while the next one stays within the
    # tolerance, so that a noisy budget far from the best cannot widen it
    within = mean - se <= np.log1p(TOLERANCE)
    budgets = list(scores.columns)
    lo = hi = budgets.index(best)
    while lo > 0 and within[budgets[lo - 1]]:
        lo -= 1
    while hi < len(budgets) - 1 and within[budgets[hi + 1]]:
        hi += 1
    return best, (budgets[lo], budgets[hi]), mean, se
