"""The two evaluation protocols of the paper.

`holdout`, the protocol of its benchmark: a training set of `num_train` rows
drawn at random, tested on up to 1024 of the other rows, for 5 seeds.
`strable`, the protocol of the STRABLE benchmark: a 3-fold cross-validation
over every row.
"""

import time

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from tile.model import TILE

PROTOCOLS = ("holdout", "strable")


def splits(y, task, protocol="holdout", num_train=1024):
    """The (name, train rows, test rows) of each split of the protocol."""
    classification = task == "classification"
    if protocol == "holdout":
        if num_train >= len(y):
            raise ValueError(f"{num_train} training rows out of {len(y)}")
        num_test = min(1024, len(y) - num_train)
        return [
            (
                f"n{num_train}_seed{seed}",
                *train_test_split(
                    np.arange(len(y)),
                    train_size=num_train,
                    test_size=num_test,
                    random_state=seed,
                    stratify=y if classification else None,
                ),
            )
            for seed in range(1, 6)
        ]
    folds = (StratifiedKFold if classification else KFold)(
        n_splits=3, shuffle=True, random_state=1234
    )
    return [
        (f"fold{i}", train, test)
        for i, (train, test) in enumerate(folds.split(np.zeros(len(y)), y))
    ]


def scores(y_true, prediction, task):
    """R2 and RMSE of a regression, ROC AUC and accuracy of a classification.

    `y_true` holds the class indices of the columns of the probabilities.
    """
    if task == "regression":
        return {
            "r2": r2_score(y_true, prediction),
            "rmse": root_mean_squared_error(y_true, prediction),
        }
    if prediction.shape[1] == 2:
        auc = roc_auc_score(y_true, prediction[:, 1])
    else:
        # One-vs-rest, macro-averaged, on the softmax of the probabilities:
        # the scores of the paper
        auc = roc_auc_score(
            y_true,
            softmax(prediction, axis=1),
            multi_class="ovr",
            labels=np.arange(prediction.shape[1]),
        )
    return {"roc_auc": auc, "accuracy": accuracy_score(y_true, prediction.argmax(1))}


def _reset_gpu_peak(device):
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _gpu_peak(device):
    if not str(device).startswith("cuda"):
        return {}
    import torch

    return {
        "gpu": torch.cuda.get_device_name(),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
    }


def evaluate_split(
    X, y, train, test, embeddings, learner, task, device="cuda", window=None
):
    """Fit TILE on the `train` rows, and score it on the `test` rows.

    `X` is the table without its target, `y` the target (class indices for a
    classification), and `embeddings` the cell embeddings of the table.
    `window` replaces the feature window of the learner.
    Besides the scores, the result holds the times of the fit and of the
    predictions, the share of the extraction time of the embeddings that
    falls on the rows used, and the peak GPU memory of the learner.
    """
    tile = TILE(learner, task, device, window)
    _reset_gpu_peak(device)

    start = time.perf_counter()
    tile.fit(X.iloc[train], y[train], embeddings.take(train))
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    if task == "regression":
        prediction = tile.predict(X.iloc[test], embeddings.take(test))
    else:
        prediction = tile.predict_proba(X.iloc[test], embeddings.take(test))
    predict_time = time.perf_counter() - start

    window = tile.features_.window
    return {
        **scores(y[test], prediction, task),
        "n_train": len(train),
        "n_test": len(test),
        "window": f"{window[0]}-{window[1]}" if window else None,
        "components_per_column": tile.features_.n_components_,
        "n_features": tile.features_.n_base_ + len(tile.features_.categories_),
        "embedding_time": embeddings.info["time_s"]
        * (len(train) + len(test))
        / embeddings.array.shape[1],
        "fit_time": fit_time,
        "predict_time": predict_time,
        **_gpu_peak(device),
    }
