"""TILE on top of a tabular learner."""

import numpy as np

from tile.features import TILEFeatures
from tile.learners import LEARNERS, make_learner, predict, predict_proba, tune_xgboost


class TILE:
    """TILE (Table-Interfaced LLM Embeddings) on top of a tabular learner.

    `fit` and the predictions take the table without its target, and the cell
    embeddings of the same rows (`Embeddings`, restricted to these rows with
    `take`). The string columns reach the learner through their embeddings,
    reduced into the feature window of the learner (see `TILEFeatures`).

    Parameters
    ----------
    learner : str
        One of `LEARNERS`.
    task : str
        "classification" or "regression".
    device : str
        Where the learner runs, "cuda" or "cpu".
    window : tuple, optional
        The feature window, a (lower, upper) number of features, in place of
        that of the learner in `LEARNERS`.
    """

    def __init__(
        self, learner="tabpfn-2.5", task="classification", device="cuda", window=None
    ):
        self.learner = learner
        self.task = task
        self.device = device
        self.window = window

    def fit(self, X, y, embeddings):
        spec = LEARNERS[self.learner]
        y = np.asarray(y)
        if self.task == "classification":
            self.classes_, y = np.unique(y, return_inverse=True)
        else:
            self.y_mean_ = float(np.mean(y))

        self.features_ = TILEFeatures(
            spec.window if self.window is None else self.window,
            spec.native_categories,
            power_transform=self.learner == "ridge",
        )
        Z = self.features_.fit_transform(X, y, embeddings, self.task)
        params = tune_xgboost(Z, y, self.task) if self.learner == "xgboost" else {}
        self.model_ = make_learner(
            self.learner, self.task, self.device, self.features_.n_base_, params
        )
        self.model_.fit(Z, y)
        return self

    def predict_proba(self, X, embeddings):
        """The probability of each class of `classes_`."""
        return predict_proba(self.model_, self.features_.transform(X, embeddings))

    def predict(self, X, embeddings):
        if self.task == "classification":
            return self.classes_[self.predict_proba(X, embeddings).argmax(axis=1)]
        y = predict(self.model_, self.features_.transform(X, embeddings))
        # A learner that fails on a row predicts the mean target there
        y[np.isnan(y)] = self.y_mean_
        return y
