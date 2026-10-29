"""The input TILE gives a tabular learner: the reduced cell embeddings of the
string columns, next to the features that bypass the LLM."""

from math import ceil, floor

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

# The PCA components of a string column, before the feature window
COMPONENTS_PER_COLUMN = 30
# A string column with fewer distinct values also reaches the learners that
# take categories as one
CARDINALITY_THRESHOLD = 50
# A string column is worth target-encoding only if enough of its rows share
# their value with another row. Below this fraction the column is nearly all
# singletons: out of fold almost every row falls back to the prior, so the
# encoding is close to a constant.
MIN_REPEAT_RATIO = 0.1


def string_columns(table):
    """The columns of text, in the order of the table: the dates are not."""
    from skrub import to_datetime

    return [
        col
        for col in table.select_dtypes(include="object").columns
        if not pd.api.types.is_datetime64_any_dtype(to_datetime(table[col]))
    ]


def components_per_column(n_columns, window, default=COMPONENTS_PER_COLUMN):
    """The number of PCA components of each of `n_columns` string columns.

    `default` each, unless their total falls outside `window`, a (lower,
    upper) number of features: then the count that brings the total to the
    bound it crosses, rounded towards the inside of the window, at least 1.
    """
    total = default * n_columns
    if n_columns == 0 or window is None or window[0] <= total <= window[1]:
        return default
    if total < window[0]:
        return max(1, ceil(window[0] / n_columns))
    return max(1, floor(window[1] / n_columns))


def _repeat_ratio(values):
    """Fraction of non-missing rows whose value occurs at least twice."""
    present = pd.Series(values).dropna().astype(str)
    if len(present) == 0:
        return 0.0
    counts = present.value_counts()
    return counts[counts >= 2].sum() / len(present)


def _target_stats(codes, y, n_classes):
    """Category -> its encoded row, and the row of a category never seen."""
    if n_classes == 0:
        means = pd.Series(y, index=codes).groupby(level=0).mean()
        return means.to_frame(), np.array([float(np.mean(y))])
    counts = pd.crosstab(codes, y).reindex(columns=range(n_classes), fill_value=0)
    proba = counts.div(counts.sum(axis=1).clip(lower=1), axis=0)
    prior = np.bincount(y, minlength=n_classes) / len(y)
    # The class probabilities sum to one: a binary target needs only P(y = 1)
    if n_classes == 2:
        return proba[[1]], prior[[1]]
    return proba, prior


def _encode(codes, mapping, fallback):
    # `copy=True`: under pandas 3 the array shares the frame's buffer and
    # comes back read-only
    values = mapping.reindex(pd.Index(codes)).to_numpy(dtype=float, copy=True)
    values[np.isnan(values).any(axis=1)] = fallback
    return values


class TargetEncoder:
    """The target encoding of the string columns whose values repeat.

    A category is encoded by P(y = k | category), one column per class (P(y = 1)
    alone for a binary target), or by its mean target for a regression; a
    missing value is a category of its own. The training rows are encoded out
    of fold, so that a row never contributes to its own encoding; the other
    rows by the statistics of every training row. A category never seen gets
    the prior.
    """

    def fit_transform(self, X, y, task, n_splits=5):
        self.columns_ = [
            col for col in X.columns if _repeat_ratio(X[col]) > MIN_REPEAT_RATIO
        ]
        y = np.asarray(y)
        self.n_classes_ = 0
        if task == "classification":
            classes, y = np.unique(y, return_inverse=True)
            self.n_classes_ = len(classes)
        width = self.n_classes_ if self.n_classes_ > 2 else 1

        n = len(X)
        folds = list(
            KFold(min(n_splits, n), shuffle=True, random_state=1234).split(
                np.arange(n)
            )
        )
        encoded = np.empty((n, len(self.columns_) * width))
        self.stats_ = []
        for j, col in enumerate(self.columns_):
            block = slice(j * width, (j + 1) * width)
            codes = X[col].astype(str).to_numpy()
            self.stats_.append(_target_stats(codes, y, self.n_classes_))
            for fit_index, encode_index in folds:
                stats = _target_stats(codes[fit_index], y[fit_index], self.n_classes_)
                encoded[encode_index, block] = _encode(codes[encode_index], *stats)
        return encoded

    def transform(self, X):
        blocks = [
            _encode(X[col].astype(str).to_numpy(), *stats)
            for col, stats in zip(self.columns_, self.stats_)
        ]
        return np.hstack(blocks) if blocks else np.empty((len(X), 0))


class TILEFeatures:
    """The input of the learner, fit on the training rows.

    In this order:
        * the cell embeddings of each string column, standardized and reduced
          by a PCA of their own;
        * the other columns, numbers and dates, through skrub's
          TableVectorizer;
        * the target encoding of the string columns (`TargetEncoder`);
        * with `native_categories`, the string columns of fewer than
          `CARDINALITY_THRESHOLD` distinct values, as strings, for the
          learners that handle categories.

    With a `window`, a (lower, upper) number of features, the PCA components
    are rescaled so that the whole input falls inside it: the features that
    bypass the LLM take their share of the window first. Without one, every
    string column keeps `COMPONENTS_PER_COLUMN` components.
    `power_transform` imputes the numbers and dates and maps them to a normal
    distribution, for a linear learner.
    """

    def __init__(self, window=None, native_categories=False, power_transform=False):
        self.window = window
        self.native_categories = native_categories
        self.power_transform = power_transform

    def fit_transform(self, X, y, embeddings, task):
        from skrub import TableVectorizer

        self.string_columns_ = string_columns(X)
        strings = set(self.string_columns_)
        self.other_columns_ = [col for col in X.columns if col not in strings]

        self.vectorizer_ = TableVectorizer(numeric="passthrough")
        if self.power_transform:
            self.vectorizer_ = make_pipeline(
                self.vectorizer_, SimpleImputer(), PowerTransformer()
            )
        others = X[self.other_columns_]
        numbers = (
            np.array(self.vectorizer_.fit_transform(others))
            if others.shape[1] > 0
            else np.empty((len(X), 0))
        )

        self.target_encoder_ = TargetEncoder()
        encoded = self.target_encoder_.fit_transform(
            X[self.string_columns_], y, task
        )

        self.categories_ = []
        if self.native_categories:
            self.categories_ = [
                col
                for col in self.string_columns_
                if X[col].nunique() < CARDINALITY_THRESHOLD
            ]

        n_bypass = numbers.shape[1] + encoded.shape[1] + len(self.categories_)
        window = self.window
        if window is not None:
            window = tuple(max(0, bound - n_bypass) for bound in window)
        self.n_components_ = components_per_column(len(self.string_columns_), window)

        self.reducers_ = {}
        reduced = []
        for col in self.string_columns_:
            block = embeddings[col]
            reducer = make_pipeline(
                StandardScaler(),
                PCA(int(min(self.n_components_, *block.shape)), random_state=1234),
            )
            reduced.append(reducer.fit_transform(block))
            self.reducers_[col] = reducer

        Z = self._assemble(reduced, numbers, encoded, X)
        # The categories come last
        self.n_base_ = Z.shape[1] - len(self.categories_)
        return Z

    def transform(self, X, embeddings):
        reduced = [
            self.reducers_[col].transform(embeddings[col])
            for col in self.string_columns_
        ]
        others = X[self.other_columns_]
        numbers = (
            np.array(self.vectorizer_.transform(others))
            if others.shape[1] > 0
            else np.empty((len(X), 0))
        )
        encoded = self.target_encoder_.transform(X[self.string_columns_])
        return self._assemble(reduced, numbers, encoded, X)

    def _assemble(self, reduced, numbers, encoded, X):
        Z = np.hstack(reduced + [numbers, encoded])
        if self.categories_:
            # A missing category is the string "nan"
            Z = np.hstack((Z, X[self.categories_].astype(str).to_numpy()))
        return Z
