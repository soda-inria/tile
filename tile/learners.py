"""The tabular learners TILE was run with, and their feature windows."""

import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder


@dataclass(frozen=True)
class Learner:
    # The (lower, upper) number of input features the learner handles best,
    # selected on the 27 tables of the probing (see the paper). None for the
    # learners without one: their string columns keep 30 components each.
    window: tuple = None
    # Whether the learner takes the low-cardinality string columns as categories
    native_categories: bool = False


LEARNERS = {
    "tabpfn-2.5": Learner((200, 400), True),
    "tabpfn-3": Learner((150, 400), True),
    "tabpfn-3.5": Learner((200, 800), True),
    "tabicl": Learner((200, 800), True),
    "exaone": Learner((150, 400), True),
    "causilo": Learner((200, 800), True),
    "kumo-small": Learner((200, 600), True),
    "kumo-medium": Learner((200, 600), True),
    "kumo-large": Learner((200, 600), True),
    "realmlp": Learner(),
    "xgboost": Learner(),
    "catboost": Learner(),
    "ridge": Learner(),
}

# The search space of XGBoost, the only learner tuned (`tune_xgboost`)
XGBOOST_SEARCH = {
    "n_estimators": randint(50, 1001),
    "max_depth": randint(2, 7),
    "min_child_weight": loguniform(1, 100),
    "subsample": uniform(0.5, 1 - 0.5),
    "learning_rate": loguniform(1e-5, 1),
    "colsample_bylevel": uniform(0.5, 1 - 0.5),
    "colsample_bytree": uniform(0.5, 1 - 0.5),
    "gamma": loguniform(1e-8, 7),
    "reg_lambda": loguniform(1, 4),
    "alpha": loguniform(1e-8, 100),
}

# TabPFN predicts every test row on its own: it predicts them by chunks of
# this many rows, to bound its memory
TABPFN_CHUNK = 1563


class _Frame(TransformerMixin, BaseEstimator):
    """The input as a DataFrame: the first `n_base` columns as floats, the
    others, the categories, as strings.

    `drop_empty` drops the columns missing on every training row, and
    `missing_categories` turns the category "nan" back into a missing value.
    """

    def __init__(self, n_base, drop_empty=False, missing_categories=False):
        self.n_base = n_base
        self.drop_empty = drop_empty
        self.missing_categories = missing_categories

    def fit(self, X, y=None):
        X = self._frame(X)
        self.columns_ = [
            col for col in X.columns if not (self.drop_empty and X[col].isna().all())
        ]
        return self

    def transform(self, X):
        return self._frame(X)[self.columns_]

    def _frame(self, X):
        X = pd.DataFrame(X)
        for col in X.columns[: self.n_base]:
            X[col] = X[col].astype(float)
        if self.missing_categories:
            for col in X.columns[self.n_base :]:
                X[col] = X[col].replace("nan", np.nan)
        return X


class _Ordinal(TransformerMixin, BaseEstimator):
    """The input as floats only, the categories as ordinal codes, the missing
    and unseen ones NaN."""

    def __init__(self, n_base):
        self.n_base = n_base

    def fit(self, X, y=None):
        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )
        categories = self._categories(X)
        if categories.shape[1] > 0:
            self.encoder_.fit(categories)
        return self

    def transform(self, X):
        base = np.asarray(X[:, : self.n_base], dtype=float)
        categories = self._categories(X)
        if categories.shape[1] == 0:
            return base
        return np.hstack((base, self.encoder_.transform(categories)))

    def _categories(self, X):
        categories = np.array(X[:, self.n_base :], dtype=object)
        categories[categories == "nan"] = np.nan
        return categories


def make_learner(name, task, device="cuda", n_base=None, params=None):
    """The learner `name` of LEARNERS, for its input from `TILEFeatures`.

    `n_base` is the number of columns of the input before the categories, and
    `params` the hyperparameters of XGBoost. The checkpoints of the foundation
    models are downloaded on first use.
    """
    regression = task == "regression"

    if name.startswith("tabpfn"):
        # TabPFN-3.5 needs tabpfn 9, the others tabpfn 8
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion

        version = ModelVersion("v" + name.split("-")[1])
        model = TabPFNRegressor if regression else TabPFNClassifier
        return model.create_default_for_version(
            version, device=device, ignore_pretraining_limits=True
        )

    if name == "tabicl":
        from tabicl import TabICLClassifier, TabICLRegressor

        model = TabICLRegressor if regression else TabICLClassifier
        return make_pipeline(
            _Frame(n_base, drop_empty=True), model(device=device, random_state=1234)
        )

    if name == "exaone":
        from exaonetabular import EXAONETabularClassifier, EXAONETabularRegressor

        model = EXAONETabularRegressor if regression else EXAONETabularClassifier
        return make_pipeline(_Ordinal(n_base), model.from_pretrained(device=device))

    if name == "causilo":
        # The columns of strings are its categorical features
        from causilo import CausiloClassifier, CausiloRegressor

        model = CausiloRegressor if regression else CausiloClassifier
        return model(device=device, random_state=1234)

    if name.startswith("kumo"):
        from tile.kumo import KumoTabularClassifier, KumoTabularRegressor

        model = KumoTabularRegressor if regression else KumoTabularClassifier
        return make_pipeline(
            _Frame(n_base, missing_categories=True),
            model(size=name.split("-")[1], device=device, random_state=1234),
        )

    if name == "realmlp":
        # Tuned by its own search
        from pytabkit import RealMLP_HPO_Classifier, RealMLP_HPO_Regressor

        settings = dict(
            device=device,
            n_cv=5,
            n_repeats=1,
            n_hyperopt_steps=50,
            random_state=20122024,
            tmp_folder=tempfile.mkdtemp(prefix="realmlp_"),
        )
        if regression:
            model = RealMLP_HPO_Regressor(**settings)
        else:
            model = RealMLP_HPO_Classifier(val_metric_name="1-auc_ovr", **settings)
        return make_pipeline(SimpleImputer(), model)

    if name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        model = XGBRegressor if regression else XGBClassifier
        return model(**(params or {}))

    if name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        # Its fit depends on the number of columns, the empty ones included
        model = CatBoostRegressor if regression else CatBoostClassifier
        return make_pipeline(
            _Frame(n_base, drop_empty=True),
            model(allow_writing_files=False, verbose=False, thread_count=-1),
        )

    if name == "ridge":
        from sklearn.linear_model import RidgeClassifierCV, RidgeCV

        model = RidgeCV if regression else RidgeClassifierCV
        alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
        return make_pipeline(SimpleImputer(), model(alphas=alphas))

    raise ValueError(f"Unknown learner {name!r}, one of {list(LEARNERS)}")


def tune_xgboost(X, y, task, n_iter=100):
    """The hyperparameters of XGBoost: the best of its defaults and
    `n_iter - 1` random draws of `XGBOOST_SEARCH`, by a 5-fold
    cross-validation repeated 5 times. An empty dict for the defaults."""
    from sklearn.model_selection import (
        RandomizedSearchCV,
        RepeatedKFold,
        RepeatedStratifiedKFold,
        cross_val_score,
    )

    if task == "regression":
        cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1234)
        scoring = "r2"
    else:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1234)
        scoring = "roc_auc" if len(np.unique(y)) == 2 else "roc_auc_ovr"

    default = make_learner("xgboost", task)
    default_score = cross_val_score(
        default, X, y, cv=cv, scoring=scoring, n_jobs=-1
    ).mean()
    search = RandomizedSearchCV(
        default,
        XGBOOST_SEARCH,
        n_iter=n_iter - 1,
        cv=cv,
        scoring=scoring,
        refit=False,
        n_jobs=-1,
        random_state=1234,
    ).fit(X, y)
    return search.best_params_ if search.best_score_ >= default_score else {}


def _chunked(method, model, X):
    if type(model).__name__.startswith("TabPFN"):
        return np.concatenate(
            [method(X[i : i + TABPFN_CHUNK]) for i in range(0, len(X), TABPFN_CHUNK)]
        )
    return method(X)


def predict_proba(model, X):
    """The class probabilities of a fitted learner."""
    if hasattr(model, "predict_proba"):
        return _chunked(model.predict_proba, model, X)
    # RidgeClassifierCV: the sigmoid of its decision function
    scores = expit(model.decision_function(X))
    if scores.ndim == 1:
        return np.column_stack((1 - scores, scores))
    return scores / scores.sum(axis=1, keepdims=True)


def predict(model, X):
    """The predictions of a fitted regressor."""
    return _chunked(model.predict, model, X)
