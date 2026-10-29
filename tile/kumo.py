"""Scikit-learn style wrappers of KumoTabular, from NVIDIA's
structured-data-models (https://github.com/NVIDIA/structured-data-models).

The settings follow the TabArena protocol of the package
(`benchmark/tabular/talent/models.py`): 8 estimators for the small and medium
models, 16 for the large one, float16 autocast on GPU, and the mean of the 999
predicted quantiles as the point prediction of a regression.

The columns of strings are its categorical features; it aligns their
categories between the context and the query by value, and maps them to
numbers itself. Its recipe keeps at most 500 columns per estimator.
"""

import numpy as np
import pandas as pd
import torch

N_ESTIMATORS = {"small": 8, "medium": 8, "large": 16}


class _KumoTabular:
    task = None

    def __init__(self, size="large", device="cuda", random_state=1234):
        import sdm

        self.size = size
        self.device = torch.device(device)
        self.random_state = random_state
        self.model = sdm.models.KumoTabular(
            task=self.task, size=size, device=self.device
        )

    def _table(self, X):
        import sdm

        X = pd.DataFrame(X)
        X.columns = [str(col) for col in X.columns]
        if not hasattr(self, "stypes_"):
            self.stypes_ = {
                col: "categorical" if X[col].dtype == object else "numerical"
                for col in X.columns
            }
        return sdm.TableTensor.from_pandas(df=X, stypes=self.stypes_, device=self.device)

    def _fit(self, X, y, y_stype):
        import sdm

        if hasattr(self, "stypes_"):
            del self.stypes_
        x_context = self._table(X)
        y_context = sdm.TableTensor.from_pandas(
            df=pd.DataFrame({"target": np.asarray(y)}),
            stypes={"target": y_stype},
            device=self.device,
        )
        generator = torch.Generator(self.device).manual_seed(self.random_state)
        with torch.amp.autocast(
            self.device.type, torch.float16, enabled=self.device.type == "cuda"
        ):
            self.model.fit(
                x=x_context,
                y=y_context,
                num_estimators=N_ESTIMATORS[self.size],
                generator=generator,
            )
        return self

    def _predict_table(self, X):
        with torch.amp.autocast(
            self.device.type, torch.float16, enabled=self.device.type == "cuda"
        ):
            return self.model.predict(self._table(X))


class KumoTabularClassifier(_KumoTabular):
    task = "classification"

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self._fit(X, y, "categorical")

    def predict_proba(self, X):
        from sdm.stype import Stype

        out = self._predict_table(X)
        # One output column per class, named by its value
        columns = list(out.columns[Stype.numerical])
        indices = [columns.index(str(c)) for c in self.classes_]
        return out.numerical[..., indices].float().cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class KumoTabularRegressor(_KumoTabular):
    task = "regression"

    def fit(self, X, y):
        return self._fit(X, np.asarray(y, dtype=float), "numerical")

    def predict(self, X):
        out = self._predict_table(X)
        return out.numerical.float().mean(dim=-1).cpu().numpy()
