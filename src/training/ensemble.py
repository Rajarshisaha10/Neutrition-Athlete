from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class WeightedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators: list[tuple[str, object, float]]):
        self.estimators = estimators
        total = sum(max(weight, 0.0) for _, _, weight in estimators)
        self.weights_ = [weight / total if total else 1.0 / len(estimators) for _, _, weight in estimators]
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        probs = []
        for (_, estimator, _), weight in zip(self.estimators, self.weights_):
            probs.append(estimator.predict_proba(X) * weight)
        return np.sum(probs, axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
