"""Step 7: the classifiers that sit on top of the ring-pooled features.

Logistic regression is the main one: it's linear (easy to export to the
browser as a weight matrix) and it gives probabilities. The others are there
to check we aren't leaving accuracy on the table.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SEED = 42


class CalibratedLogReg:
    """Logistic regression + temperature scaling.

    Balanced class weights push the model to be over-confident on rare classes.
    Temperature scaling fixes that with one number T: probabilities become
    softmax(scores / T). T is picked on out-of-fold scores from 3 inner folds of
    the TRAINING data, so validation/test wafers never touch it. Dividing every
    score by the same T never changes which class wins, so F1 is unchanged;
    only the confidence numbers get more honest.
    """

    def __init__(self):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=SEED))
        self.temperature = 1.0

    def fit(self, X, y):
        inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
        scores = cross_val_predict(self.model, X, y, cv=inner, method="decision_function")
        self.model.fit(X, y)
        target = np.searchsorted(self.model.classes_, y)

        def nll(t):
            return -log_softmax(scores / t, axis=1)[np.arange(len(y)), target].mean()

        self.temperature = minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return np.exp(log_softmax(self.model.decision_function(X) / self.temperature, axis=1))


def make_model(name: str):
    if name == "logreg":
        return CalibratedLogReg()
    if name == "ridge":
        return make_pipeline(StandardScaler(), RidgeClassifier(class_weight="balanced"))
    if name == "svm":
        return make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", C=0.1,
                                                         max_iter=5000, random_state=SEED))
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2,
                                      n_jobs=-1, random_state=SEED)
    raise ValueError(f"unknown model {name}")
