"""The two models, each as a single pipeline that takes the MODEL_FEATURES frame."""

from __future__ import annotations

from typing import get_args

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from loan_default.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from loan_default.schemas import HomeOwnership, Purpose, VerificationStatus

# Heavy right tails: the linear baseline sees these on a log scale.
SKEWED_FEATURES = ["loan_amnt", "annual_inc", "revol_bal"]


def _linear_preprocess(sparse_output: bool = True) -> ColumnTransformer:
    """One-hot categories, median imputation, scaling, and a log scale for skewed amounts."""
    plain = [c for c in NUMERIC_FEATURES if c not in SKEWED_FEATURES]
    impute = SimpleImputer(strategy="median", add_indicator=True)
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output), CATEGORICAL_FEATURES),
            ("num", make_pipeline(impute, StandardScaler()), plain),
            (
                "skewed",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                    StandardScaler(),
                ),
                SKEWED_FEATURES,
            ),
        ]
    )


def build_baseline() -> Pipeline:
    """Logistic regression with one-hot categories and median imputation."""
    return Pipeline([("preprocess", _linear_preprocess()), ("model", LogisticRegression(max_iter=1000))])


def build_linear_svm() -> Pipeline:
    """Linear support vector machine. Its distances from the margin are calibrated to probabilities later."""
    # dual=False is the fast choice when there are many more rows than columns.
    model = LinearSVC(dual=False, random_state=0)
    return Pipeline([("preprocess", _linear_preprocess()), ("model", model)])


def build_naive_bayes() -> Pipeline:
    """Gaussian Naive Bayes on the same inputs as the baseline. Needs dense input."""
    return Pipeline([("preprocess", _linear_preprocess(sparse_output=False)), ("model", GaussianNB())])


class LogOddsLGBMClassifier(LGBMClassifier):
    """LightGBM that also exposes its raw score, so calibration works on log-odds."""

    def decision_function(self, X):
        return self.predict(X, raw_score=True)


# Starting point for LightGBM. Tuning overrides some of these (see tune.py).
GBM_DEFAULTS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 200,
    "random_state": 0,
    "verbose": -1,
}


def build_gbm(**params) -> Pipeline:
    """LightGBM gradient boosting. Handles missing values natively."""
    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = LogOddsLGBMClassifier(**{**GBM_DEFAULTS, **params})
    return Pipeline([("preprocess", preprocess), ("model", model)])


# +1: the score may only rise as the feature rises; -1: only fall. Chosen where the
# default rate moves in one direction across all ten deciles of the training data and
# the direction makes credit sense (docs/retraining.md). Other features stay free.
MONOTONE_DIRECTIONS = {
    "fico": -1,
    "annual_inc": -1,
    "credit_history_months": -1,
    "mort_acc": -1,
    "dti": 1,
    "revol_util": 1,
    "acc_open_past_24mths": 1,
}


def build_gbm_monotone(**params) -> Pipeline:
    """LightGBM with monotonic constraints on MONOTONE_DIRECTIONS.

    The categories are fixed to the API's lists, so the one-hot columns always have the
    same width and the constraint list lines up with the columns.
    """
    categories = [list(get_args(level)) for level in (Purpose, HomeOwnership, VerificationStatus)]
    width = sum(len(c) for c in categories)
    constraints = [0] * width + [MONOTONE_DIRECTIONS.get(name, 0) for name in NUMERIC_FEATURES]
    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = LogOddsLGBMClassifier(**{**GBM_DEFAULTS, **params, "monotone_constraints": constraints})
    return Pipeline([("preprocess", preprocess), ("model", model)])


def fit_calibrated(pipeline: Pipeline, X_fit, y_fit, X_cal, y_cal) -> CalibratedClassifierCV:
    """Fit the pipeline, then calibrate its probabilities on later, unseen loans."""
    pipeline.fit(X_fit, y_fit)
    calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
    return calibrated.fit(X_cal, y_cal)


def predict_pd(model, X) -> np.ndarray:
    """Probability of default for each row."""
    return model.predict_proba(X)[:, 1]
