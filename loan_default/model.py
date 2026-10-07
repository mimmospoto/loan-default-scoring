"""The two models, each as a single pipeline that takes the MODEL_FEATURES frame."""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from loan_default.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Heavy right tails: the linear baseline sees these on a log scale.
SKEWED_FEATURES = ["loan_amnt", "annual_inc", "revol_bal"]


def build_baseline() -> Pipeline:
    """Logistic regression with one-hot categories and median imputation."""
    plain = [c for c in NUMERIC_FEATURES if c not in SKEWED_FEATURES]
    impute = SimpleImputer(strategy="median", add_indicator=True)
    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
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
    return Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=1000))])


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


def fit_calibrated(pipeline: Pipeline, X_fit, y_fit, X_cal, y_cal) -> CalibratedClassifierCV:
    """Fit the pipeline, then calibrate its probabilities on later, unseen loans."""
    pipeline.fit(X_fit, y_fit)
    calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
    return calibrated.fit(X_cal, y_cal)


def predict_pd(model, X) -> np.ndarray:
    """Probability of default for each row."""
    return model.predict_proba(X)[:, 1]
