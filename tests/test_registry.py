import json

import joblib
import numpy as np
import pandas as pd
import pytest

from loan_default import registry
from loan_default.features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES
from loan_default.model import build_gbm, fit_calibrated, predict_pd


def synthetic_loans(n: int = 400) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, len(NUMERIC_FEATURES))), columns=NUMERIC_FEATURES)
    X.loc[::7, "mths_since_last_delinq"] = np.nan
    for column in CATEGORICAL_FEATURES:
        X[column] = rng.choice(["a", "b", "c"], size=n)
    y = pd.Series((X["dti"] + rng.normal(size=n) > 0.8).astype(int))
    return X[MODEL_FEATURES], y


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    """A throwaway MLflow store, so tests never touch the real one."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(registry, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_exported_champion_predicts_like_the_trained_model(local_mlflow):
    X, y = synthetic_loans()
    model = fit_calibrated(build_gbm(n_estimators=20, min_child_samples=5), X[:200], y[:200], X[200:300], y[200:300])
    metadata = {"features": MODEL_FEATURES, "band_thresholds": [0.1, 0.2, 0.3, 0.4]}

    version = registry.log_run(model, metadata, {"n_estimators": 20}, {"test_auc_gbm": 0.7}, X.head(5))
    registry.promote(version)
    exported = registry.export(local_mlflow / "artifacts")

    assert exported["model_version"] == version
    assert exported["band_thresholds"] == metadata["band_thresholds"]
    on_disk = json.loads((local_mlflow / "artifacts" / "metadata.json").read_text())
    assert on_disk["model_version"] == version

    reloaded = joblib.load(local_mlflow / "artifacts" / "model.joblib")
    np.testing.assert_array_equal(predict_pd(reloaded, X[300:]), predict_pd(model, X[300:]))


def test_promote_moves_the_champion(local_mlflow):
    X, y = synthetic_loans()
    model = fit_calibrated(build_gbm(n_estimators=5, min_child_samples=5), X[:200], y[:200], X[200:300], y[200:300])
    first = registry.log_run(model, {}, {}, {"test_auc_gbm": 0.6}, X.head(5))
    second = registry.log_run(model, {}, {}, {"test_auc_gbm": 0.7}, X.head(5))

    registry.promote(first)
    registry.promote(second)

    champions = [row["version"] for row in registry.list_versions() if row["champion"]]
    assert champions == [int(second)]


def test_run_name_and_note_are_recorded(local_mlflow):
    X, y = synthetic_loans()
    model = fit_calibrated(build_gbm(n_estimators=5, min_child_samples=5), X[:200], y[:200], X[200:300], y[200:300])
    registry.log_run(model, {}, {}, {"test_auc_gbm": 0.6}, X.head(5), name="wider grid", note="more leaves")
    registry.log_run(model, {}, {}, {"test_auc_gbm": 0.7}, X.head(5))

    by_version = {row["version"]: row for row in registry.list_versions()}
    assert by_version[1]["name"] == "wider grid"
    assert by_version[1]["note"] == "more leaves"
    assert by_version[2]["name"].startswith("lgbm-")
    assert by_version[2]["note"] == ""
