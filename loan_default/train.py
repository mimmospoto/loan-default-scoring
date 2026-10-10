"""Tune, validate and train the models, evaluate them out of time, and register the result.

Run: python -m loan_default.train
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import lightgbm
import pandas as pd
import sklearn

from loan_default.bands import fit_thresholds
from loan_default.config import (
    BAND_LABELS,
    CALIBRATION_PERIOD,
    FIT_PERIOD,
    PROCESSED_FILE,
    TEST_PERIOD,
    TRAIN_PERIOD,
)
from loan_default.features import MODEL_FEATURES
from loan_default.metrics import band_table, discrimination, feature_psi, psi
from loan_default.model import build_baseline, build_gbm, fit_calibrated, predict_pd
from loan_default.registry import log_run
from loan_default.splits import between, split_xy
from loan_default.tune import tune_gbm
from loan_default.validate import walk_forward


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune, train and register the model.")
    parser.add_argument("--name", help="run name in MLflow (default: lgbm-<date>-<time>)")
    parser.add_argument("--note", help="why this run was made, shown in 'registry list'")
    args = parser.parse_args()

    if not PROCESSED_FILE.exists():
        print("No modelling table. Run: python -m loan_default.data.prepare", file=sys.stderr)
        return 1
    table = pd.read_parquet(PROCESSED_FILE)

    X_fit, y_fit = split_xy(between(table, FIT_PERIOD))
    X_cal, y_cal = split_xy(between(table, CALIBRATION_PERIOD))
    X_train, y_train = split_xy(between(table, TRAIN_PERIOD))
    X_test, y_test = split_xy(between(table, TEST_PERIOD))
    print(f"fit {len(X_fit):,}  calibrate {len(X_cal):,}  test {len(X_test):,}")
    print(f"default rate: train {y_train.mean():.3f}  test {y_test.mean():.3f}\n")

    gbm_params, tuning = tune_gbm(between(table, TRAIN_PERIOD))
    print("LightGBM tuning, best five settings (AUC on time-ordered folds):")
    print(tuning.head(5).round(4).to_string(index=False))
    print(f"Chosen: {gbm_params}\n")

    folds = walk_forward(table, gbm_params)
    print("Walk-forward validation (inside the train period):")
    print(folds.round(4).to_string(index=False), "\n")

    models = {
        "baseline": fit_calibrated(build_baseline(), X_fit, y_fit, X_cal, y_cal),
        "gbm": fit_calibrated(build_gbm(**gbm_params), X_fit, y_fit, X_cal, y_cal),
    }
    test_metrics = {name: discrimination(y_test, predict_pd(m, X_test)) for name, m in models.items()}
    print("Test period:")
    print(pd.DataFrame(test_metrics).T.round(4).to_string(), "\n")

    model = models["gbm"]
    pd_train, pd_test = predict_pd(model, X_train), predict_pd(model, X_test)
    thresholds = fit_thresholds(pd_train)
    bands = band_table(y_test, pd_test, thresholds)
    print(f"Band thresholds (from training PDs): {thresholds}")
    print(bands.round(4).to_string(), "\n")

    score_psi = psi(pd_train, pd_test)
    features_psi = feature_psi(X_train, X_test)
    print(f"PSI train -> test, model PD: {score_psi:.4f}")
    print(features_psi.head(5).round(4).to_string(), "\n")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lightgbm.__version__,
        "features": MODEL_FEATURES,
        "band_labels": BAND_LABELS,
        "band_thresholds": thresholds,
        "periods": {"fit": FIT_PERIOD, "calibration": CALIBRATION_PERIOD, "test": TEST_PERIOD},
        "gbm_params": gbm_params,
        "tuning": tuning.to_dict(orient="records"),
        "walk_forward": folds.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "test_bands": bands.to_dict(orient="index"),
        "psi_train_test": {"pd": score_psi, "features": features_psi.to_dict()},
    }
    params = {
        **gbm_params,
        "fit_period": " to ".join(FIT_PERIOD),
        "calibration_period": " to ".join(CALIBRATION_PERIOD),
        "test_period": " to ".join(TEST_PERIOD),
        "n_features": len(MODEL_FEATURES),
    }
    metrics = {f"test_{metric}_{name}": value for name, scores in test_metrics.items() for metric, value in scores.items()}
    metrics["psi_pd_train_test"] = score_psi
    metrics["test_mean_pd"] = float(pd_test.mean())
    metrics["test_default_rate"] = float(y_test.mean())
    version = log_run(model, metadata, params, metrics, input_example=X_test.head(5), name=args.name, note=args.note)
    print(f"Logged to MLflow and registered as version {version}.")
    print(f"To put it live: python -m loan_default.registry promote {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
