"""Challenge the champion with new models, and promote the best only if it passes a gate.

Run:
    python -m loan_default.retrain                # compare only, nothing is saved
    python -m loan_default.retrain --register     # also log the challenger as a new version
    python -m loan_default.retrain --promote      # register, and promote if the gate passes

The challengers are fitted and calibrated on the same windows as the champion.
The best one in walk-forward validation (inside the train period) is then scored
once on the test period, so the test period is not used to choose between them.
Promotion never exports or deploys: that stays a separate, manual step
(docs/retraining.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm
import pandas as pd
import sklearn

from loan_default.bands import fit_thresholds
from loan_default.config import (
    ARTIFACTS_DIR,
    BAND_LABELS,
    CALIBRATION_PERIOD,
    FIT_PERIOD,
    PROCESSED_FILE,
    TEST_PERIOD,
    TRAIN_PERIOD,
)
from loan_default.features import MODEL_FEATURES
from loan_default.metrics import band_table, discrimination, feature_psi, psi
from loan_default.model import (
    build_gbm,
    build_gbm_monotone,
    build_linear_svm,
    build_naive_bayes,
    fit_calibrated,
    predict_pd,
)
from loan_default.registry import promote, log_run
from loan_default.splits import between, split_xy
from loan_default.validate import walk_forward

# Gate values: a starting point, not tuned on this data.
MAX_CALIBRATION_GAP = 0.01  # mean predicted PD within 1 percentage point of the observed rate

# Every challenger gets a fresh pipeline. The LightGBM ones use the champion's tuned settings.
CHALLENGERS = {
    "gbm-monotone": lambda params: build_gbm_monotone(**params),
    "linear-svm": lambda params: build_linear_svm(),
    "naive-bayes": lambda params: build_naive_bayes(),
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def evaluate(model, X, y) -> dict[str, float]:
    """The numbers the gate looks at, on one holdout."""
    probabilities = predict_pd(model, X)
    scores = discrimination(y, probabilities)
    mean_pd, observed = float(probabilities.mean()), float(y.mean())
    return {**scores, "mean_pd": mean_pd, "observed": observed, "calibration_gap": mean_pd - observed}


def gate(champion: dict[str, float], challenger: dict[str, float]) -> list[Check]:
    """Three checks. The challenger is promotable only if all of them pass."""
    return [
        Check(
            "AUC no lower than the champion",
            challenger["auc"] >= champion["auc"],
            f"{challenger['auc']:.4f} against {champion['auc']:.4f}",
        ),
        Check(
            "Brier no worse than the champion",
            challenger["brier"] <= champion["brier"],
            f"{challenger['brier']:.4f} against {champion['brier']:.4f}",
        ),
        Check(
            f"Mean PD within {MAX_CALIBRATION_GAP:.0%} of the observed rate",
            abs(challenger["calibration_gap"]) <= MAX_CALIBRATION_GAP,
            f"gap {challenger['calibration_gap']:+.4f}",
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a challenger and compare it with the champion.")
    parser.add_argument(
        "--challengers",
        nargs="+",
        choices=list(CHALLENGERS),
        default=list(CHALLENGERS),
        help="which models compete; the best in walk-forward validation faces the champion",
    )
    parser.add_argument("--register", action="store_true", help="log the challenger to MLflow as a new version")
    parser.add_argument("--promote", action="store_true", help="register, and make it champion if the gate passes")
    args = parser.parse_args()

    if not PROCESSED_FILE.exists():
        print("No modelling table. Run: python -m loan_default.data.prepare", file=sys.stderr)
        return 2
    model_file, metadata_file = ARTIFACTS_DIR / "model.joblib", ARTIFACTS_DIR / "metadata.json"
    if not model_file.exists() or not metadata_file.exists():
        print("No champion in artifacts/. Run: python -m loan_default.registry export", file=sys.stderr)
        return 2
    champion = joblib.load(model_file)
    champion_meta = json.loads(metadata_file.read_text())
    # Same LightGBM settings as the champion, so the constraints are the only change.
    params = champion_meta["gbm_params"]

    table = pd.read_parquet(PROCESSED_FILE)
    X_fit, y_fit = split_xy(between(table, FIT_PERIOD))
    X_cal, y_cal = split_xy(between(table, CALIBRATION_PERIOD))
    X_train, _ = split_xy(between(table, TRAIN_PERIOD))
    X_test, y_test = split_xy(between(table, TEST_PERIOD))
    print(f"Champion: version {champion_meta['model_version']}. Challengers: {', '.join(args.challengers)}, same windows.")
    print(f"fit {len(X_fit):,}  calibrate {len(X_cal):,}  test {len(X_test):,}\n")

    # Walk-forward validation inside the train period: the test period stays untouched.
    builders = {"champion": lambda: build_gbm(**params)}
    builders.update({name: (lambda build=CHALLENGERS[name]: build(params)) for name in args.challengers})
    folds = walk_forward(table, params, builders=builders)
    print("Walk-forward validation (inside the train period):")
    print(folds.round(4).to_string(index=False), "\n")

    mean_auc = folds[folds["model"] != "champion"].groupby("model")["auc"].mean().sort_values(ascending=False)
    print("Mean walk-forward AUC of the challengers:")
    print(mean_auc.round(4).to_string(), "\n")
    challenger_name = mean_auc.index[0]
    print(f"Best in validation: {challenger_name}. Only this one is scored on the test period.\n")

    challenger = fit_calibrated(CHALLENGERS[challenger_name](params), X_fit, y_fit, X_cal, y_cal)
    results = {"champion": evaluate(champion, X_test, y_test), challenger_name: evaluate(challenger, X_test, y_test)}
    print(f"Test period {TEST_PERIOD[0]} to {TEST_PERIOD[1]}, scored once:")
    print(pd.DataFrame(results).T.round(4).to_string(), "\n")

    checks = gate(results["champion"], results[challenger_name])
    for check in checks:
        print(f"[{'pass' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    passed = all(check.passed for check in checks)
    print(f"\nGate: {'PASSED' if passed else 'FAILED'}")

    if args.register or args.promote:
        pd_train, pd_test = predict_pd(challenger, X_train), predict_pd(challenger, X_test)
        thresholds = fit_thresholds(pd_train)
        bands = band_table(y_test, pd_test, thresholds)
        score_psi, features_psi = psi(pd_train, pd_test), feature_psi(X_train, X_test)
        metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "lightgbm_version": lightgbm.__version__,
            "features": MODEL_FEATURES,
            "band_labels": BAND_LABELS,
            "band_thresholds": thresholds,
            "periods": {"fit": FIT_PERIOD, "calibration": CALIBRATION_PERIOD, "test": TEST_PERIOD},
            "gbm_params": params,
            "challenger": challenger_name,
            "challengers_compared": args.challengers,
            "walk_forward": folds.to_dict(orient="records"),
            "test_metrics": {"gbm": results[challenger_name], "champion_at_the_time": results["champion"]},
            "gate": [check.__dict__ for check in checks],
            "test_bands": bands.to_dict(orient="index"),
            "psi_train_test": {"pd": score_psi, "features": features_psi.to_dict()},
        }
        run_params = {**params, "challenger": challenger_name, "gate_passed": passed, "n_features": len(MODEL_FEATURES)}
        metrics = {f"test_{k}_gbm": v for k, v in results[challenger_name].items() if k in ("auc", "gini", "brier")}
        metrics["test_calibration_gap"] = results[challenger_name]["calibration_gap"]
        note = f"Challenger {challenger_name} against champion v{champion_meta['model_version']}: gate {'passed' if passed else 'failed'}"
        version = log_run(
            challenger, metadata, run_params, metrics, input_example=X_test.head(5), name=f"{challenger_name}", note=note
        )
        print(f"Registered as version {version}.")
        if args.promote and passed:
            promote(version)
            print(f"Version {version} is now the champion. Next: python -m loan_default.registry export")
        elif args.promote:
            print("Not promoted: the gate failed.")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
