import numpy as np
import pandas as pd
import pytest

from loan_default.features import NUMERIC_FEATURES
from loan_default.model import MONOTONE_DIRECTIONS, build_gbm_monotone
from loan_default.retrain import MAX_CALIBRATION_GAP, gate
from loan_default.synthetic import make_applications
from tests.test_api import training_style_frame

CHAMPION = {"auc": 0.68, "brier": 0.12, "calibration_gap": -0.007}


def challenger(**changes):
    return {**CHAMPION, **changes}


def failed(checks):
    return [c.name for c in checks if not c.passed]


def test_gate_passes_a_challenger_that_matches_or_beats_the_champion():
    assert failed(gate(CHAMPION, CHAMPION)) == []
    assert failed(gate(CHAMPION, challenger(auc=0.69, brier=0.119))) == []


def test_gate_fails_on_lower_auc():
    [name] = failed(gate(CHAMPION, challenger(auc=0.679)))
    assert name.startswith("AUC")


def test_gate_fails_on_worse_brier():
    [name] = failed(gate(CHAMPION, challenger(brier=0.1201)))
    assert name.startswith("Brier")


@pytest.mark.parametrize("gap", [MAX_CALIBRATION_GAP + 0.001, -MAX_CALIBRATION_GAP - 0.001])
def test_gate_fails_when_calibration_is_off_in_either_direction(gap):
    [name] = failed(gate(CHAMPION, challenger(calibration_gap=gap)))
    assert name.startswith("Mean PD")


def test_monotone_model_never_moves_the_wrong_way():
    """Train on a target that is NOT monotone in dti; the constrained model must still be."""
    X = training_style_frame(make_applications(3000, seed=1))
    rng = np.random.default_rng(2)
    y = pd.Series((np.sin(X["dti"] / 3) + rng.normal(0, 0.5, len(X)) > 0).astype(int))
    model = build_gbm_monotone(n_estimators=50, min_child_samples=20, num_leaves=7).fit(X, y)

    base = X.head(100).reset_index(drop=True)
    for feature, direction in [("dti", 1), ("fico", -1)]:
        grid = np.linspace(X[feature].min(), X[feature].max(), 15)
        scores = np.column_stack(
            [model.predict_proba(base.assign(**{feature: value}))[:, 1] for value in grid]
        )
        assert (np.diff(scores, axis=1) * direction >= -1e-12).all(), feature


def test_constraints_line_up_with_the_columns():
    X = training_style_frame(make_applications(300, seed=3))
    y = pd.Series(np.random.default_rng(4).integers(0, 2, len(X)))
    model = build_gbm_monotone(n_estimators=5, min_child_samples=5).fit(X, y)
    constraints = model.named_steps["model"].get_params()["monotone_constraints"]
    width = model.named_steps["preprocess"].transform(X.head(2)).shape[1]
    assert len(constraints) == width
    assert constraints[-len(NUMERIC_FEATURES):] == [MONOTONE_DIRECTIONS.get(f, 0) for f in NUMERIC_FEATURES]


@pytest.mark.parametrize("name", ["gbm-monotone", "linear-svm", "naive-bayes"])
def test_every_challenger_fits_calibrates_and_gives_probabilities(name):
    from loan_default.model import fit_calibrated, predict_pd
    from loan_default.retrain import CHALLENGERS

    X = training_style_frame(make_applications(1200, seed=5))
    y = pd.Series((X["dti"] + np.random.default_rng(6).normal(0, 8, len(X)) > 22).astype(int))
    model = fit_calibrated(CHALLENGERS[name]({"n_estimators": 10, "min_child_samples": 10}), X[:700], y[:700], X[700:], y[700:])
    probabilities = predict_pd(model, X[700:])
    assert probabilities.shape == (500,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert probabilities.std() > 0  # not a constant
