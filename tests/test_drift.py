import json

import numpy as np
import pandas as pd
import pytest

from loan_default.drift import compare, load_request_log, status, summary
from loan_default.features import MODEL_FEATURES
from loan_default.synthetic import make_applications
from tests.test_api import training_style_frame


def scored_frame(n: int, seed: int) -> pd.DataFrame:
    """Applications with a PD and band attached, as the drift report expects them."""
    frame = training_style_frame(make_applications(n, seed=seed))
    rng = np.random.default_rng(seed)
    frame["pd"] = rng.uniform(0.02, 0.4, n)
    frame["risk_band"] = pd.cut(frame["pd"], [0, 0.07, 0.10, 0.14, 0.20, 1], labels=list("ABCDE")).astype(str)
    return frame


@pytest.mark.parametrize(
    "value, expected", [(0.0, "stable"), (0.099, "stable"), (0.10, "warning"), (0.249, "warning"), (0.25, "drift")]
)
def test_status_thresholds(value, expected):
    assert status(value) == expected


@pytest.fixture(scope="module")
def reference():
    return scored_frame(4000, seed=1)


def test_same_distribution_is_stable(reference):
    result = compare(reference, scored_frame(2000, seed=2))
    assert result.verdict == "stable"
    assert result.psi.max() < 0.1


def test_shifted_feature_is_flagged_as_drift(reference):
    shifted = scored_frame(2000, seed=3)
    shifted["fico"] = shifted["fico"] - 60  # every applicant 60 points weaker
    result = compare(reference, shifted)
    assert result.verdict == "drift"
    assert result.psi.index[0] == "fico"
    assert result.psi["fico"] > 0.25
    assert result.psi.drop("fico").max() < 0.1  # nothing else is blamed


def test_shifted_scores_are_flagged(reference):
    shifted = scored_frame(2000, seed=4)
    shifted["pd"] = shifted["pd"] * 0.5
    result = compare(reference, shifted)
    assert result.psi.index[0] == "pd"
    assert result.verdict == "drift"


def test_summary_names_the_verdict_and_the_worst_column(reference):
    shifted = scored_frame(2000, seed=3)
    shifted["fico"] = shifted["fico"] - 60
    text = summary(compare(reference, shifted), "training", "last week")
    assert "**Verdict: drift**" in text
    assert "| fico |" in text and "| drift |" in text
    assert "## Share missing" in text


def test_load_request_log_reads_score_lines_and_skips_the_rest(tmp_path):
    application = make_applications(1, seed=5)[0]
    record = {
        "event": "score",
        "request_id": "r1",
        "input": {**{name: None for name in MODEL_FEATURES}, **application},
        "pd": 0.12,
        "risk_band": "C",
    }
    log = tmp_path / "requests.jsonl"
    log.write_text(
        "INFO:     Started server process [1]\n" + json.dumps(record) + "\n" + json.dumps({"event": "other"}) + "\n"
    )
    frame = load_request_log(log)
    assert frame.columns.tolist() == MODEL_FEATURES + ["pd", "risk_band"]
    assert len(frame) == 1 and frame.loc[0, "pd"] == 0.12


def test_load_request_log_without_score_lines_is_an_error(tmp_path):
    log = tmp_path / "requests.jsonl"
    log.write_text("INFO: nothing here\n")
    with pytest.raises(ValueError, match="No score lines"):
        load_request_log(log)
