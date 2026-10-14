import json
from typing import get_args

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from loan_default.api import create_app
from loan_default.bands import assign_band
from loan_default.features import MODEL_FEATURES, NUMERIC_FEATURES
from loan_default.model import build_gbm, fit_calibrated, predict_pd
from loan_default.schemas import MAX_BATCH, HomeOwnership, Purpose, VerificationStatus

THRESHOLDS = [0.1, 0.2, 0.3, 0.4]
OPTIONAL = ["emp_length_years", "mths_since_last_delinq", "revol_util"]


def make_applications(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    applications = []
    for _ in range(n):
        total = int(rng.integers(2, 60))
        application = {
            "purpose": str(rng.choice(get_args(Purpose))),
            "home_ownership": str(rng.choice(get_args(HomeOwnership))),
            "verification_status": str(rng.choice(get_args(VerificationStatus))),
            "loan_amnt": float(rng.integers(1000, 35000)),
            "annual_inc": float(rng.integers(15000, 200000)),
            "dti": float(rng.uniform(0, 40)),
            "fico": float(rng.integers(660, 840)),
            "credit_history_months": float(rng.integers(36, 500)),
            "delinq_2yrs": float(rng.integers(0, 4)),
            "inq_last_6mths": float(rng.integers(0, 5)),
            "open_acc": float(rng.integers(1, total + 1)),
            "total_acc": float(total),
            "pub_rec": float(rng.integers(0, 3)),
            "pub_rec_bankruptcies": float(rng.integers(0, 2)),
            "revol_bal": float(rng.integers(0, 60000)),
            "mort_acc": float(rng.integers(0, 5)),
            "acc_open_past_24mths": float(rng.integers(0, 10)),
        }
        # Leave each optional field out about a third of the time.
        for name, value in {
            "emp_length_years": float(rng.integers(0, 11)),
            "mths_since_last_delinq": float(rng.integers(0, 120)),
            "revol_util": float(rng.uniform(0, 100)),
        }.items():
            if rng.random() > 0.33:
                application[name] = value
        applications.append(application)
    return applications


def training_style_frame(applications: list[dict]) -> pd.DataFrame:
    """The same applications laid out the way the training table is: float columns, NaN for missing."""
    frame = pd.DataFrame(applications).reindex(columns=MODEL_FEATURES)
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype("float64")
    return frame


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A small model exported the way `registry export` does it."""
    X = training_style_frame(make_applications(600, seed=1))
    y = pd.Series((X["dti"] + np.random.default_rng(2).normal(0, 8, len(X)) > 22).astype(int))
    model = fit_calibrated(build_gbm(n_estimators=20, min_child_samples=5), X[:400], y[:400], X[400:], y[400:])
    directory = tmp_path_factory.mktemp("artifacts")
    joblib.dump(model, directory / "model.joblib")
    metadata = {"features": MODEL_FEATURES, "band_thresholds": THRESHOLDS, "model_version": 7}
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return model, directory


@pytest.fixture(scope="module")
def client(trained):
    with TestClient(create_app(trained[1])) as test_client:
        yield test_client


def test_api_predicts_exactly_like_the_training_code(trained, client):
    model = trained[0]
    applications = make_applications(50, seed=3)

    expected = predict_pd(model, training_style_frame(applications))
    response = client.post("/predict/batch", json={"applications": applications})

    assert response.status_code == 200
    body = response.json()
    np.testing.assert_array_equal([p["pd"] for p in body["predictions"]], expected)
    assert [p["risk_band"] for p in body["predictions"]] == list(assign_band(expected, THRESHOLDS))


def test_single_prediction_matches_the_batch(client):
    applications = make_applications(5, seed=4)
    batch = client.post("/predict/batch", json={"applications": applications}).json()["predictions"]

    for application, expected in zip(applications, batch):
        body = client.post("/predict", json=application).json()
        assert (body["pd"], body["risk_band"]) == (expected["pd"], expected["risk_band"])


def test_response_carries_version_and_a_fresh_request_id(client):
    application = make_applications(1, seed=5)[0]
    first = client.post("/predict", json=application).json()
    second = client.post("/predict", json=application).json()

    assert first["model_version"] == "7"
    assert 0 < first["pd"] < 1
    assert first["risk_band"] in "ABCDE"
    assert first["request_id"] != second["request_id"]


def test_optional_fields_can_be_left_out(client):
    application = make_applications(1, seed=6)[0]
    for name in OPTIONAL:
        application.pop(name, None)

    assert client.post("/predict", json=application).status_code == 200


def test_health_reports_the_model_version(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "7"}


@pytest.mark.parametrize(
    "change",
    [
        {"fico": 900},  # above the FICO scale
        {"dti": -1},
        {"loan_amnt": "a lot"},
        {"purpose": "holiday"},  # not a known purpose
        {"home_ownership": "rent"},  # case matters
        {"open_acc": 80, "total_acc": 10},  # open lines cannot exceed all lines
        {"fico_score": 700},  # misspelled or unknown field
    ],
)
def test_invalid_applications_are_rejected(client, change):
    application = {**make_applications(1, seed=7)[0], **change}

    assert client.post("/predict", json=application).status_code == 422


def test_missing_required_field_is_rejected_and_named(client):
    application = make_applications(1, seed=8)[0]
    del application["fico"]

    response = client.post("/predict", json=application)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "fico"]


def test_one_bad_application_rejects_the_whole_batch(client):
    good, bad = make_applications(2, seed=9)
    bad["fico"] = 900

    response = client.post("/predict/batch", json={"applications": [good, bad]})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][:3] == ["body", "applications", 1]


def test_batch_size_limits(client):
    application = make_applications(1, seed=10)[0]

    assert client.post("/predict/batch", json={"applications": []}).status_code == 422
    too_many = {"applications": [application] * (MAX_BATCH + 1)}
    assert client.post("/predict/batch", json=too_many).status_code == 422


def test_start_up_fails_without_an_exported_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="registry export"):
        with TestClient(create_app(tmp_path)):
            pass


@pytest.fixture(scope="module")
def secured_client(trained):
    with TestClient(create_app(trained[1], api_key="s3cret")) as test_client:
        yield test_client


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}])
def test_scoring_needs_the_right_key_when_one_is_set(secured_client, headers):
    application = make_applications(1, seed=12)[0]

    single = secured_client.post("/predict", json=application, headers=headers)
    batch = secured_client.post("/predict/batch", json={"applications": [application]}, headers=headers)

    assert single.status_code == 401
    assert batch.status_code == 401


def test_the_right_key_is_accepted(secured_client):
    application = make_applications(1, seed=13)[0]
    headers = {"X-API-Key": "s3cret"}

    assert secured_client.post("/predict", json=application, headers=headers).status_code == 200
    batch = {"applications": [application]}
    assert secured_client.post("/predict/batch", json=batch, headers=headers).status_code == 200


def test_health_stays_open_when_a_key_is_set(secured_client):
    assert secured_client.get("/health").status_code == 200


def test_the_key_is_checked_before_the_input(secured_client):
    # A caller without the key learns nothing about the input rules.
    assert secured_client.post("/predict", json={}).status_code == 401
