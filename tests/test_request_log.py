import json
import logging

import pytest

from loan_default.request_log import LOGGER_NAME
from tests.test_api import client, make_applications, secured_client, trained  # noqa: F401  (fixtures)


@pytest.fixture
def lines():
    """The request log lines written during a test."""
    captured: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    handler = Collect()
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)
    yield captured
    logger.removeHandler(handler)


def test_predict_logs_one_line_matching_the_response(client, lines):
    application = make_applications(1, seed=1)[0]
    body = client.post("/predict", json=application).json()
    [record] = [json.loads(line) for line in lines]
    assert record["event"] == "score"
    assert record["request_id"] == body["request_id"]
    assert record["model_version"] == body["model_version"]
    assert record["pd"] == body["pd"]
    assert record["risk_band"] == body["risk_band"]
    # Optional fields that were not sent are logged as null, so the report sees them as missing.
    assert record["input"]["loan_amnt"] == application["loan_amnt"]
    assert set(record["input"]) >= set(application)


def test_batch_logs_one_line_per_application_with_a_shared_request_id(client, lines):
    applications = make_applications(5, seed=2)
    body = client.post("/predict/batch", json={"applications": applications}).json()
    records = [json.loads(line) for line in lines]
    assert [r["index"] for r in records] == [0, 1, 2, 3, 4]
    assert {r["request_id"] for r in records} == {body["request_id"]}
    assert [r["pd"] for r in records] == [p["pd"] for p in body["predictions"]]


def test_rejected_requests_are_not_logged(client, secured_client, lines):
    client.post("/predict", json={"loan_amnt": 1})  # fails validation
    secured_client.post("/predict", json=make_applications(1, seed=3)[0])  # no key
    assert lines == []
