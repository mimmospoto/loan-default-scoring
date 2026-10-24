"""One JSON line per scored application, written to stdout.

On Azure Container Apps stdout goes to Log Analytics, so this needs no extra
infrastructure. The drift report reads these lines back (loan_default/drift.py).

Only the model inputs and outputs are logged. There is no name, address or
other identifier, and nothing the API does not already receive.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from loan_default.schemas import LoanApplication

LOGGER_NAME = "loan_default.requests"


def configure() -> logging.Logger:
    """Send the request log to stdout as bare JSON lines. Safe to call twice."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # the root logger would print every line twice
    return logger


def log_scores(
    request_id: str,
    model_version: str,
    applications: list[LoanApplication],
    scores: list[tuple[float, str]],
) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for index, (application, (probability, band)) in enumerate(zip(applications, scores)):
        record = {
            "event": "score",
            "timestamp": timestamp,
            "request_id": request_id,
            "index": index,
            "model_version": model_version,
            "input": application.model_dump(),
            "pd": probability,
            "risk_band": band,
        }
        logger.info(json.dumps(record))
