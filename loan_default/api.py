"""The scoring service.

Run: python -m uvicorn loan_default.api:app
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from loan_default.config import ARTIFACTS_DIR
from loan_default.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    LoanApplication,
    Prediction,
    PredictResponse,
)
from loan_default.service import Scorer


def create_app(artifacts_dir: Path | None = None, api_key: str | None = None) -> FastAPI:
    directory = artifacts_dir or Path(os.environ.get("ARTIFACTS_DIR", ARTIFACTS_DIR))
    # With no key configured, scoring is open (local use). Deployments set API_KEY.
    expected_key = api_key or os.environ.get("API_KEY")
    key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def require_key(supplied: str | None = Security(key_header)) -> None:
        if expected_key and not (supplied and secrets.compare_digest(supplied, expected_key)):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fails at start-up, not on the first request, if there is no model.
        app.state.scorer = Scorer.load(directory)
        yield

    app = FastAPI(
        title="Loan default scoring",
        description="Calibrated probability of default and a risk band (A-E) for a loan application.",
        lifespan=lifespan,
    )

    # Plain `def` endpoints: FastAPI runs them in a thread pool, so scoring does not block the server.
    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        return HealthResponse(status="ok", model_version=request.app.state.scorer.version)

    @app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_key)])
    def predict(application: LoanApplication, request: Request):
        scorer: Scorer = request.app.state.scorer
        [(probability, band)] = scorer.score([application])
        return PredictResponse(
            pd=probability, risk_band=band, model_version=scorer.version, request_id=str(uuid.uuid4())
        )

    @app.post("/predict/batch", response_model=BatchResponse, dependencies=[Depends(require_key)])
    def predict_batch(batch: BatchRequest, request: Request):
        scorer: Scorer = request.app.state.scorer
        scores = scorer.score(batch.applications)
        return BatchResponse(
            predictions=[Prediction(pd=p, risk_band=b) for p, b in scores],
            model_version=scorer.version,
            request_id=str(uuid.uuid4()),
        )

    return app


app = create_app()
