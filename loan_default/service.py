"""Scoring from the exported artifacts. No MLflow needed.

Turns validated applications into the frame the model was trained on, then
reuses the training code (predict_pd, assign_band) so the numbers match.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from loan_default.bands import assign_band
from loan_default.features import MODEL_FEATURES, NUMERIC_FEATURES
from loan_default.model import predict_pd
from loan_default.schemas import LoanApplication


class Scorer:
    def __init__(self, model, metadata: dict):
        if metadata["features"] != MODEL_FEATURES:
            raise ValueError("The model was trained on different features than this code expects.")
        self.model = model
        self.thresholds = metadata["band_thresholds"]
        self.version = str(metadata["model_version"])

    @classmethod
    def load(cls, directory: Path) -> Scorer:
        model_file, metadata_file = directory / "model.joblib", directory / "metadata.json"
        if not model_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(
                f"No exported model in {directory}. Run: python -m loan_default.registry export"
            )
        return cls(joblib.load(model_file), json.loads(metadata_file.read_text()))

    def score(self, applications: list[LoanApplication]) -> list[tuple[float, str]]:
        """(pd, risk band) for each application."""
        frame = pd.DataFrame([a.model_dump() for a in applications], columns=MODEL_FEATURES)
        # Training data has float columns with NaN for missing; match that exactly.
        frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype("float64")
        probabilities = predict_pd(self.model, frame)
        bands = assign_band(probabilities, self.thresholds)
        return [(float(p), str(b)) for p, b in zip(probabilities, bands)]
