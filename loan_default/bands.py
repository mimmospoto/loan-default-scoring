"""Risk bands A-E: fixed PD thresholds learned once from the training period."""

from __future__ import annotations

import numpy as np

from loan_default.config import BAND_LABELS


def fit_thresholds(pd_train: np.ndarray) -> list[float]:
    """Cut points that split training PDs into equally sized bands."""
    quantiles = np.linspace(0, 1, len(BAND_LABELS) + 1)[1:-1]
    thresholds = [round(float(t), 4) for t in np.quantile(pd_train, quantiles)]
    if sorted(set(thresholds)) != thresholds:
        raise ValueError(f"Band thresholds are not strictly increasing: {thresholds}")
    return thresholds


def assign_band(pd_values: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Band label per PD. A PD equal to a threshold falls in the riskier band."""
    index = np.searchsorted(thresholds, pd_values, side="right")
    return np.asarray(BAND_LABELS)[index]
