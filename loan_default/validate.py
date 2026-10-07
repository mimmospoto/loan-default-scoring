"""Walk-forward validation: repeat fit, calibrate, validate on earlier periods.

Shows how stable ranking and calibration are from one period to the next,
without touching the test period.
"""

from __future__ import annotations

import pandas as pd

from loan_default.config import WALK_FORWARD_FOLDS
from loan_default.metrics import discrimination
from loan_default.model import build_baseline, build_gbm, fit_calibrated, predict_pd
from loan_default.splits import between, split_xy


def walk_forward(table: pd.DataFrame, gbm_params: dict, folds: list[dict] = WALK_FORWARD_FOLDS) -> pd.DataFrame:
    """One row per fold and model, scored on that fold's validation period."""
    builders = {"baseline": build_baseline, "gbm": lambda: build_gbm(**gbm_params)}
    rows = []
    for fold in folds:
        X_fit, y_fit = split_xy(between(table, fold["fit"]))
        X_cal, y_cal = split_xy(between(table, fold["calibration"]))
        X_val, y_val = split_xy(between(table, fold["validation"]))
        for name, build in builders.items():
            model = fit_calibrated(build(), X_fit, y_fit, X_cal, y_cal)
            pd_val = predict_pd(model, X_val)
            rows.append(
                {
                    "validation": f"{fold['validation'][0]} to {fold['validation'][1]}",
                    "model": name,
                    "fit_loans": len(X_fit),
                    **discrimination(y_val, pd_val),
                    "predicted": float(pd_val.mean()),
                    "observed": float(y_val.mean()),
                }
            )
    return pd.DataFrame(rows)
