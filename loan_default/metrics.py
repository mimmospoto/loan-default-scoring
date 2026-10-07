"""Discrimination, calibration and stability metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from loan_default.bands import assign_band
from loan_default.config import BAND_LABELS

PSI_FLOOR = 1e-6  # avoids log(0) when a bin is empty on one side


def discrimination(y_true, pd_values) -> dict[str, float]:
    auc = float(roc_auc_score(y_true, pd_values))
    return {"auc": auc, "gini": 2 * auc - 1, "brier": float(brier_score_loss(y_true, pd_values))}


def band_table(y_true, pd_values, thresholds: list[float]) -> pd.DataFrame:
    """Per band: share of loans, mean predicted PD, observed default rate."""
    frame = pd.DataFrame(
        {"band": assign_band(pd_values, thresholds), "pd": pd_values, "default": np.asarray(y_true)}
    )
    grouped = frame.groupby("band")
    table = pd.DataFrame(
        {
            "loans": grouped.size(),
            "share": grouped.size() / len(frame),
            "predicted": grouped["pd"].mean(),
            "observed": grouped["default"].mean(),
        }
    )
    return table.reindex(BAND_LABELS)


def _shares(values: pd.Series, edges: np.ndarray | None) -> pd.Series:
    """Share of rows per bin; missing values form their own bin."""
    if edges is None:
        bins = values.astype("object")
    else:
        bins = pd.Series(np.searchsorted(edges, values, side="right"), index=values.index).astype("object")
    bins = bins.where(values.notna(), "missing")
    return bins.value_counts(normalize=True)


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index of `actual` against `expected`.

    Numeric values are binned on the deciles of `expected`; other values are
    compared by category. Rule of thumb: < 0.1 stable, > 0.25 major shift.
    """
    expected, actual = pd.Series(expected), pd.Series(actual)
    edges = None
    if pd.api.types.is_numeric_dtype(expected):
        inner = np.linspace(0, 1, bins + 1)[1:-1]
        edges = np.unique(np.quantile(expected.dropna(), inner))
    e, a = _shares(expected, edges), _shares(actual, edges)
    e, a = e.align(a, fill_value=0.0)
    e, a = e.clip(lower=PSI_FLOOR), a.clip(lower=PSI_FLOOR)
    return float(((a - e) * np.log(a / e)).sum())


def feature_psi(expected: pd.DataFrame, actual: pd.DataFrame) -> pd.Series:
    """PSI per column, largest first."""
    values = {column: psi(expected[column], actual[column]) for column in expected.columns}
    return pd.Series(values).sort_values(ascending=False)
