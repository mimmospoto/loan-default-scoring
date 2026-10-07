"""Select loans by issue period and split them into features and target."""

from __future__ import annotations

import pandas as pd

from loan_default.features import MODEL_FEATURES


def between(table: pd.DataFrame, period: tuple[str, str]) -> pd.DataFrame:
    """Labelled loans issued within the period (inclusive)."""
    start, end = period
    in_period = (table["issue_month"] >= start) & (table["issue_month"] <= end)
    return table[in_period & table["default"].notna()]


def split_xy(table: pd.DataFrame):
    return table[MODEL_FEATURES], table["default"].astype(int)
