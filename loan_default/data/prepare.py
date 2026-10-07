"""Build the modelling table from the raw file.

Keeps 36-month loans in the train, test and drift periods, derives the model
features and the target, and saves one parquet file.
Run: python -m loan_default.data.prepare
"""

from __future__ import annotations

import sys

import pandas as pd

from loan_default.columns import FRAMING_COLUMNS, RAW_FEATURES
from loan_default.config import (
    COHORT_TERM_MONTHS,
    PERIODS,
    PROCESSED_FILE,
    TARGET_MAP,
    find_accepted_file,
)
from loan_default.features import MONTH_FORMAT, build_features


def assign_period(issue_month: pd.Series) -> pd.Series:
    """Label each 'YYYY-MM' issue month as train, test, drift, or missing."""
    period = pd.Series(pd.NA, index=issue_month.index, dtype="string")
    for name, (start, end) in PERIODS.items():
        period = period.mask((issue_month >= start) & (issue_month <= end), name)
    return period


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    # The file ends with a few summary rows that have no issue date.
    raw = raw.dropna(subset=["issue_d"])
    term_months = raw["term"].str.extract(r"(\d+)", expand=False).astype(int)
    raw = raw[term_months == COHORT_TERM_MONTHS]

    issue_month = pd.to_datetime(raw["issue_d"], format=MONTH_FORMAT).dt.strftime("%Y-%m")
    period = assign_period(issue_month)
    raw = raw[period.notna()]

    table = build_features(raw)
    table["issue_month"] = issue_month[raw.index]
    table["period"] = period[raw.index]
    # Missing for loans without a final status; only expected in the drift period.
    table["default"] = raw["loan_status"].map(TARGET_MAP).astype("Int8")
    return table.reset_index(drop=True)


def main() -> int:
    path = find_accepted_file()
    if path is None:
        print("No data found. Run: python -m loan_default.data.download", file=sys.stderr)
        return 1
    print(f"Reading {path.name}...")
    raw = pd.read_csv(path, usecols=RAW_FEATURES + FRAMING_COLUMNS)
    table = prepare(raw)

    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(PROCESSED_FILE, index=False)
    summary = table.groupby("period").agg(
        loans=("default", "size"),
        labelled=("default", "count"),
        default_rate=("default", "mean"),
    )
    print(summary.round(4).to_string())
    print(f"Saved {len(table):,} rows to {PROCESSED_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
