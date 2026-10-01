"""Check the framing decisions against the real data.

Prints (1) the snapshot date and which vintages are fully matured,
(2) status mix and default rate per issue year and term, and (3) how every
column in the file is classified. Run: python -m loan_default.data.profile
"""

from __future__ import annotations

import sys
from collections import defaultdict

import pandas as pd

from loan_default.columns import RAW_FEATURES, classify
from loan_default.config import (
    CHARGE_OFF_LAG_MONTHS,
    COHORT_TERM_MONTHS,
    DRIFT_PERIOD,
    TARGET_MAP,
    TEST_PERIOD,
    TRAIN_PERIOD,
    find_accepted_file,
)

MONTH_FORMAT = "%b-%Y"  # e.g. "Dec-2015"


def load_framing_columns(path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["issue_d", "term", "loan_status", "last_pymnt_d"])
    # The file ends with a few summary rows that have no issue date.
    df = df.dropna(subset=["issue_d"])
    df["issue_month"] = pd.to_datetime(df["issue_d"], format=MONTH_FORMAT).dt.to_period("M")
    df["term_months"] = df["term"].str.extract(r"(\d+)", expand=False).astype(int)
    df["terminal"] = df["loan_status"].isin(TARGET_MAP)
    df["default"] = df["loan_status"].map(TARGET_MAP)
    return df


def vintage_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby([df["issue_month"].dt.year.rename("issue_year"), "term_months"])
    return pd.DataFrame(
        {
            "loans": grouped.size(),
            "pct_terminal": (grouped["terminal"].mean() * 100).round(1),
            "default_rate_terminal": (grouped["default"].mean() * 100).round(1),
        }
    )


def period_table(df: pd.DataFrame) -> pd.DataFrame:
    cohort = df[df["term_months"] == COHORT_TERM_MONTHS]
    rows = []
    for name, (start, end) in [("train", TRAIN_PERIOD), ("test", TEST_PERIOD), ("drift", DRIFT_PERIOD)]:
        part = cohort[(cohort["issue_month"] >= start) & (cohort["issue_month"] <= end)]
        rows.append(
            {
                "period": name,
                "from": start,
                "to": end,
                "loans": len(part),
                "pct_terminal": round(part["terminal"].mean() * 100, 1),
                "default_rate_terminal": round(part["default"].mean() * 100, 1),
            }
        )
    return pd.DataFrame(rows).set_index("period")


def main() -> int:
    path = find_accepted_file()
    if path is None:
        print("No data found. Run: python -m loan_default.data.download", file=sys.stderr)
        return 1
    print(f"File: {path}\n")

    header = pd.read_csv(path, nrows=0).columns.tolist()
    df = load_framing_columns(path)

    # The latest payment date in the file is the best proxy for when it was extracted.
    snapshot = pd.to_datetime(df["last_pymnt_d"], format=MONTH_FORMAT).max().to_period("M")
    last_matured = snapshot - COHORT_TERM_MONTHS - CHARGE_OFF_LAG_MONTHS
    print(f"Loans: {len(df):,}   issue months: {df['issue_month'].min()} to {df['issue_month'].max()}")
    print(f"Estimated snapshot month: {snapshot}")
    print(
        f"Last fully matured {COHORT_TERM_MONTHS}-month vintage "
        f"(allowing {CHARGE_OFF_LAG_MONTHS} months for charge-off): {last_matured}"
    )
    if pd.Period(TEST_PERIOD[1], "M") > last_matured:
        print(f"WARNING: test period ends {TEST_PERIOD[1]}, after the last fully matured vintage.")

    print("\nStatus counts:")
    print(df["loan_status"].value_counts().to_string())

    print("\nBy issue year and term:")
    print(vintage_table(df).to_string())

    print(f"\nPlanned split ({COHORT_TERM_MONTHS}-month loans):")
    print(period_table(df).to_string())

    missing = [c for c in RAW_FEATURES if c not in header]
    by_role = defaultdict(list)
    for column in header:
        by_role[classify(column)].append(column)
    print(f"\nColumn classification ({len(header)} columns in file):")
    for role, columns in by_role.items():
        print(f"  {role} ({len(columns)}): {', '.join(columns)}")
    if missing:
        print(f"\nERROR: features not found in file: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
