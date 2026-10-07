"""Model inputs: their names, and how to derive them from raw Lending Club rows.

MODEL_FEATURES is the contract shared by training and serving. The API will
accept exactly these fields. build_features() is training-only: it undoes
Lending Club's file formatting, which a live caller never sends.
"""

from __future__ import annotations

import pandas as pd

CATEGORICAL_FEATURES = ["purpose", "home_ownership", "verification_status"]

NUMERIC_FEATURES = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "emp_length_years",
    "fico",
    "credit_history_months",
    "delinq_2yrs",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "open_acc",
    "total_acc",
    "pub_rec",
    "pub_rec_bankruptcies",
    "revol_bal",
    "revol_util",
    "mort_acc",
    "acc_open_past_24mths",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Raw columns that keep their name and meaning.
_UNCHANGED = [
    "purpose",
    "verification_status",
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "open_acc",
    "total_acc",
    "pub_rec",
    "pub_rec_bankruptcies",
    "revol_bal",
    "revol_util",
    "mort_acc",
    "acc_open_past_24mths",
]

HOME_OWNERSHIP_LEVELS = ["RENT", "MORTGAGE", "OWN"]
MONTH_FORMAT = "%b-%Y"  # e.g. "Dec-2015"


def parse_emp_length(raw: pd.Series) -> pd.Series:
    """'10+ years' -> 10, '< 1 year' -> 0, '3 years' -> 3, missing stays missing."""
    years = raw.str.extract(r"(\d+)", expand=False).astype(float)
    return years.mask(raw.str.startswith("<", na=False), 0.0)


def months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Whole months from start to end, both given as 'Mon-YYYY' strings."""
    start = pd.to_datetime(start, format=MONTH_FORMAT)
    end = pd.to_datetime(end, format=MONTH_FORMAT)
    return ((end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)).astype(float)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw Lending Club rows into the MODEL_FEATURES frame."""
    features = raw[_UNCHANGED].copy()
    # Rare levels (ANY, NONE, OTHER) are merged so the API can offer a fixed list.
    features["home_ownership"] = raw["home_ownership"].where(
        raw["home_ownership"].isin(HOME_OWNERSHIP_LEVELS) | raw["home_ownership"].isna(), "OTHER"
    )
    features["emp_length_years"] = parse_emp_length(raw["emp_length"])
    features["fico"] = raw["fico_range_low"]
    features["credit_history_months"] = months_between(raw["earliest_cr_line"], raw["issue_d"])
    return features[MODEL_FEATURES]
