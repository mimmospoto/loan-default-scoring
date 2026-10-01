"""Framing decisions as code: paths, cohort, target and split dates.

docs/framing.md explains the reasoning; this module is the source of truth.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

KAGGLE_DATASET = "wordsforthewise/lending-club"
ACCEPTED_GLOB = "accepted_2007_to_2018*.csv*"

# Cohort: 36-month loans only, so every loan in train/test has had time to
# reach a terminal status by the snapshot date (removes most maturity bias).
COHORT_TERM_MONTHS = 36

# Months between the last scheduled payment and a charge-off being booked.
CHARGE_OFF_LAG_MONTHS = 5

# Out-of-time split on issue month (inclusive bounds, "YYYY-MM").
TRAIN_PERIOD = ("2012-01", "2014-12")
TEST_PERIOD = ("2015-01", "2015-10")
# Later vintages are not fully matured: features only, used for drift (PSI).
DRIFT_PERIOD = ("2016-01", "2017-12")

# loan_status -> default flag. Any status not listed here is excluded.
TARGET_MAP = {
    "Charged Off": 1,
    "Default": 1,
    "Fully Paid": 0,
}


def find_accepted_file() -> Path | None:
    """Locate the accepted-loans file under data/raw, preferring plain CSV."""
    matches = sorted(
        (p for p in RAW_DIR.rglob(ACCEPTED_GLOB) if p.is_file()),
        key=lambda p: (p.suffix == ".gz", str(p)),
    )
    return matches[0] if matches else None
