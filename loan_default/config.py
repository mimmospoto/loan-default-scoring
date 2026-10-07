"""Framing decisions as code: paths, cohort, target and split dates.

docs/framing.md explains the reasoning; this module is the source of truth.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "loans.parquet"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

KAGGLE_DATASET = "wordsforthewise/lending-club"
ACCEPTED_GLOB = "accepted_2007_to_2018*.csv*"

# Cohort: 36-month loans only, so every loan in train/test has had time to
# reach a terminal status by the snapshot date (removes most maturity bias).
COHORT_TERM_MONTHS = 36

# Months between the last scheduled payment and a charge-off being booked.
CHARGE_OFF_LAG_MONTHS = 5

# Out-of-time split on issue month (inclusive bounds, "YYYY-MM").
TRAIN_PERIOD = ("2012-05", "2014-12")
TEST_PERIOD = ("2015-01", "2015-10")
# Later vintages are not fully matured: features only, used for drift (PSI).
DRIFT_PERIOD = ("2016-01", "2017-12")
PERIODS = {"train": TRAIN_PERIOD, "test": TEST_PERIOD, "drift": DRIFT_PERIOD}

# The train period is split in time: the model is fitted on the first part
# and its probabilities are calibrated on the last six months.
FIT_PERIOD = ("2012-05", "2014-06")
CALIBRATION_PERIOD = ("2014-07", "2014-12")

# Walk-forward validation inside the train period. Each fold repeats the
# final procedure on earlier data: fit, calibrate on the next six months,
# validate on the six months after that.
WALK_FORWARD_FOLDS = [
    {"fit": ("2012-05", "2012-12"), "calibration": ("2013-01", "2013-06"), "validation": ("2013-07", "2013-12")},
    {"fit": ("2012-05", "2013-06"), "calibration": ("2013-07", "2013-12"), "validation": ("2014-01", "2014-06")},
    {"fit": ("2012-05", "2013-12"), "calibration": ("2014-01", "2014-06"), "validation": ("2014-07", "2014-12")},
]

# Risk bands, lowest risk first. Thresholds are quantiles of training PDs.
BAND_LABELS = ["A", "B", "C", "D", "E"]

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
