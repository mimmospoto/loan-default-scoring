import pandas as pd

from loan_default.config import CALIBRATION_PERIOD, FIT_PERIOD, TEST_PERIOD, TRAIN_PERIOD, WALK_FORWARD_FOLDS
from loan_default.tune import month_splits


def test_final_split_is_ordered_in_time():
    assert FIT_PERIOD[0] == TRAIN_PERIOD[0]
    assert FIT_PERIOD[1] < CALIBRATION_PERIOD[0]
    assert CALIBRATION_PERIOD[1] == TRAIN_PERIOD[1]
    assert TRAIN_PERIOD[1] < TEST_PERIOD[0]


def test_walk_forward_folds_never_look_ahead():
    for fold in WALK_FORWARD_FOLDS:
        fit, calibration, validation = fold["fit"], fold["calibration"], fold["validation"]
        assert fit[0] <= fit[1] < calibration[0] <= calibration[1] < validation[0] <= validation[1]
        # Folds stay inside the train period, so the test period is never used.
        assert fit[0] >= TRAIN_PERIOD[0]
        assert validation[1] <= TRAIN_PERIOD[1]


def test_tuning_folds_train_on_earlier_months_only():
    months = pd.Series([f"2013-{m:02d}" for m in range(1, 13) for _ in range(5)])
    splits = list(month_splits(months, n_splits=3))
    assert len(splits) == 3
    for train_rows, validation_rows in splits:
        assert months.iloc[train_rows].max() < months.iloc[validation_rows].min()
        # A month is never split across the two sides.
        assert set(months.iloc[train_rows]).isdisjoint(months.iloc[validation_rows])
