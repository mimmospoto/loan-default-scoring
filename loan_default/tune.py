"""Choose LightGBM settings with a time-ordered grid search.

Folds come from TimeSeriesSplit applied to issue months, so each fold trains
on earlier months and is scored on later ones. Only the train period is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from loan_default.model import build_gbm
from loan_default.splits import split_xy

TUNING_SPLITS = 3

# Kept small on purpose: the aim is a defensible choice, not the last decimal.
PARAM_GRID = {
    "model__n_estimators": [100, 200, 400],
    "model__num_leaves": [7, 15, 31],
    "model__min_child_samples": [100, 400],
}


def month_splits(issue_month: pd.Series, n_splits: int = TUNING_SPLITS):
    """Row positions for each fold, with boundaries between whole months."""
    months = np.sort(issue_month.unique())
    for earlier, later in TimeSeriesSplit(n_splits=n_splits).split(months):
        train_rows = np.flatnonzero(issue_month.isin(months[earlier]))
        validation_rows = np.flatnonzero(issue_month.isin(months[later]))
        yield train_rows, validation_rows


def tune_gbm(train: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Best settings by mean AUC across folds, and the full results table."""
    X, y = split_xy(train)
    search = GridSearchCV(
        build_gbm(),
        PARAM_GRID,
        scoring="roc_auc",
        cv=list(month_splits(train["issue_month"])),
        refit=False,
    )
    search.fit(X, y)

    results = pd.DataFrame(search.cv_results_["params"]).rename(columns=lambda c: c.removeprefix("model__"))
    results["mean_auc"] = search.cv_results_["mean_test_score"]
    for fold in range(TUNING_SPLITS):
        results[f"fold_{fold + 1}"] = search.cv_results_[f"split{fold}_test_score"]
    results = results.sort_values("mean_auc", ascending=False).reset_index(drop=True)

    best = {name.removeprefix("model__"): value for name, value in search.best_params_.items()}
    return best, results
