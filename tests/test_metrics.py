import numpy as np
import pandas as pd
import pytest

from loan_default.bands import assign_band, fit_thresholds
from loan_default.metrics import band_table, discrimination, psi


def test_thresholds_split_training_pds_evenly():
    pd_train = np.linspace(0.01, 0.50, 1000)
    thresholds = fit_thresholds(pd_train)
    assert len(thresholds) == 4
    shares = pd.Series(assign_band(pd_train, thresholds)).value_counts(normalize=True)
    assert shares.round(1).tolist() == [0.2] * 5


def test_assign_band_boundaries():
    thresholds = [0.05, 0.10, 0.15, 0.20]
    bands = assign_band(np.array([0.0, 0.05, 0.0999, 0.19, 0.20, 0.9]), thresholds)
    assert bands.tolist() == ["A", "B", "B", "D", "E", "E"]


def test_fit_thresholds_rejects_ties():
    with pytest.raises(ValueError):
        fit_thresholds(np.full(100, 0.1))


def test_discrimination_perfect_ranking():
    result = discrimination([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["auc"] == 1.0
    assert result["gini"] == 1.0


def test_band_table_reports_every_band():
    table = band_table([0, 1], [0.01, 0.9], [0.05, 0.10, 0.15, 0.20])
    assert table.index.tolist() == ["A", "B", "C", "D", "E"]
    assert table.loc["A", "observed"] == 0.0
    assert table.loc["E", "observed"] == 1.0


def test_psi_zero_for_identical_and_positive_for_shifted():
    rng = np.random.default_rng(0)
    base = rng.normal(size=5000)
    assert psi(base, base) == pytest.approx(0.0, abs=1e-9)
    assert psi(base, base + 1.0) > 0.25


def test_psi_handles_categories_and_missing():
    expected = pd.Series(["a"] * 50 + ["b"] * 50)
    assert psi(expected, expected) == pytest.approx(0.0, abs=1e-9)
    assert psi(expected, pd.Series(["a"] * 90 + ["b"] * 10)) > 0.25
    numeric = pd.Series([1.0, 2.0, 3.0, 4.0] * 25)
    assert psi(numeric, numeric.mask(numeric > 3)) > 0.25
