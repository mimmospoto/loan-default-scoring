import pandas as pd

from loan_default.data.prepare import prepare
from loan_default.features import MODEL_FEATURES, build_features, parse_emp_length


def raw_loans(**overrides) -> pd.DataFrame:
    row = {
        "loan_amnt": 10000.0,
        "term": " 36 months",
        "purpose": "credit_card",
        "emp_length": "10+ years",
        "home_ownership": "RENT",
        "annual_inc": 60000.0,
        "verification_status": "Verified",
        "dti": 15.0,
        "fico_range_low": 700.0,
        "earliest_cr_line": "Jan-2003",
        "delinq_2yrs": 0.0,
        "inq_last_6mths": 1.0,
        "mths_since_last_delinq": None,
        "open_acc": 8.0,
        "total_acc": 20.0,
        "pub_rec": 0.0,
        "pub_rec_bankruptcies": 0.0,
        "revol_bal": 5000.0,
        "revol_util": 40.0,
        "mort_acc": 0.0,
        "acc_open_past_24mths": 3.0,
        "issue_d": "Mar-2013",
        "loan_status": "Fully Paid",
    }
    rows = [{**row, **change} for change in overrides.get("rows", [{}])]
    return pd.DataFrame(rows)


def test_parse_emp_length():
    raw = pd.Series(["10+ years", "< 1 year", "1 year", "3 years", None])
    assert parse_emp_length(raw).tolist()[:4] == [10.0, 0.0, 1.0, 3.0]
    assert parse_emp_length(raw).isna().tolist() == [False, False, False, False, True]


def test_build_features_columns_and_derived_values():
    features = build_features(raw_loans(rows=[{}, {"home_ownership": "ANY"}]))
    assert features.columns.tolist() == MODEL_FEATURES
    assert features["credit_history_months"].tolist() == [122.0, 122.0]
    assert features["fico"].tolist() == [700.0, 700.0]
    assert features["home_ownership"].tolist() == ["RENT", "OTHER"]


def test_prepare_filters_cohort_and_sets_target():
    raw = raw_loans(
        rows=[
            {"loan_status": "Charged Off"},
            {"term": " 60 months"},
            {"issue_d": "Jun-2010"},
            {"issue_d": "Feb-2015"},
            {"issue_d": "Jun-2017", "loan_status": "Current"},
            {"issue_d": None},
        ]
    )
    table = prepare(raw)
    assert table["period"].tolist() == ["train", "test", "drift"]
    assert table["default"].tolist()[:2] == [1, 0]
    assert table["default"].isna().tolist() == [False, False, True]
