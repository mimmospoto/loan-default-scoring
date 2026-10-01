"""Which raw Lending Club columns the model may use, and which it may not.

Features are an allowlist: a column is used only if it is named in
RAW_FEATURES. The exclusion groups below exist to document *why* the
dangerous columns are out; `python -m loan_default.data.profile` checks
both lists against the real file header.
"""

from __future__ import annotations

# Application-time inputs, kept deliberately compact so the API contract
# stays small enough to type and validate by hand.
RAW_FEATURES = [
    "loan_amnt",
    "purpose",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "dti",
    "fico_range_low",
    "earliest_cr_line",  # becomes credit history length at application
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

# Used to build the cohort, the split and the target; never model inputs.
FRAMING_COLUMNS = ["issue_d", "term", "loan_status"]

IDENTIFIERS = ["id", "member_id", "url"]

# Only known after the loan was issued: using any of these is target leakage.
POST_ORIGINATION = [
    "funded_amnt",
    "funded_amnt_inv",
    "pymnt_plan",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "deferral_term",
    "payment_plan_start_date",
    "orig_projected_additional_accrued_interest",
]
POST_ORIGINATION_PREFIXES = ("hardship", "settlement")

# Lending Club's own risk score and the price derived from it. Known at
# origination, but the service is what a lender would use to *set* these.
LENDER_PRICING = ["grade", "sub_grade", "int_rate", "installment"]

# Precise geography proxies protected attributes.
GEOGRAPHY = ["zip_code", "addr_state"]

FREE_TEXT = ["emp_title", "title", "desc"]

# How Lending Club processed the loan, not a borrower attribute.
LENDER_PROCESS = ["initial_list_status", "policy_code", "disbursement_method"]

EXCLUSION_GROUPS = {
    "identifier": IDENTIFIERS,
    "post-origination (leakage)": POST_ORIGINATION,
    "lender pricing": LENDER_PRICING,
    "geography": GEOGRAPHY,
    "free text": FREE_TEXT,
    "lender process": LENDER_PROCESS,
}


def classify(column: str) -> str:
    """Return the role of a raw column: feature, framing, an exclusion group, or unused."""
    if column in RAW_FEATURES:
        return "feature"
    if column in FRAMING_COLUMNS:
        return "framing"
    if column.startswith(POST_ORIGINATION_PREFIXES):
        return "post-origination (leakage)"
    for group, names in EXCLUSION_GROUPS.items():
        if column in names:
            return group
    return "unused"
