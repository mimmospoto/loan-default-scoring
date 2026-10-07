# Problem framing

This document fixes what the model predicts, on which loans, and with which
inputs, before any modelling. The same decisions live as code in
[`config.py`](../loan_default/config.py) and
[`columns.py`](../loan_default/columns.py);
`python -m loan_default.data.profile` checks them against the data.

> Status: checked against `accepted_2007_to_2018Q4.csv` (2,260,668 loans,
> issued 2007-06 to 2018-12). One number below was corrected by this check.

## Scenario

A lender calls the service when a loan application arrives and receives a
calibrated probability of default (PD) and a risk band (A–E). The lender uses
them to approve, decline, or price the loan.

One rule follows from this: **point-in-time correctness**. The model may only
use information the lender has at application time.

## Target

| | |
|---|---|
| Unit | One loan application |
| `default = 1` | `loan_status` is `Charged Off` or `Default` |
| `default = 0` | `loan_status` is `Fully Paid` |
| Excluded | Every other status |

Excluded statuses are the non-terminal ones (`Current`, `In Grace Period`,
`Late`), whose outcome is unknown, and the `Does not meet the credit policy`
statuses, which come from loans underwritten under different rules.

## Cohort and maturity bias

Dropping non-terminal loans biases the target. In a recent vintage, the loans
that already have a final status are mostly the ones that defaulted early or
prepaid early; the well-behaved loans still paying on schedule are dropped.
The observed default rate in such a vintage is not the rate the vintage will
end with.

To avoid this, training and testing use only **36-month loans issued early
enough to have fully matured** by the date the dataset was extracted. A loan
counts as matured once its term plus five months (the time for a missed final
payment to be booked as a charge-off) has passed.

60-month loans are out of scope: restricting them to matured vintages would
leave too few years for an out-of-time split.

## Out-of-time split

| Period | Issue months | Loans | Terminal | Default rate | Used for |
|---|---|---|---|---|---|
| Train | 2012-05 to 2014-12 | 297,284 | 100.0% | 13.3% | Fitting, calibration, band thresholds |
| Test | 2015-01 to 2015-10 | 227,565 | 100.0% | 14.8% | All reported model metrics |
| Drift | 2016-01 to 2017-12 | 643,914 | 56.0% | — | Feature drift (PSI) for the monitoring demo |

The split is by issue month, never random, because the service will always
score loans newer than anything it was trained on.

Training starts in 2012-05 for two reasons. Volume in 2007–2011 is thin (603
to 14,101 36-month loans per year, against 43,000+ from 2012 on). And two
features, `mort_acc` and `acc_open_past_24mths`, are empty in the file for
loans issued before 2012-04 and only fully populated from 2012-05.

The test period ends 2015-10, not 2015-12: the file's last `last_pymnt_d` is
2019-03, so the last 36-month vintage fully matured (term + 5-month
charge-off lag) by then is 2015-10. `python -m loan_default.data.profile`
checks this automatically and warns if a future re-download shifts it.

The drift period (2016–2017) is only 56% terminal, confirming the maturity
bias this doc warns about: it is used for PSI only, never for a default
rate.

## Features

Features are an **allowlist** of 20 raw columns. A column is used only if it
is named explicitly, so a leaking column cannot enter by accident.

| Group | Columns |
|---|---|
| Loan request | `loan_amnt`, `purpose` |
| Borrower | `emp_length`, `home_ownership`, `annual_inc`, `verification_status`, `dti` |
| Credit score and history | `fico_range_low`, `earliest_cr_line`, `delinq_2yrs`, `inq_last_6mths`, `mths_since_last_delinq` |
| Accounts | `open_acc`, `total_acc`, `mort_acc`, `acc_open_past_24mths` |
| Public records | `pub_rec`, `pub_rec_bankruptcies` |
| Revolving credit | `revol_bal`, `revol_util` |

Three raw columns are reshaped before modelling: `earliest_cr_line` becomes
`credit_history_months` (months of credit history at the issue date),
`emp_length` becomes `emp_length_years` (0 to 10), and `fico_range_low`
is renamed `fico`. Rare `home_ownership` values are merged into `OTHER`.
The set is kept small on purpose: every feature becomes a typed,
range-checked field in the API contract.

## Exclusions

| Group | Columns | Why |
|---|---|---|
| Post-origination | Payments, outstanding principal, recoveries, `last_*` fields, `funded_amnt*`, all `hardship*` and `settlement*` fields | Known only after the loan exists: target leakage |
| Lender pricing | `grade`, `sub_grade`, `int_rate`, `installment` | Lending Club's own risk score and the price derived from it |
| Geography | `zip_code`, `addr_state` | Proxies for protected attributes |
| Free text | `emp_title`, `title`, `desc` | Unstructured; can encode protected attributes |
| Lender process | `initial_list_status`, `policy_code`, `disbursement_method` | Describe processing, not the borrower |
| Identifiers | `id`, `member_id`, `url` | No predictive meaning |

The full post-origination list is in `columns.py`. All remaining columns are
application-time fields left unused to keep the contract small.

Excluding `grade` and `int_rate` costs accuracy. They are excluded because a
model that reads the lender's existing grade is a wrapper around that grade,
and because the service is meant to be called before a price exists.

## Data file

Kaggle dataset `wordsforthewise/lending-club`, file
`accepted_2007_to_2018Q4.csv` (the Q4 extract — the brief's working title
named Q2, which does not exist in this mirror). The file is never committed;
run `python -m loan_default.data.download` to fetch it.

### License

| Source | What it says |
|---|---|
| Kaggle dataset page | Licensed CC0: Public Domain (read from the Kaggle API, 2026-10-05) |
| LendingClub terms of use | Restrict scraping the website and copying its content. They do not mention the downloadable historical loan files (the site is now run by Happen Bank) |

CC0 is the label set by the Kaggle uploader. It does not prove that the
uploader had the right to release the data, and LendingClub's terms do not
settle the question either. This project therefore takes the cautious route:

- The data is not in the repository and is not redistributed. Every user
  downloads it from Kaggle.
- The repository contains only code, documentation and aggregate results.
  It publishes no loan-level rows.
- The trained model file is not committed (`artifacts/` is git-ignored).

This is a portfolio project, not legal advice.

## Known limitations

- **Accepted loans only.** The data contains loans Lending Club approved. The
  model has never seen rejected applicants, so its PDs are not valid for
  them (no reject inference is attempted).
- **Field timing.** Credit-bureau fields are assumed to be as of application.
  The dataset does not document this for every column.
- **`verification_status`** is set by the lender during processing, close to
  but not strictly at application time.
- **No fairness audit.** Proxy features are removed, but outcomes across
  groups are not measured.

## Metrics

Reported on the test period, against a logistic regression baseline:

- Discrimination: AUC and Gini
- Calibration: Brier score; predicted vs observed default rate per risk band
- Stability: PSI between train and test periods

Risk bands A–E are PD thresholds fixed on the training period. They are
unrelated to Lending Club's A–G grades.
