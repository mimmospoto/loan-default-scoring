# Drift report

- Reference: training loans issued 2012-05 to 2014-12, model v1 (297,284 loans)
- Current: loans issued 2017-01 to 2017-12 (320,419 loans)
- PSI thresholds: below 0.1 stable, 0.1 to 0.25 warning, 0.25 or more drift
- **Verdict: warning**

## PSI per column

| Column | PSI | Status |
|---|---|---|
| revol_util | 0.154 | warning |
| purpose | 0.079 | stable |
| inq_last_6mths | 0.071 | stable |
| verification_status | 0.051 | stable |
| mort_acc | 0.048 | stable |
| loan_amnt | 0.047 | stable |
| total_acc | 0.044 | stable |
| fico | 0.041 | stable |
| revol_bal | 0.030 | stable |
| emp_length_years | 0.026 | stable |
| open_acc | 0.025 | stable |
| dti | 0.024 | stable |
| acc_open_past_24mths | 0.014 | stable |
| annual_inc | 0.014 | stable |
| credit_history_months | 0.013 | stable |
| home_ownership | 0.009 | stable |
| pd | 0.007 | stable |
| pub_rec | 0.004 | stable |
| mths_since_last_delinq | 0.003 | stable |
| pub_rec_bankruptcies | 0.002 | stable |
| delinq_2yrs | 0.000 | stable |

## Share of loans per risk band

| Band | Reference | Current |
|---|---|---|
| A | 20.0% | 22.4% |
| B | 20.0% | 19.5% |
| C | 20.0% | 19.6% |
| D | 20.0% | 19.3% |
| E | 20.0% | 19.2% |

## Share missing

| Feature | Reference | Current |
|---|---|---|
| emp_length_years | 5.4% | 7.9% |
| mths_since_last_delinq | 52.3% | 49.9% |
| revol_util | 0.1% | 0.1% |
