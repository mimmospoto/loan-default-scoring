# Monitoring

The service logs what it scores, and a drift report compares that traffic with
the training data. It watches inputs and scores. It cannot watch outcomes (see
"What it cannot see").

## What is monitored

| What | How |
|---|---|
| The 20 input features | PSI against the training period |
| The model's PD | PSI against the training PDs |
| Risk band mix | Share per band; training puts about 20% in each |
| Missing values | Share missing for the three optional features |

PSI thresholds: below 0.1 stable, 0.1 to 0.25 warning, 0.25 or more drift.
These are the usual rules of thumb, not values tuned on this data.

## The request log

Each scored application is one JSON line on stdout: timestamp, `request_id`,
`model_version`, the 20 inputs, `pd` and `risk_band`. A batch writes one line per
application. On Azure Container Apps stdout goes to Log Analytics, so there is
nothing extra to run.

- Only model inputs and outputs are logged. There is no name or identifier.
- Requests that fail validation or the key check are not logged.

## The drift report

```
python -m loan_default.drift --current-period 2017-01 2017-12   # later vintage from the data
python -m loan_default.drift --current-log requests.jsonl        # exported request log
python -m loan_default.drift --self-check                        # made-up traffic, no data needed
```

- The reference is always the full training period (2012-05 to 2014-12), scored
  by the exported model. At most 50,000 loans are sampled per side.
- It writes `report.html` (Evidently) and `summary.md` to `reports/`.
- The exit code is 1 at the status chosen by `--fail-on` (default `drift`), so a
  scheduled job can fail. It refuses to run on fewer than 500 current loans.
- Install with `pip install -r requirements-monitor.txt`. Evidently is not in the
  Docker image.

**One PSI definition.** Evidently's own PSI uses 30 bins and reads higher than
the 10-bin PSI that the 0.1 / 0.25 thresholds are meant for. For example it gave
`loan_amnt` 0.16 on 2017 against 0.05 with deciles. The report therefore passes
our own `psi` (deciles of the reference, the same as at training time) to
Evidently. Evidently provides the HTML report and plots.

**The banner.** Evidently's "Dataset Drift" banner counts the share of columns that
drifted, and by default needs half of them. One badly shifted feature would show
"NOT detected" while our verdict said drift. The report sets that share to 0.01,
so one column at 0.25 or more is enough, the same as the verdict. The 0.1 to 0.25
warning level is only in `summary.md`, not in the banner.

## Result on later vintages

The model was trained on 2012-05 to 2014-12. Compared with that period:

| Window | Verdict | Highest PSI | PD PSI |
|---|---|---|---|
| 2016 | stable | `verification_status` 0.091 | 0.004 |
| 2017 | **warning** | `revol_util` 0.154 | 0.010 |

- With `--fail-on warning` the 2017 run exits with 1. That is the report firing.
  The full table is in [drift-report-2017.md](drift-report-2017.md).
- The drift is mild. No column reaches 0.25, and the score distribution barely
  moves. I did not lower the thresholds to get a louder result.
- The "drift" level is shown by a synthetic shift (every FICO score 60 lower,
  PSI above 4). It is tested in `tests/test_drift.py` and runs in the self-check.

## The schedule

`.github/workflows/drift.yml` runs every Monday at 06:00 UTC, and on demand from
the Actions tab. It runs `--self-check`: made-up applications go through the
committed model, once unchanged and once shifted. The job fails if the unchanged
window is not stable or the shifted one is not flagged. The reports are attached
to the run, and the summaries are shown on its page.

**This is a test of the monitoring pipeline, not of live traffic.** The Lending
Club data is never in the repository, and the live service has almost no
traffic. A report on real traffic is run by hand with `--current-log`.

## What it cannot see

- **Concept drift** (the link between inputs and default changes) needs the real
  outcomes. They arrive after the 36-month term plus the charge-off lag, so years
  after the score. The 2016 and 2017 vintages are only about 56% terminal, so an
  alarm on them would be biased towards early defaults. A back-test of predicted
  against observed default rate per band is possible on matured vintages, and
  was done for 2015 (see [model.md](model.md)). It is not scheduled.
- **Fairness.** The data has no protected-attribute columns, so disparate impact
  cannot be measured here. The safeguard is the design rule: no features that
  proxy protected attributes (geography, free text and Lending Club's own
  grading are excluded). A lender would need group data for a real audit. This is
  a non-goal of the project.
- **Whether a shift matters.** PSI says the inputs moved, not whether the model
  got worse.

## Limitations

- The scheduled job does not read the Azure logs. Exporting them from Log
  Analytics is manual.
- No alert is sent; a failed run shows in GitHub's notifications and the Actions
  tab.
- One reference window and fixed thresholds. No per-feature thresholds.
- The report ignores the model version in the log lines; after a retrain the
  reference must be rebuilt for the new model.
