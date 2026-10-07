# Model

How the model is built and how it performs out of time. The problem
definition is in [framing.md](framing.md). Every number here is printed by
`python -m loan_default.train` and saved to `artifacts/metadata.json`.

## Design

| Step | Choice | Why |
|---|---|---|
| Inputs | 20 typed features ([`features.py`](../loan_default/features.py)) | The same list becomes the API contract |
| Baseline | Logistic regression | The standard scorecard model; sets the bar |
| Model | Gradient boosting (LightGBM) | Handles missing values natively; the standard choice for tabular credit data |
| Tuning | Small grid search on time-ordered folds | Settings are chosen by a method, not by hand, and never on the test period |
| Calibration | Sigmoid on the model's log-odds, fitted on later loans | Turns scores into probabilities without changing their ranking |
| Bands | Quintiles of training PDs, then frozen | A band keeps the same meaning as the population shifts |

Each model is one object that takes the feature table and returns a PD.
Preprocessing lives inside it, so training and serving cannot diverge.

The training period is split in time:

| Part | Issue months | Loans | Used for |
|---|---|---|---|
| Fit | 2012-05 to 2014-06 | 206,669 | Fitting the model |
| Calibration | 2014-07 to 2014-12 | 90,615 | Fitting the calibration |
| Test | 2015-01 to 2015-10 | 227,543 | The final results below |

## Tuning

LightGBM's settings are chosen by a grid search over 18 combinations
([`tune.py`](../loan_default/tune.py)). The folds come from scikit-learn's
`TimeSeriesSplit`, applied to issue months: each fold trains on earlier
months and is scored on the eight months that follow. A month is never
split between the two sides.

| Fold | Trained on | Scored on |
|---|---|---|
| 1 | 2012-05 to 2012-12 | 2013-01 to 2013-08 |
| 2 | 2012-05 to 2013-08 | 2013-09 to 2014-04 |
| 3 | 2012-05 to 2014-04 | 2014-05 to 2014-12 |

The combination with the highest mean AUC wins:

| Setting | Values tried | Chosen |
|---|---|---|
| Number of trees | 100, 200, 400 | 400 |
| Leaves per tree | 7, 15, 31 | 7 |
| Minimum loans per leaf | 100, 400 | 400 |

The search prefers small, heavily regularised trees. The differences are
small: the top five combinations are within 0.002 AUC of each other
(0.670 for the winner). On the test period the tuned model scores the same
as the untuned one (AUC 0.681). The benefit is on smaller training sets,
where the untuned model fell behind the baseline (see walk-forward below).

## Results on the test period

| Model | AUC | Gini | Brier |
|---|---|---|---|
| Logistic regression | 0.674 | 0.347 | 0.1202 |
| LightGBM | 0.681 | 0.362 | 0.1196 |

LightGBM beats the baseline by 0.7 AUC points. The gain is small:
with 20 conventional credit features, most of the signal is close to linear.
The discrimination level is modest by design, because Lending Club's own
grade and interest rate are excluded (see framing).

### Risk bands

Thresholds from training PDs: 0.0714, 0.1047, 0.1425, 0.1959.

| Band | Share of loans | Predicted PD | Observed default rate |
|---|---|---|---|
| A | 19.7% | 5.0% | 4.9% |
| B | 19.0% | 8.8% | 9.2% |
| C | 19.3% | 12.3% | 13.5% |
| D | 19.8% | 16.7% | 17.9% |
| E | 22.2% | 25.8% | 26.8% |

Observed default rates rise in every band, from 4.9% to 26.8%.

The model under-predicts slightly in bands B to E, and by 0.7 points overall
(mean PD 14.1% against 14.8% observed). The default rate rose from 13.3% in
training to 14.8% in the test period, and a model calibrated on the past
cannot know that. This gap is the reason the service monitors drift.

### Stability, training to test

| Measure | PSI |
|---|---|
| Model PD | 0.004 |
| `verification_status` | 0.066 |
| `inq_last_6mths` | 0.041 |
| `dti` | 0.024 |

All values are under 0.1, the usual threshold for a stable population.

## Walk-forward validation

One test period gives one number. To see how the results move over time, the
full procedure is repeated three times on earlier data, inside the training
period. Each fold fits a model, calibrates it on the next six months, and
scores it on the six months after that. The test period is never used.

| Validated on | Loans fitted on | AUC, baseline | AUC, LightGBM | Predicted PD | Observed |
|---|---|---|---|---|---|
| 2013-07 to 2013-12 | 34,292 | 0.660 | 0.658 | 12.6% | 12.1% |
| 2014-01 to 2014-06 | 75,416 | 0.661 | 0.664 | 12.8% | 13.2% |
| 2014-07 to 2014-12 | 134,714 | 0.669 | 0.675 | 13.6% | 14.1% |
| Test: 2015-01 to 2015-10 | 206,669 | 0.674 | 0.681 | 14.1% | 14.8% |

Predicted and observed values are for LightGBM. The last row is the final
model, shown for comparison.

What the folds show:

- **LightGBM's lead grows with data.** With 34,000 loans it only matches the
  baseline. Its lead widens to 0.7 AUC points at 200,000 loans. For a small
  portfolio the baseline is just as good.
- **Both models improve as data grows**, from an AUC of about 0.66 to 0.68.
- **Calibration is off by about half a point, in either direction.** The
  0.7-point gap on the test period is typical, not a one-off.
- **The gap follows the trend in defaults.** The observed rate climbs in
  every period, from 12.1% to 14.8%. A model calibrated on the recent past
  under-predicts whenever defaults keep rising.

The LightGBM settings were tuned on these same months, so its fold scores
are slightly optimistic. The test-period row is not affected.

## Limitations

- **Light tuning only.** The grid has 18 combinations, and the winner sits
  at the edge of it. A wider search might find slightly better settings.
  Leaderboard accuracy is a non-goal.
- **Calibration ages.** PDs are calibrated to 2014 default levels.
- **One feature is unstable in early data.** `mths_since_last_delinq` has a
  missing rate that moves between 30% and 71% during 2012, which suggests a
  recording change in the source.
