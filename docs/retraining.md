# Retraining

How a new model challenges the live one, and when it is allowed to replace it.
The code is in [`retrain.py`](../loan_default/retrain.py).

## The idea

A new model is a **challenger**. It is trained and scored under the same
conditions as the **champion** (the model that is live), and it replaces the
champion only if it passes a gate. A person then promotes, exports and deploys
it. Nothing here deploys by itself.

A challenger can differ from the champion in the model, the data window, or the
features. This phase covers the first: same fit window, same calibration
window, same test period, so the model is the only difference.

## Running it

```
python -m loan_default.retrain              # compare only, nothing is saved
python -m loan_default.retrain --register   # log the challenger as a new version
python -m loan_default.retrain --promote    # register, and promote only if the gate passes
```

It needs the modelling table and the exported champion in `artifacts/`. The
exit code is 1 when the gate fails. After a promotion the usual path applies:
`python -m loan_default.registry export`, commit, push, and CI deploys it
([registry.md](registry.md), [deploy.md](deploy.md)).

## The challengers

Three models compete, all on the same windows:

| Name | Model | Why |
|---|---|---|
| `gbm-monotone` | The champion's LightGBM settings, with monotonic constraints on seven features | A score that moves the wrong way for one applicant is hard to explain and usually noise |
| `linear-svm` | Linear support vector machine, calibrated like the others | A different kind of linear model |
| `naive-bayes` | Gaussian Naive Bayes | A simple probabilistic model |

- **Monotone constraints.** The score may only fall as the feature rises
  (`fico`, `annual_inc`, `credit_history_months`, `mort_acc`) or only rise
  (`dti`, `revol_util`, `acc_open_past_24mths`). I chose features whose default
  rate moves in one direction across all ten deciles of the training data, where
  the direction makes credit sense. Ten features stay free. `revol_bal` looks
  monotone but only because it moves with income, so it is not constrained.
- **Why a linear SVM and not an SVM regressor or a kernel SVM.** A regressor gives
  scores, not probabilities, and a kernel SVM on 207,000 loans takes hours.
- **No API change.** All three use the same inputs. A challenger with new features
  would change the API contract, so it is outside this phase.
- **Untuned.** Only the LightGBM challenger reuses tuned settings (the
  champion's). The SVM and Naive Bayes run with library defaults, as the logistic
  regression baseline does.

**How one is chosen.** All challengers and the champion go through the walk-forward
folds in the train period. The challenger with the best mean AUC is the only one
scored on the test period, so the test period is not used to choose between
candidates. `--challengers linear-svm` forces a single one.

## The gate

All three must pass, on the test period (2015-01 to 2015-10), scored once:

| Check | Rule |
|---|---|
| Discrimination | AUC no lower than the champion's |
| Probability quality | Brier no worse than the champion's |
| Calibration | Mean predicted PD within 1 percentage point of the observed rate |

The values are a starting point, not tuned on this data.

**Test period used once.** That period was already used to report the champion,
so trying many challengers on it and keeping the best would make the result
optimistic. Walk-forward validation inside the train period is printed first for
choosing between candidates. The test period is the final check.

## Result

Walk-forward validation, mean AUC over three folds in the train period:

| Model | Mean AUC |
|---|---|
| Champion's LightGBM | 0.6658 |
| `gbm-monotone` | 0.6658 |
| `linear-svm` | 0.6636 |
| `naive-bayes` | 0.6108 |

Naive Bayes is clearly worse: its independence assumption does not hold here
(`open_acc` and `total_acc` move together, for example). The linear SVM is close
to the logistic regression baseline. `gbm-monotone` won validation, so it faced
the champion on the test period, scored once:

| | AUC | Brier | Mean PD | Observed |
|---|---|---|---|---|
| Champion (v1) | 0.6808 | 0.1196 | 14.07% | 14.81% |
| `gbm-monotone` | 0.6798 | 0.1197 | 14.05% | 14.81% |

- **The gate failed** on AUC (by 0.001) and on Brier (by 0.0001). Calibration passed.
- `--promote` registered it as version 2 and refused to promote it. Version 1
  stays champion.
- The constraints cost nothing measurable, and gain nothing in accuracy. If the
  explainability matters more than a tenth of an AUC point, that is a decision
  for a person, and the gate output makes it visible. I did not loosen the gate.
- The SVM and Naive Bayes were not scored on the test period, by design.

## Deciding to retrain

A drift **warning** or **drift** verdict ([monitoring.md](monitoring.md)) is a
reason to look. It does not trigger retraining by itself, because PSI says the
inputs moved, not that the model got worse.

## Limitations

- Retraining runs on a laptop with the Kaggle data. CI never has the data, so
  there is no scheduled retraining. Doing that would need the data in storage
  such as Azure Blob.
- Promotion and deployment are manual by design. A credit model should have a
  person sign off.
- The data ends in 2015-10 for labelled 36-month loans, so there is no newer
  window to retrain on. A challenger with a newer window would need a fresh
  holdout after it.
- Three challengers, one gate. The 1-point calibration limit and the "no lower"
  rules are not tuned, and a tie fails by design.
