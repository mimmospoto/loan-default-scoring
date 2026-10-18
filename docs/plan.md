# Plan

Build order. Each phase ends with something working.

| # | Phase | Status | Ends with |
|---|---|---|---|
| 1 | Framing | Done | Framing doc, feature allowlist, data scripts |
| 2 | Model as package | Done | Tuned, calibrated model with out-of-time results |
| 3 | Registry (MLflow) | Done | Versioned model, exported to `artifacts/` |
| 4 | API (FastAPI) | Done | `/predict`, `/predict/batch`, `/health` running locally |
| 5 | Docker | Done | Container that gives the same predictions as training |
| 6 | Tests + CI | Done | Green CI badge |
| 7 | Deploy (Azure Container Apps) | Done | Live endpoint, automated deploy |
| 8 | Monitoring | Next | Scheduled drift report, shown firing once |
| 9 | Retraining | Optional | Automated retrain and promotion check |

## Success criteria

Model:
- AUC and Gini against a logistic regression baseline (done)
- Brier score (done)
- Predicted against observed default rate per band (done)
- PSI between train and test (done)

System:
- Live endpoint
- Container gives the same predictions as training, tested
- CI green
- Automated deploy
- Scheduled drift report, shown firing once on later vintages

## Decisions taken

- **Cohort:** 36-month loans only, issued 2012-05 to 2015-10.
- **Split:** train 2012-05 to 2014-12, test 2015-01 to 2015-10, drift 2016 to 2017 (features only).
- **Excluded features:** Lending Club's grade, sub-grade, interest rate and installment; geography; free text.
- **Model:** LightGBM, with a logistic regression baseline, both sigmoid-calibrated.
- **Validation:** time-ordered tuning folds, plus walk-forward folds with calibration in each.
- **Registry:** local MLflow store; promotion is a separate, manual step; the API reads only `artifacts/`.
- **API:** input ranges from the data with margins; unknown fields rejected; see [api.md](api.md).
- **Docker:** multi-stage, serving-only requirements, model baked into the image; see [docker.md](docker.md).
- **CI:** GitHub Actions builds the image and runs all tests; the exported model is committed so CI can build; see [ci.md](ci.md).
- **Deploy:** Azure Container Apps in North Europe, deployed from CI with passwordless login, scale to zero, one shared API key; see [deploy.md](deploy.md).
- **Layout:** flat (`loan_default/` at the repo root), plain `.venv`, requirements files. The brief said `src/` layout; this replaced it.
- **Data:** never committed. Kaggle labels it CC0; see the license section in [framing.md](framing.md).

## Open items

- **Phase 8:** log inputs and predictions; the API does not yet.
- **Before going public:** the repo is private and the trained model is committed (see [ci.md](ci.md)); confirm that is fine to publish.
- **Not yet done:** no architecture diagram.
- **Live URL:** kept out of the README on purpose, because bots would keep the app awake and cost money; shared privately with the key.

## Working style

- Explain before coding; go phase by phase.
- Flag uncertainty explicitly.
- Ask before system-wide installs.
- No co-author lines in commits.
