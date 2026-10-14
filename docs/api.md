# API

A lender sends an application and gets back a calibrated probability of
default and a risk band. The code is in [`api.py`](../loan_default/api.py),
[`schemas.py`](../loan_default/schemas.py) and
[`service.py`](../loan_default/service.py).

## Run it

```bash
.venv/bin/python -m loan_default.registry export       # once: writes artifacts/
.venv/bin/python -m uvicorn loan_default.api:app       # serves on http://127.0.0.1:8000
```

Interactive docs are at `/docs`. `ARTIFACTS_DIR` points the service at another
folder.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | One application |
| POST | `/predict/batch` | 1 to 1,000 applications in one call |
| GET | `/health` | Status and the model version that is loaded |

`/predict` returns:

```json
{"pd": 0.1838, "risk_band": "D", "model_version": "1", "request_id": "10ca67e9-..."}
```

`/predict/batch` takes `{"applications": [...]}` and returns
`{"predictions": [{"pd", "risk_band"}, ...], "model_version", "request_id"}`.
Predictions come back in the order sent. One `request_id` covers the call.

`risk_band` runs from A (lowest risk) to E. The thresholds are fixed at
training time and ship with the model version.

## Input

The fields are the model features, and nothing else is accepted. Every field
has a type and a range; see `/docs` for the full list.

- **Required:** the three categories and every number that is never missing
  in training data.
- **Optional:** `emp_length_years`, `mths_since_last_delinq`, `revol_util`.
  These are missing for some real loans, so the model already handles them.
  Leave the field out, or send `null`.
- **Categories:** fixed lists. `home_ownership` is `RENT`, `MORTGAGE`, `OWN` or
  `OTHER`.
- **Rule across fields:** `open_acc` cannot exceed `total_acc`.

Invalid input gets a `422` that names the field. In a batch, one bad
application rejects the whole call, and the error says which one.

## Authentication

If the `API_KEY` environment variable is set, `/predict` and `/predict/batch`
need a matching `X-API-Key` header, and anything else gets a `401`. `/health`
stays open so the platform can probe it. With no `API_KEY`, scoring is open,
which suits local runs and tests. A deployment must set the key.

```bash
curl -X POST http://127.0.0.1:8000/predict -H "X-API-Key: <key>" ...
```

The key is compared in constant time and checked before the input, so a caller
without it learns nothing about the input rules.

## Design decisions

- **Ranges come from the data, with margins.** Each limit is well outside what
  training saw (for example `dti` up to 100 where training tops out at 40).
  They exist to catch typos and impossible values, not unusual applicants.
- **Unknown fields are errors.** A misspelled optional field would otherwise
  count as missing and silently change the score.
- **Checked against all 1.17M loans in the table.** 891 are rejected, all data
  oddities: 534 with `dti` above 100 (mostly 2016–17, up to 999), 2
  `educational` loans (a purpose the model never saw), and 2 incomes of 11M and
  61M with a `dti` near 0.
- **FICO accepts 300 to 850,** the full scale. Training only has 660 and above,
  so scores below that are extrapolation and the service does not warn.
- **The model loads at start-up.** With no exported model the service refuses
  to start, rather than failing on the first request.
- **The service reads only `artifacts/`.** No MLflow, no database.
- **Same code as training.** `service.py` calls `predict_pd` and `assign_band`.
  A test sends 50 applications through the API and compares the result with
  the training code. The probabilities must be identical.

## Limitations

- One shared key, no per-caller keys and no rate limit. That is enough for a
  demo, not for real clients.
- The service is open if `API_KEY` is left unset, so the deployment must set it.
- Inputs and predictions are not logged yet. Drift monitoring (Phase 8) needs
  them.
- Batch size is capped at 1,000 to keep response time predictable.
