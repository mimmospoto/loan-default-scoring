# Loan default scoring service

[![CI](https://github.com/mimmospoto/loan-default-scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/mimmospoto/loan-default-scoring/actions/workflows/ci.yml)

A credit default model taken from training to a live, monitored API. The
model is deliberately simple; the engineering around it is the point.

**Status:** Phase 7 of 9 (Deploy). The model trains, is versioned, and is served with FastAPI in a container. CI tests it and deploys it to Azure Container Apps.

## What it does

A lender sends a loan application and receives a calibrated probability of
default and a risk band (A–E).

- [docs/framing.md](docs/framing.md): target, cohort, validation split, and
  feature rules.
- [docs/model.md](docs/model.md): model design and out-of-time results.
- [docs/registry.md](docs/registry.md): how a model version goes live.
- [docs/api.md](docs/api.md): endpoints, input rules, and design decisions.
- [docs/docker.md](docs/docker.md): the image, and how it is tested against training.
- [docs/ci.md](docs/ci.md): what CI runs and why the model is committed.
- [docs/deploy.md](docs/deploy.md): how it is deployed to Azure, cost, and why no Kubernetes.

On loans issued in 2015, unseen in training, the model reaches an AUC of
0.681 (Gini 0.362), against 0.674 for a logistic regression baseline.
Observed default rates rise from 4.9% in band A to 26.8% in band E.

## Architecture

```mermaid
flowchart LR
    subgraph local["Training (local)"]
        kaggle[("Kaggle data<br/>never committed")] --> prep["prepare<br/>features + target"]
        prep --> train["train<br/>LightGBM + calibration"]
        train --> mlflow[("MLflow<br/>tracking + registry")]
        mlflow -->|"promote + export"| art["model.joblib<br/>metadata.json"]
    end

    art -->|"git push"| repo["GitHub repo"]

    subgraph ci["GitHub Actions"]
        test["test<br/>unit + container parity"] --> publish["publish<br/>image to GHCR"]
        publish --> deploy["deploy<br/>OIDC login"]
    end

    repo --> test
    deploy --> app

    subgraph azure["Azure Container Apps (North Europe)"]
        app["FastAPI service<br/>scale 0 to 1"]
    end

    client["Lender"] -->|"POST /predict<br/>X-API-Key"| app
    app -->|"pd + risk band"| client

    app -.->|"planned: logs"| drift["Drift report<br/>PSI, scheduled"]
    kaggle -.->|"planned: later vintages"| drift
```

- **Training stays offline.** The service never touches MLflow or the data; it
  loads two exported files.
- **One image is one model version.** CI builds the image from the committed
  model and tests that it predicts exactly like the training code.
- **Deploys are automatic** after the tests pass. Dashed lines are Phase 8,
  not built yet.

## Live endpoint

The service is deployed on Azure Container Apps (North Europe). Its address and
API key are not published here, to keep it from being scanned and billed. Ask
the author for both; see [docs/deploy.md](docs/deploy.md) for the setup, cost
and design decisions.

## Try it

The trained model is in the repo, so scoring needs no data and no training.

```bash
git clone https://github.com/mimmospoto/loan-default-scoring.git
cd loan-default-scoring
docker build -t loan-default .
docker run --rm -p 8000:8000 loan-default
```

Then open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) and use
"Try it out" on `POST /predict`, or send a request:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "purpose": "debt_consolidation", "home_ownership": "RENT",
  "verification_status": "Verified", "loan_amnt": 12000, "annual_inc": 55000,
  "dti": 18.5, "fico": 690, "credit_history_months": 150, "delinq_2yrs": 0,
  "inq_last_6mths": 1, "open_acc": 9, "total_acc": 20, "pub_rec": 0,
  "pub_rec_bankruptcies": 0, "revol_bal": 9000, "mort_acc": 0,
  "acc_open_past_24mths": 3, "emp_length_years": 4, "revol_util": 52
}'
```

The response has `pd`, `risk_band`, `model_version` and `request_id`. Without
Docker: `pip install -r requirements.txt`, then
`python -m uvicorn loan_default.api:app`.

## Setup

Requires Python 3.12+ and a Kaggle API token. Generate one at
[kaggle.com/settings/api](https://www.kaggle.com/settings/api) and save the
key to `~/.kaggle/access_token` (plain text, `chmod 600`), or run
`kaggle auth login`. On macOS, LightGBM also needs `brew install libomp`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m loan_default.data.download  # fetches data into data/raw/
.venv/bin/python -m loan_default.data.profile   # checks the framing against the data
.venv/bin/python -m loan_default.data.prepare   # builds the modelling table
.venv/bin/python -m loan_default.train          # tunes, trains, logs to MLflow (--name, --note optional)
.venv/bin/python -m loan_default.registry promote 1   # makes version 1 the champion
.venv/bin/python -m loan_default.registry export      # writes the champion to artifacts/
.venv/bin/python -m uvicorn loan_default.api:app      # serves the API at http://127.0.0.1:8000/docs
docker build -t loan-default:dev . && docker run --rm -p 8000:8000 loan-default:dev
.venv/bin/python -m pytest                      # runs the tests
```

The data is never committed to this repository. It comes from the Kaggle
dataset `wordsforthewise/lending-club`, labelled CC0; see
[docs/framing.md](docs/framing.md#license) for the license check.

## Not in scope

This is a portfolio project, not a production credit decision system. It does
not chase leaderboard accuracy and includes no fairness audit.
