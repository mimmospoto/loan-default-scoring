# Loan default scoring service

A credit default model taken from training to a live, monitored API. The
model is deliberately simple; the engineering around it is the point.

**Status:** Phase 2 of 9 (model). The model trains and is evaluated; no endpoint yet.

## What it does

A lender sends a loan application and receives a calibrated probability of
default and a risk band (A–E).

- [docs/framing.md](docs/framing.md): target, cohort, validation split, and
  feature rules.
- [docs/model.md](docs/model.md): model design and out-of-time results.

On loans issued in 2015, unseen in training, the model reaches an AUC of
0.681 (Gini 0.362), against 0.674 for a logistic regression baseline.
Observed default rates rise from 4.9% in band A to 26.8% in band E.

## Setup

Requires Python 3.12+ and a Kaggle API token. Generate one at
[kaggle.com/settings/api](https://www.kaggle.com/settings/api) and save the
key to `~/.kaggle/access_token` (plain text, `chmod 600`), or run
`kaggle auth login`. On macOS, LightGBM also needs `brew install libomp`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m loan_default.data.download  # fetches data into data/raw/
.venv/bin/python -m loan_default.data.profile   # checks the framing against the data
.venv/bin/python -m loan_default.data.prepare   # builds the modelling table
.venv/bin/python -m loan_default.train          # trains, evaluates, saves to artifacts/
.venv/bin/python -m pytest                      # runs the tests
```

The data is never committed to this repository. It comes from the Kaggle
dataset `wordsforthewise/lending-club`, labelled CC0; see
[docs/framing.md](docs/framing.md#license) for the license check.

## Not in scope

This is a portfolio project, not a production credit decision system. It does
not chase leaderboard accuracy and includes no fairness audit.
