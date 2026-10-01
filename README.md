# Loan default scoring service

A credit default model taken from training to a live, monitored API. The
model is deliberately simple; the engineering around it is the point.

**Status:** Phase 1 of 9 (framing). No model or endpoint yet.

## What it does

A lender sends a loan application and receives a calibrated probability of
default and a risk band (A–E). See [docs/framing.md](docs/framing.md) for the
target, cohort, validation split, and feature rules.

## Setup

Requires Python 3.12+ and a Kaggle API token. Generate one at
[kaggle.com/settings/api](https://www.kaggle.com/settings/api) and save the
key to `~/.kaggle/access_token` (plain text, `chmod 600`), or run
`kaggle auth login`.

```bash
python3 -m venv .venv
.venv/bin/pip install -e . --group data   # or: pip install -e ".[data]" on older pip
PYTHONPATH=src .venv/bin/python -m loan_default.data.download  # fetches data into data/raw/
PYTHONPATH=src .venv/bin/python -m loan_default.data.profile   # checks the framing against the data
```

The data is never committed to this repository.

## Not in scope

This is a portfolio project, not a production credit decision system. It does
not chase leaderboard accuracy and includes no fairness audit.
