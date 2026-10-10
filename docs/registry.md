# Tracking and registry

How a trained model becomes the model the service uses. The code is in
[`registry.py`](../loan_default/registry.py).

## The path from training to serving

| Step | Command | What happens |
|---|---|---|
| 1. Train | `python -m loan_default.train --name "wider grid" --note "more leaves"` | The run is logged to MLflow and the model is registered as a new version. Name and note are optional |
| 2. Review | `python -m loan_default.registry list` | Shows every version with its test AUC and which one is champion |
| 3. Promote | `python -m loan_default.registry promote <version>` | Marks that version as the champion |
| 4. Export | `python -m loan_default.registry export` | Writes the champion to `artifacts/` |

Without `--name`, a run is called `lgbm-<date>-<time>`, such as
`lgbm-2026-10-10-2237`. `registry list` shows each version's name and note.

Training never changes what is live. A new version only goes live when
someone promotes it. Rolling back is the same command with an older version.

## What each run records

| Kind | Content |
|---|---|
| Parameters | The tuned LightGBM settings; the fit, calibration and test periods |
| Metrics | AUC, Gini and Brier for both models on the test period; mean PD against the observed default rate; PSI of the PD |
| Model | The calibrated pipeline, as one object |
| Metadata | `metadata.json`: feature list, band thresholds, tuning table, walk-forward results, band table, library versions |

The band thresholds travel with the model version, so a model can never be
served with thresholds from another training run.

## What the service loads

Export writes two files:

- `artifacts/model.joblib`: the model.
- `artifacts/metadata.json`: the metadata above, plus `model_version`.

The API reads only these two files. It does not need MLflow or a database,
which keeps the container small and removes a runtime dependency. The
`model_version` in every API response comes from this file.

A test checks the whole path: a model that is logged, promoted and exported
gives exactly the same predictions as the model that was trained.

## Design decisions

- **Local store.** Runs go to a SQLite file (`mlflow.db`) and a folder
  (`mlruns/`) in the project. Both are git-ignored. Setting
  `MLFLOW_TRACKING_URI` points the same code at a shared server instead.
- **Aliases, not stages.** The champion is an MLflow alias. Stages are
  deprecated in current MLflow versions.
- **Restricted loading.** MLflow stores the model in the skops format, which
  only loads types named in an allowlist. The list is `TRUSTED_TYPES` in
  `registry.py`.

## Limitations

- The local store lives on one machine. Nobody else can see the runs.
- Promotion is manual. There is no automatic check that a new version beats
  the champion.
