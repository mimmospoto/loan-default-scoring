"""MLflow tracking and model registry.

Training logs every run and registers a new model version. A version goes
live in two explicit steps:

    python -m loan_default.registry list
    python -m loan_default.registry promote <version>
    python -m loan_default.registry export

Export writes the champion to artifacts/, which is all the API needs.
MLflow itself is never required at serving time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from loan_default.config import ARTIFACTS_DIR, PROJECT_ROOT

EXPERIMENT_NAME = "loan-default"
MODEL_NAME = "loan-default-pd"
CHAMPION = "champion"
METADATA_FILE = "metadata.json"

# MLflow stores models in the skops format, which refuses to load any type
# that is not listed here. Add to this list when the model's classes change.
TRUSTED_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "loan_default.model.LogOddsLGBMClassifier",
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
]


def configure() -> None:
    """Point MLflow at the local store, unless MLFLOW_TRACKING_URI says otherwise."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def default_run_name() -> str:
    """Unique, sortable name such as 'lgbm-2026-10-10-2237'."""
    return datetime.now().strftime("lgbm-%Y-%m-%d-%H%M")


def log_run(
    model, metadata: dict, params: dict, metrics: dict, input_example, name: str | None = None, note: str | None = None
) -> str:
    """Log one training run and register its model. Returns the new version."""
    configure()
    with mlflow.start_run(run_name=name or default_run_name(), description=note):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(metadata, METADATA_FILE)
        with warnings.catch_warnings():
            # MLflow warns that the predicted class is an integer. The service
            # only uses probabilities, so the warning does not apply.
            warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name=MODEL_NAME,
                input_example=input_example,
                skops_trusted_types=TRUSTED_TYPES,
            )
    return str(info.registered_model_version)


def promote(version: str) -> None:
    """Make this version the champion. The previous champion is replaced."""
    configure()
    MlflowClient().set_registered_model_alias(MODEL_NAME, CHAMPION, version)


def export(out_dir: Path = ARTIFACTS_DIR) -> dict:
    """Write the champion model and its metadata to out_dir. Returns the metadata."""
    configure()
    champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION}")
    metadata = mlflow.artifacts.load_dict(f"runs:/{champion.run_id}/{METADATA_FILE}")
    metadata.update(model_name=MODEL_NAME, model_version=str(champion.version), run_id=champion.run_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))
    return metadata


def list_versions() -> list[dict]:
    """Every registered version, newest first, with its headline test metric."""
    configure()
    client = MlflowClient()
    # Searching versions does not return aliases, so look the champion up directly.
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION).version
    except MlflowException:
        champion = None
    rows = []
    for version in client.search_model_versions(f"name = '{MODEL_NAME}'"):
        run = client.get_run(version.run_id)
        rows.append(
            {
                "version": int(version.version),
                "champion": version.version == champion,
                "name": run.info.run_name,
                "note": run.data.tags.get("mlflow.note.content", ""),
                "test_auc": run.data.metrics.get("test_auc_gbm"),
                "run_id": version.run_id,
            }
        )
    return sorted(rows, key=lambda row: row["version"], reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage registered model versions.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show registered versions")
    commands.add_parser("promote", help="make a version the champion").add_argument("version")
    commands.add_parser("export", help="write the champion to artifacts/")
    args = parser.parse_args()

    if args.command == "list":
        for row in list_versions():
            mark = "champion" if row["champion"] else ""
            note = f"  - {row['note']}" if row["note"] else ""
            print(f"v{row['version']:<4} {row['name']:<24} test AUC {row['test_auc']:.4f}  {mark}{note}")
    elif args.command == "promote":
        promote(args.version)
        print(f"Version {args.version} is now the {CHAMPION}.")
    else:
        metadata = export()
        print(f"Exported version {metadata['model_version']} to {ARTIFACTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
