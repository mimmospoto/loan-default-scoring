"""Drift report: has the data the service sees moved away from the training data?

Run:
    python -m loan_default.drift --current-period 2017-01 2017-12
    python -m loan_default.drift --current-log requests.jsonl

Evidently computes the PSI of every feature and of the model's PD, and writes
the HTML report. This module adds what Evidently does not: a three-level
status, the share of loans per risk band, missing-value rates, and an exit
code so a scheduled job can fail when drift is found.

Only inputs and scores are compared. Checking for concept drift needs the
real outcomes, which arrive years after the loan (docs/monitoring.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from loan_default.bands import assign_band
from loan_default.config import ARTIFACTS_DIR, BAND_LABELS, PROCESSED_FILE, TRAIN_PERIOD
from loan_default.features import MODEL_FEATURES
from loan_default.metrics import psi as decile_psi
from loan_default.model import predict_pd
from loan_default.service import Scorer
from loan_default.schemas import LoanApplication
from loan_default.splits import between
from loan_default.synthetic import make_applications

# Rule of thumb used throughout the project (see loan_default/metrics.py).
WARNING_PSI = 0.10
DRIFT_PSI = 0.25
STATUS_ORDER = {"stable": 0, "warning": 1, "drift": 2}

OPTIONAL_FEATURES = ["emp_length_years", "mths_since_last_delinq", "revol_util"]
MAX_ROWS = 50_000  # per side; PSI is stable well below this and Evidently gets slow above it
MIN_CURRENT_ROWS = 500  # below this a PSI says more about the sample than about drift


def psi_test(reference: pd.Series, current: pd.Series, feature_type, threshold: float) -> tuple[float, bool]:
    """Our PSI as an Evidently drift method.

    Evidently's built-in PSI uses 30 equal-width bins, which reads higher than the
    10-bin PSI that the 0.1 / 0.25 rule of thumb is calibrated for, and than the
    PSI computed at training time. Using one definition everywhere keeps the numbers comparable.
    """
    value = decile_psi(reference, current)
    return value, value >= threshold


PSI_METHOD = "decile_psi"
# Evidently's banner says "Dataset Drift" when this share of columns drifted. Its default is
# 0.5, so one badly shifted feature goes unnoticed. This small value makes one column enough,
# the same rule as our verdict.
ANY_COLUMN = 0.01


def register_psi_method() -> None:
    """Make psi_test available to Evidently under the name PSI_METHOD. Safe to call twice."""
    from evidently.legacy.calculations.stattests.registry import StatTest, register_stattest
    from evidently.legacy.core import ColumnType

    register_stattest(
        StatTest(
            name=PSI_METHOD,
            display_name="PSI (deciles)",
            allowed_feature_types=[ColumnType.Categorical, ColumnType.Numerical],
            default_threshold=DRIFT_PSI,
        ),
        psi_test,
    )


def status(value: float) -> str:
    if value >= DRIFT_PSI:
        return "drift"
    if value >= WARNING_PSI:
        return "warning"
    return "stable"


@dataclass
class DriftResult:
    psi: pd.Series  # PSI per feature and for "pd", largest first
    verdict: str  # worst status over all columns
    reference_rows: int
    current_rows: int
    bands: pd.DataFrame  # share of loans per risk band
    missing: pd.DataFrame  # share missing per optional feature
    html: str


def scored_reference(table: pd.DataFrame, scorer: Scorer) -> pd.DataFrame:
    """Loans from `table` as the service sees them: features plus the model's PD and band."""
    frame = table[MODEL_FEATURES].reset_index(drop=True)
    probabilities = predict_pd(scorer.model, frame)
    frame["pd"] = probabilities
    frame["risk_band"] = assign_band(probabilities, scorer.thresholds)
    return frame


def load_request_log(path: Path) -> pd.DataFrame:
    """Read the JSON lines written by loan_default/request_log.py. Other lines (uvicorn's own) are skipped."""
    rows = []
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("event") == "score":
            rows.append({**record["input"], "pd": record["pd"], "risk_band": record["risk_band"]})
    if not rows:
        raise ValueError(f"No score lines in {path}.")
    frame = pd.DataFrame(rows)
    return frame[MODEL_FEATURES + ["pd", "risk_band"]]


def _sample(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sample(MAX_ROWS, random_state=0) if len(frame) > MAX_ROWS else frame


def compare(reference: pd.DataFrame, current: pd.DataFrame) -> DriftResult:
    """PSI of every feature and of the PD, current against reference."""
    os.environ.setdefault("EVIDENTLY_DISABLE_TELEMETRY", "1")
    from evidently import Report
    from evidently.presets import DataDriftPreset

    register_psi_method()
    columns = MODEL_FEATURES + ["pd"]
    ref, cur = _sample(reference)[columns], _sample(current)[columns]

    snapshot = Report([DataDriftPreset(method=PSI_METHOD, threshold=DRIFT_PSI, drift_share=ANY_COLUMN)]).run(cur, ref)
    values = {
        metric["config"]["column"]: float(metric["value"])
        for metric in snapshot.dict()["metrics"]
        if metric["config"]["type"].endswith(":ValueDrift")
    }
    missing_columns = set(columns) - set(values)
    if missing_columns:
        raise RuntimeError(f"Evidently returned no PSI for: {sorted(missing_columns)}")
    psi = pd.Series(values).sort_values(ascending=False)

    def shares(frame: pd.DataFrame) -> pd.Series:
        return frame["risk_band"].value_counts(normalize=True).reindex(BAND_LABELS, fill_value=0.0)

    bands = pd.DataFrame({"reference": shares(reference), "current": shares(current)})
    missing = pd.DataFrame(
        {"reference": reference[OPTIONAL_FEATURES].isna().mean(), "current": current[OPTIONAL_FEATURES].isna().mean()}
    )
    verdict = max((status(v) for v in psi), key=STATUS_ORDER.get)
    return DriftResult(psi, verdict, len(reference), len(current), bands, missing, snapshot.get_html_str(as_iframe=False))


def summary(result: DriftResult, reference_label: str, current_label: str) -> str:
    """The short Markdown version of the report."""
    lines = [
        "# Drift report",
        "",
        f"- Reference: {reference_label} ({result.reference_rows:,} loans)",
        f"- Current: {current_label} ({result.current_rows:,} loans)",
        f"- PSI thresholds: below {WARNING_PSI} stable, {WARNING_PSI} to {DRIFT_PSI} warning, {DRIFT_PSI} or more drift",
        f"- **Verdict: {result.verdict}**",
        "",
        "## PSI per column",
        "",
        "| Column | PSI | Status |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | {value:.3f} | {status(value)} |" for name, value in result.psi.items()]
    lines += ["", "## Share of loans per risk band", "", "| Band | Reference | Current |", "|---|---|---|"]
    lines += [f"| {band} | {r.reference:.1%} | {r.current:.1%} |" for band, r in result.bands.iterrows()]
    lines += ["", "## Share missing", "", "| Feature | Reference | Current |", "|---|---|---|"]
    lines += [f"| {name} | {r.reference:.1%} | {r.current:.1%} |" for name, r in result.missing.iterrows()]
    lines.append("")
    return "\n".join(lines)


def score_applications(applications: list[dict], scorer: Scorer) -> pd.DataFrame:
    """Made-up applications scored by the real model, laid out like the training table."""
    scores = scorer.score([LoanApplication(**a) for a in applications])
    frame = pd.DataFrame(applications).reindex(columns=MODEL_FEATURES)
    frame["pd"] = [p for p, _ in scores]
    frame["risk_band"] = [b for _, b in scores]
    return frame


def self_check(scorer: Scorer, out: Path) -> int:
    """Prove the monitoring pipeline works, with no real data needed.

    Scores made-up applications with the exported model, then compares an unchanged
    window and a shifted one (every FICO score 60 points lower) with the reference.
    Passes when the first is stable and the second is flagged as drift.
    """
    reference = score_applications(make_applications(5000, seed=100), scorer)
    unchanged = score_applications(make_applications(2000, seed=101), scorer)
    shifted_applications = make_applications(2000, seed=102)
    for application in shifted_applications:
        application["fico"] = max(300.0, application["fico"] - 60)
    shifted = score_applications(shifted_applications, scorer)

    passed = True
    for name, current, expected in [("unchanged", unchanged, "stable"), ("shifted", shifted, "drift")]:
        result = compare(reference, current)
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "report.html").write_text(result.html)
        (folder / "summary.md").write_text(summary(result, "made-up applications", f"{name} window"))
        worst = result.psi.index[0]
        print(f"{name}: verdict {result.verdict} (expected {expected}); highest PSI {worst} {result.psi.iloc[0]:.3f}")
        passed &= result.verdict == expected
    print("Self-check passed." if passed else "Self-check FAILED.")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare current loans with the training data.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--self-check", action="store_true", help="test the pipeline on made-up traffic; needs no data")
    source.add_argument("--current-period", nargs=2, metavar=("FROM", "TO"), help="issue months, e.g. 2017-01 2017-12")
    source.add_argument("--current-log", type=Path, help="request log written by the API (JSON lines)")
    parser.add_argument("--out", type=Path, default=Path("reports"), help="folder for report.html and summary.md")
    parser.add_argument(
        "--fail-on", choices=["warning", "drift"], default="drift", help="exit code 1 at this status or worse"
    )
    args = parser.parse_args()

    scorer = Scorer.load(Path(os.environ.get("ARTIFACTS_DIR", ARTIFACTS_DIR)))
    if args.self_check:
        return self_check(scorer, args.out)

    if not PROCESSED_FILE.exists():
        print("No modelling table. Run: python -m loan_default.data.prepare", file=sys.stderr)
        return 2
    table = pd.read_parquet(PROCESSED_FILE)
    reference = scored_reference(between(table, TRAIN_PERIOD), scorer)
    reference_label = f"training loans issued {TRAIN_PERIOD[0]} to {TRAIN_PERIOD[1]}, model v{scorer.version}"

    if args.current_log:
        current, current_label = load_request_log(args.current_log), f"request log {args.current_log.name}"
    else:
        start, end = args.current_period
        in_period = (table["issue_month"] >= start) & (table["issue_month"] <= end)
        current = scored_reference(table[in_period], scorer)
        current_label = f"loans issued {start} to {end}"

    if len(current) < MIN_CURRENT_ROWS:
        print(f"Only {len(current)} current loans; need {MIN_CURRENT_ROWS} for a meaningful PSI.", file=sys.stderr)
        return 2

    result = compare(reference, current)
    text = summary(result, reference_label, current_label)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.html").write_text(result.html)
    (args.out / "summary.md").write_text(text)
    print(text)
    return 1 if STATUS_ORDER[result.verdict] >= STATUS_ORDER[args.fail_on] else 0


if __name__ == "__main__":
    sys.exit(main())
