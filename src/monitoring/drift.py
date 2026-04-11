from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import settings

LOG_DIR = settings.PROJECT_ROOT / "data" / "logs"
REPORT_DIR = settings.PROJECT_ROOT / "data" / "reports"


@dataclass
class DriftAnalysis:
    status: str
    reason: str | None = None
    drift_detected: bool | None = None
    drift_share: float | None = None
    reference_rows: int = 0
    current_rows: int = 0
    report_path: str | None = None
    summary_path: str | None = None
    column_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_predictions(log_dir: Path = LOG_DIR) -> pd.DataFrame:
    path = Path(log_dir) / "predictions.jsonl"
    rows = _read_jsonl(path)
    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "user_id",
                "variant",
                "items",
                "latency_ms",
                "num_items",
                "top_item",
                "request_hour",
                "request_dayofweek",
            ]
        )

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    else:
        df["timestamp"] = pd.NaT

    if "items" not in df.columns:
        df["items"] = [[] for _ in range(len(df))]
    df["items"] = df["items"].apply(lambda value: value if isinstance(value, list) else [])
    df["num_items"] = df["items"].apply(len)
    df["top_item"] = df["items"].apply(lambda items: items[0] if items else None)
    df["latency_ms"] = pd.to_numeric(df.get("latency_ms"), errors="coerce")
    df["variant"] = df.get("variant", pd.Series(["unknown"] * len(df))).fillna("unknown")
    df["request_hour"] = df["timestamp"].dt.hour.fillna(-1).astype(int)
    df["request_dayofweek"] = df["timestamp"].dt.dayofweek.fillna(-1).astype(int)
    return df.sort_values("timestamp", na_position="last").reset_index(drop=True)


def load_feedback(log_dir: Path = LOG_DIR) -> pd.DataFrame:
    path = Path(log_dir) / "feedback.jsonl"
    rows = _read_jsonl(path)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "logged_at", "user_id", "item_id", "action"])

    df = pd.DataFrame(rows)
    for col in ("timestamp", "logged_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df.sort_values("logged_at", na_position="last").reset_index(drop=True)


def build_drift_dataset(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "latency_ms",
                "num_items",
                "variant",
                "request_hour",
                "request_dayofweek",
            ]
        )

    dataset = predictions[
        ["latency_ms", "num_items", "variant", "request_hour", "request_dayofweek"]
    ].copy()
    dataset["variant"] = dataset["variant"].astype(str)
    dataset["latency_ms"] = pd.to_numeric(dataset["latency_ms"], errors="coerce")
    dataset["num_items"] = pd.to_numeric(dataset["num_items"], errors="coerce")
    dataset["request_hour"] = pd.to_numeric(dataset["request_hour"], errors="coerce")
    dataset["request_dayofweek"] = pd.to_numeric(dataset["request_dayofweek"], errors="coerce")
    return dataset.dropna(how="all").reset_index(drop=True)


def split_reference_current(
    dataset: pd.DataFrame,
    reference_size: float = 0.5,
    min_rows: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset.empty:
        return dataset.copy(), dataset.copy()

    reference_size = min(max(reference_size, 0.1), 0.9)
    split_idx = int(len(dataset) * reference_size)
    split_idx = max(split_idx, min_rows)
    split_idx = min(split_idx, len(dataset) - min_rows)

    if split_idx <= 0 or split_idx >= len(dataset):
        return dataset.iloc[0:0].copy(), dataset.copy()

    return (
        dataset.iloc[:split_idx].reset_index(drop=True),
        dataset.iloc[split_idx:].reset_index(drop=True),
    )


def _categorical_drift(reference: pd.Series, current: pd.Series) -> tuple[float, bool]:
    ref_dist = reference.fillna("missing").astype(str).value_counts(normalize=True)
    cur_dist = current.fillna("missing").astype(str).value_counts(normalize=True)
    keys = sorted(set(ref_dist.index).union(cur_dist.index))
    tvd = 0.5 * sum(abs(ref_dist.get(key, 0.0) - cur_dist.get(key, 0.0)) for key in keys)
    return float(tvd), bool(tvd > 0.1)


def _numeric_drift(reference: pd.Series, current: pd.Series) -> tuple[float, bool]:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return 0.0, False

    ref_mean = float(ref.mean())
    cur_mean = float(cur.mean())
    scale = max(float(ref.std(ddof=0)), 1.0)
    score = abs(cur_mean - ref_mean) / scale
    return float(score), bool(score > 0.5)


def _fallback_drift_analysis(reference: pd.DataFrame, current: pd.DataFrame) -> DriftAnalysis:
    column_results: list[dict[str, Any]] = []
    drifted_columns = 0

    for column in reference.columns:
        ref_col = reference[column]
        cur_col = current[column]
        if pd.api.types.is_numeric_dtype(ref_col):
            score, drifted = _numeric_drift(ref_col, cur_col)
            test_name = "normalized_mean_shift"
        else:
            score, drifted = _categorical_drift(ref_col, cur_col)
            test_name = "total_variation_distance"

        if drifted:
            drifted_columns += 1

        column_results.append(
            {
                "column": column,
                "drift_score": round(score, 4),
                "drift_detected": drifted,
                "method": test_name,
            }
        )

    total_columns = max(len(column_results), 1)
    drift_share = drifted_columns / total_columns

    return DriftAnalysis(
        status="fallback",
        reason="Evidently is unavailable; using lightweight local drift heuristics.",
        drift_detected=drift_share > 0.3,
        drift_share=round(drift_share, 4),
        reference_rows=len(reference),
        current_rows=len(current),
        column_results=column_results,
    )


def _extract_evidently_results(
    report_dict: dict[str, Any]
) -> tuple[bool | None, float | None, list[dict[str, Any]]]:
    metrics = report_dict.get("metrics", [])
    if not metrics:
        return None, None, []

    result = metrics[0].get("result", {})
    drift_detected = result.get("dataset_drift")
    number_of_columns = result.get("number_of_columns") or 0
    drifted_columns = result.get("number_of_drifted_columns") or 0
    drift_share = None
    if number_of_columns:
        drift_share = drifted_columns / number_of_columns

    column_results: list[dict[str, Any]] = []
    drift_by_columns = result.get("drift_by_columns", {}) or {}
    for column_name, column_result in drift_by_columns.items():
        column_results.append(
            {
                "column": column_name,
                "drift_score": column_result.get("drift_score"),
                "drift_detected": column_result.get("drift_detected"),
                "method": column_result.get("stattest_name"),
            }
        )

    return drift_detected, drift_share, column_results


def analyze_prediction_drift(
    log_dir: Path = LOG_DIR,
    report_dir: Path = REPORT_DIR,
    reference_size: float = 0.5,
    min_rows: int = 20,
) -> DriftAnalysis:
    predictions = load_predictions(log_dir=Path(log_dir))
    dataset = build_drift_dataset(predictions)
    reference, current = split_reference_current(
        dataset, reference_size=reference_size, min_rows=min_rows
    )

    if len(reference) < min_rows or len(current) < min_rows:
        return DriftAnalysis(
            status="not_enough_data",
            reason=(
                f"Need at least {min_rows} rows in both windows to compute drift "
                f"(got reference={len(reference)}, current={len(current)})."
            ),
            reference_rows=len(reference),
            current_rows=len(current),
        )

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "drift_report.html"
    summary_path = report_dir / "drift_summary.json"

    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html(str(report_path))

        report_dict = report.as_dict()
        drift_detected, drift_share, column_results = _extract_evidently_results(report_dict)

        analysis = DriftAnalysis(
            status="ok",
            drift_detected=drift_detected,
            drift_share=round(drift_share, 4) if drift_share is not None else None,
            reference_rows=len(reference),
            current_rows=len(current),
            report_path=str(report_path),
            summary_path=str(summary_path),
            column_results=column_results,
        )
    except Exception as exc:
        analysis = _fallback_drift_analysis(reference, current)
        analysis.report_path = str(report_path)
        analysis.summary_path = str(summary_path)
        analysis.reason = f"{analysis.reason} Root cause: {exc}"

    summary_path.write_text(json.dumps(analysis.to_dict(), indent=2), encoding="utf-8")
    return analysis


def load_saved_summary(report_dir: Path = REPORT_DIR) -> dict[str, Any] | None:
    summary_path = Path(report_dir) / "drift_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate RecEngine prediction drift report.")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--reference-size", type=float, default=0.5)
    parser.add_argument("--min-rows", type=int, default=20)
    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()
    analysis = analyze_prediction_drift(
        log_dir=args.log_dir,
        report_dir=args.report_dir,
        reference_size=args.reference_size,
        min_rows=args.min_rows,
    )
    print(json.dumps(analysis.to_dict(), indent=2))


if __name__ == "__main__":
    main()
