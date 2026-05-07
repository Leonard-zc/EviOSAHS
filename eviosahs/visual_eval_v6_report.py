from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = ["Accuracy (%)", "Sensitivity (%)", "Specificity (%)", "F1-Score (%)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize V6 image-control metrics against F5 baseline.")
    parser.add_argument("--baseline-metrics", required=True, help="F5 baseline metrics.json.")
    parser.add_argument("--shuffle-metrics", required=True, help="V6 image-shuffle metrics.json.")
    parser.add_argument("--blur-metrics", required=True, help="V6 blur metrics.json.")
    parser.add_argument("--output-md", default="results/visual_audit/v6_summary.md")
    parser.add_argument("--output-csv", default="results/visual_audit/v6_summary.csv")
    return parser.parse_args()


def load_metrics(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    report = payload.get("report", {})
    return {
        "path": str(path),
        "total": payload.get("total"),
        "valid": payload.get("valid"),
        "tp": payload.get("tp"),
        "tn": payload.get("tn"),
        "fp": payload.get("fp"),
        "fn": payload.get("fn"),
        **{key: float(report.get(key, 0.0)) for key in METRIC_KEYS},
    }


def build_rows(baseline: dict[str, Any], controls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = [
        {
            "Setting": "F5 baseline",
            "Control": "original image",
            **{key: baseline[key] for key in METRIC_KEYS},
            "Delta Accuracy": 0.0,
            "Delta Sensitivity": 0.0,
            "Delta Specificity": 0.0,
            "Delta F1": 0.0,
            "TP": baseline["tp"],
            "TN": baseline["tn"],
            "FP": baseline["fp"],
            "FN": baseline["fn"],
        }
    ]
    for setting, metrics in controls:
        rows.append(
            {
                "Setting": setting,
                "Control": setting,
                **{key: metrics[key] for key in METRIC_KEYS},
                "Delta Accuracy": metrics["Accuracy (%)"] - baseline["Accuracy (%)"],
                "Delta Sensitivity": metrics["Sensitivity (%)"] - baseline["Sensitivity (%)"],
                "Delta Specificity": metrics["Specificity (%)"] - baseline["Specificity (%)"],
                "Delta F1": metrics["F1-Score (%)"] - baseline["F1-Score (%)"],
                "TP": metrics["tp"],
                "TN": metrics["tn"],
                "FP": metrics["fp"],
                "FN": metrics["fn"],
            }
        )
    return rows


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_md(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "Setting",
        "Accuracy (%)",
        "Sensitivity (%)",
        "Specificity (%)",
        "F1-Score (%)",
        "Delta Accuracy",
        "Delta Sensitivity",
        "Delta Specificity",
        "Delta F1",
        "TP",
        "TN",
        "FP",
        "FN",
    ]
    lines = ["# V6 Image-Control Summary", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row[column]) for column in columns) + " |")
    lines.append("")
    lines.append("Interpretation: a clear drop under image-shuffle or blur supports that the pipeline uses image information beyond clinical text alone.")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline = load_metrics(args.baseline_metrics)
    shuffle = load_metrics(args.shuffle_metrics)
    blur = load_metrics(args.blur_metrics)
    rows = build_rows(baseline, [("image-shuffle", shuffle), ("blur", blur)])
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
